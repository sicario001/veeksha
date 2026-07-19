// veeksha_native — a native (C++) streaming receive path exposed to Python.
//
// A single OS thread runs a poll() loop over `concurrency` sockets, parses SSE,
// and timestamps each chunk at true socket-read time (CLOCK_MONOTONIC). It
// returns, per completed request, the list of chunk arrival offsets in ms
// (relative to that request's send time). Python then computes drift metrics
// with the SAME code it uses for the Python pipeline, so native vs Python is a
// true apples-to-apples comparison.
//
// Declared `py::mod_gil_not_used()` so it is safe on free-threaded CPython and
// does not re-enable the GIL. The receive loop releases the GIL-equivalent while
// blocking in poll()/recv via py::gil_scoped_release semantics (a no-op benefit
// under free-threading, but correct either way).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Wall-clock (CLOCK_REALTIME) in ms — the domain the kernel stamps received
// packets in (SO_TIMESTAMP/SO_TIMESTAMPNS). Used for the RECEIVE side so send and
// receive timestamps share one clock; loop timing (deadlines/timeouts) stays on
// the monotonic now_ms().
static double now_realtime_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Ask the kernel to timestamp each received packet on arrival. Then recvmsg can
// read that stamp instead of taking a userspace clock when the loop finally
// reads the socket — eliminating the head-of-line lag where a chunk that arrived
// while the loop serviced other sockets would be stamped late.
static void enable_rx_timestamp(int fd) {
  int on = 1;
#if defined(SO_TIMESTAMPNS)
  setsockopt(fd, SOL_SOCKET, SO_TIMESTAMPNS, &on, sizeof(on));
#endif
#if defined(SO_TIMESTAMP)
  setsockopt(fd, SOL_SOCKET, SO_TIMESTAMP, &on, sizeof(on));
#endif
}

// recvmsg that extracts the kernel receive timestamp (ms, CLOCK_REALTIME domain).
// Falls back to a userspace realtime read when the kernel didn't attach one.
static ssize_t recv_ts(int fd, char* buf, size_t len, double* ts_ms) {
  struct iovec iov;
  iov.iov_base = buf;
  iov.iov_len = len;
  char control[512];
  struct msghdr msg;
  memset(&msg, 0, sizeof(msg));
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = control;
  msg.msg_controllen = sizeof(control);
  ssize_t n = recvmsg(fd, &msg, 0);
  double ts = -1.0;
  if (n > 0) {
    for (struct cmsghdr* cm = CMSG_FIRSTHDR(&msg); cm != nullptr;
         cm = CMSG_NXTHDR(&msg, cm)) {
      if (cm->cmsg_level != SOL_SOCKET) continue;
#if defined(SCM_TIMESTAMPNS)
      if (cm->cmsg_type == SCM_TIMESTAMPNS) {
        struct timespec t;
        memcpy(&t, CMSG_DATA(cm), sizeof(t));
        ts = t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
        break;
      }
#endif
#if defined(SCM_TIMESTAMP)
      if (cm->cmsg_type == SCM_TIMESTAMP) {
        struct timeval tv;
        memcpy(&tv, CMSG_DATA(cm), sizeof(tv));
        ts = tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
        break;
      }
#endif
    }
  }
  *ts_ms = (ts >= 0.0) ? ts : now_realtime_ms();
  return n;
}

struct Conn {
  bool connected = false;
  bool done = false;
  double send_time = 0.0;
  std::string inbuf;
  std::vector<double> offsets;  // chunk arrival offsets (ms) from send_time
};

static std::string build_request(const std::string& host, int num_chunks) {
  std::string body =
      "{\"model\":\"d\",\"stream\":true,\"max_completion_tokens\":" +
      std::to_string(num_chunks) + "}";
  std::string req = "POST /v1/chat/completions HTTP/1.1\r\n";
  req += "Host: " + host + "\r\n";
  req += "Content-Type: application/json\r\n";
  req += "Content-Length: " + std::to_string(body.size()) + "\r\n";
  req += "Connection: close\r\n\r\n";
  req += body;
  return req;
}

static int make_conn(const std::string& host, int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int flags = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  enable_rx_timestamp(fd);  // kernel-stamp arrivals for drift-free receive times
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
  int r = connect(fd, (struct sockaddr*)&addr, sizeof(addr));
  if (r < 0 && errno != EINPROGRESS) {
    close(fd);
    return -1;
  }
  return fd;
}

static void parse_lines(Conn& c) {
  double ts = now_ms();
  size_t pos;
  while ((pos = c.inbuf.find('\n')) != std::string::npos) {
    std::string line = c.inbuf.substr(0, pos);
    c.inbuf.erase(0, pos + 1);
    while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
      line.pop_back();
    if (line.rfind("data:", 0) == 0) {
      std::string d = line.substr(5);
      while (!d.empty() && d.front() == ' ') d.erase(0, 1);
      if (d == "[DONE]") {
        c.done = true;
        return;
      }
      c.offsets.push_back(ts - c.send_time);
    }
  }
}

// Returns one list of chunk-arrival offsets (ms) per completed request.
static std::vector<std::vector<double>> receive(const std::string& host, int port,
                                                int concurrency, int num_chunks,
                                                int total_requests,
                                                double timeout_s) {
  struct rlimit rl;
  getrlimit(RLIMIT_NOFILE, &rl);
  rl.rlim_cur = std::max<rlim_t>(rl.rlim_cur, (rlim_t)(concurrency + 64));
  if (rl.rlim_cur > rl.rlim_max) rl.rlim_cur = rl.rlim_max;
  setrlimit(RLIMIT_NOFILE, &rl);

  std::string request = build_request(host, num_chunks);
  std::unordered_map<int, Conn> conns;
  std::vector<std::vector<double>> finished;
  int launched = 0, completed = 0;
  double t0 = now_ms();

  auto try_launch = [&]() {
    while ((int)conns.size() < concurrency && launched < total_requests) {
      int fd = make_conn(host, port);
      if (fd < 0) break;
      Conn c;
      c.send_time = now_ms();
      conns.emplace(fd, std::move(c));
      launched++;
    }
  };
  try_launch();

  while (completed < total_requests && (now_ms() - t0) < timeout_s * 1000.0) {
    std::vector<struct pollfd> pfds;
    pfds.reserve(conns.size());
    for (auto& kv : conns) {
      struct pollfd p;
      p.fd = kv.first;
      p.events = kv.second.connected ? POLLIN : POLLOUT;
      p.revents = 0;
      pfds.push_back(p);
    }
    int n = poll(pfds.data(), pfds.size(), 50);
    if (n < 0) {
      if (errno == EINTR) continue;
      break;
    }
    std::vector<int> to_close;
    for (auto& p : pfds) {
      auto it = conns.find(p.fd);
      if (it == conns.end()) continue;
      Conn& c = it->second;
      if (!c.connected && (p.revents & (POLLOUT | POLLERR | POLLHUP))) {
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(p.fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
          to_close.push_back(p.fd);
          continue;
        }
        c.connected = true;
        c.send_time = now_ms();
        ssize_t w = send(p.fd, request.data(), request.size(), 0);
        (void)w;
      } else if (c.connected && (p.revents & POLLIN)) {
        char buf[16384];
        ssize_t r = recv(p.fd, buf, sizeof(buf), 0);
        if (r > 0) {
          c.inbuf.append(buf, r);
          parse_lines(c);
          if (c.done) to_close.push_back(p.fd);
        } else if (r == 0) {
          to_close.push_back(p.fd);
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          to_close.push_back(p.fd);
        }
      } else if (p.revents & (POLLERR | POLLHUP)) {
        to_close.push_back(p.fd);
      }
    }
    for (int fd : to_close) {
      auto it = conns.find(fd);
      if (it == conns.end()) continue;
      if (!it->second.offsets.empty())
        finished.push_back(std::move(it->second.offsets));
      close(fd);
      conns.erase(it);
      completed++;
    }
    try_launch();
  }
  for (auto& kv : conns) close(kv.first);
  return finished;
}

// Python-facing wrapper: releases the GIL around the blocking receive loop.
static std::vector<std::vector<double>> py_receive(const std::string& host, int port,
                                                   int concurrency, int num_chunks,
                                                   int total_requests,
                                                   double timeout_s) {
  py::gil_scoped_release release;
  return receive(host, port, concurrency, num_chunks, total_requests, timeout_s);
}

// ===========================================================================
// Real per-request receive engine (P1/P2): owns connection concurrency (one
// poll() loop over many sockets), sends caller-provided HTTP requests, and
// returns per-request results with the chunk timeline timestamped at true
// socket-read time. Python builds the request bytes (method/path/headers/body)
// and parses the returned content; native owns transport + timing.
// ===========================================================================

struct ReqResult {
  int index = -1;
  int status = 0;
  std::string content;             // concatenated SSE data payloads (sans [DONE])
  std::vector<double> offsets_ms;  // per-event arrival offset (ms from send)
  std::vector<int> sizes;          // per-event payload size (bytes)
  double dispatch_offset_ms = 0.0;  // actual launch time (ms from run start)
  std::string error;
};

struct EngineConn {
  int index = -1;
  bool connected = false;
  bool headers_done = false;
  bool done = false;
  double send_time = 0.0;
  int status = 0;
  std::string header_buf;
  std::string inbuf;
  std::string content;
  std::vector<double> offsets;
  std::vector<int> sizes;
  bool sse = true;
};

// Parse the status line + headers out of the front of the stream. Returns true
// once the CRLFCRLF header terminator has been seen; leftover bytes are the body.
static void parse_headers(EngineConn& c) {
  size_t term = c.header_buf.find("\r\n\r\n");
  if (term == std::string::npos) return;
  std::string head = c.header_buf.substr(0, term);
  std::string rest = c.header_buf.substr(term + 4);
  // status line: "HTTP/1.1 200 OK"
  size_t sp = head.find(' ');
  if (sp != std::string::npos) {
    c.status = atoi(head.c_str() + sp + 1);
  }
  c.headers_done = true;
  c.inbuf = rest;  // remaining bytes belong to the body
}

// Scan the body for SSE `data:` lines, stamping each with `ts` — the kernel
// receive time of the read that delivered these bytes (see recv_ts), so the
// offset reflects true arrival, not when the loop got around to parsing.
static void parse_body_sse(EngineConn& c, double ts) {
  size_t pos;
  while ((pos = c.inbuf.find('\n')) != std::string::npos) {
    std::string line = c.inbuf.substr(0, pos);
    c.inbuf.erase(0, pos + 1);
    while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
      line.pop_back();
    if (line.rfind("data:", 0) == 0) {
      std::string d = line.substr(5);
      while (!d.empty() && d.front() == ' ') d.erase(0, 1);
      if (d == "[DONE]") {
        c.done = true;
        return;
      }
      c.offsets.push_back(ts - c.send_time);
      c.sizes.push_back((int)d.size());
      c.content += d;
    }
  }
}

// One shard's poll() loop: services the request indices in `indices` with up to
// `concurrency` in-flight, writing into results[idx] (disjoint per shard, so
// threads never touch the same element — no hot-path locks). `t0` is shared
// across shards so open-loop arrival deadlines line up on one global clock.
static void run_batch_worker(const std::string& host, int port,
                             const std::vector<std::string>& requests,
                             const std::vector<int>& indices, int concurrency,
                             double timeout_s, bool sse,
                             const std::vector<double>& dispatch_offsets_ms,
                             double t0, std::vector<ReqResult>& results) {
  int total = (int)indices.size();
  std::unordered_map<int, EngineConn> conns;
  size_t cur = 0;
  int completed = 0;
  bool open_loop = !dispatch_offsets_ms.empty();

  auto try_launch = [&]() -> double {
    double now = now_ms() - t0;
    while ((int)conns.size() < concurrency && cur < indices.size()) {
      int idx = indices[cur];
      if (open_loop && dispatch_offsets_ms[idx] > now + 0.05) {
        return dispatch_offsets_ms[idx] - now;  // not due yet
      }
      int fd = make_conn(host, port);
      if (fd < 0) {
        results[idx].error = "connect failed";
        completed++;
        cur++;
        continue;
      }
      EngineConn c;
      c.index = idx;
      c.sse = sse;
      results[idx].dispatch_offset_ms = now_ms() - t0;
      conns.emplace(fd, std::move(c));
      cur++;
    }
    return 1e9;
  };
  double next_dispatch = try_launch();

  while (completed < total && (now_ms() - t0) < timeout_s * 1000.0) {
    std::vector<struct pollfd> pfds;
    pfds.reserve(conns.size());
    for (auto& kv : conns) {
      struct pollfd p;
      p.fd = kv.first;
      p.events = kv.second.connected ? POLLIN : POLLOUT;
      p.revents = 0;
      pfds.push_back(p);
    }
    int poll_ms = 50;
    if (open_loop && next_dispatch < poll_ms) poll_ms = (int)std::max(0.0, next_dispatch);
    int n = poll(pfds.data(), pfds.size(), poll_ms);
    if (n < 0) {
      if (errno == EINTR) continue;
      break;
    }
    std::vector<int> to_close;
    for (auto& p : pfds) {
      auto it = conns.find(p.fd);
      if (it == conns.end()) continue;
      EngineConn& c = it->second;
      if (!c.connected && (p.revents & (POLLOUT | POLLERR | POLLHUP))) {
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(p.fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
          results[c.index].error = "connect error";
          to_close.push_back(p.fd);
          continue;
        }
        c.connected = true;
        c.send_time = now_realtime_ms();  // realtime: matches kernel rx stamps
        const std::string& req = requests[c.index];
        ssize_t w = send(p.fd, req.data(), req.size(), 0);
        (void)w;
      } else if (c.connected && (p.revents & POLLIN)) {
        char buf[16384];
        double kts;  // kernel receive timestamp for this read
        ssize_t r = recv_ts(p.fd, buf, sizeof(buf), &kts);
        if (r > 0) {
          if (!c.headers_done) {
            c.header_buf.append(buf, r);
            parse_headers(c);
            if (c.headers_done && c.sse) parse_body_sse(c, kts);
            else if (c.headers_done) {
              c.content += c.inbuf;
              c.inbuf.clear();
            }
          } else if (c.sse) {
            c.inbuf.append(buf, r);
            parse_body_sse(c, kts);
          } else {
            c.content.append(buf, r);
          }
          if (c.done) to_close.push_back(p.fd);
        } else if (r == 0) {
          to_close.push_back(p.fd);
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          results[c.index].error = "recv error";
          to_close.push_back(p.fd);
        }
      } else if (p.revents & (POLLERR | POLLHUP)) {
        to_close.push_back(p.fd);
      }
    }
    for (int fd : to_close) {
      auto it = conns.find(fd);
      if (it == conns.end()) continue;
      EngineConn& c = it->second;
      ReqResult& res = results[c.index];
      res.status = c.status;
      res.content = std::move(c.content);
      res.offsets_ms = std::move(c.offsets);
      res.sizes = std::move(c.sizes);
      if (!c.sse) {
        res.offsets_ms.push_back(now_realtime_ms() - c.send_time);
        res.sizes.push_back((int)res.content.size());
      }
      close(fd);
      conns.erase(it);
      completed++;
    }
    next_dispatch = try_launch();
  }
  for (auto& kv : conns) {
    ReqResult& res = results[kv.second.index];
    if (res.error.empty()) res.error = "timeout";
    close(kv.first);
  }
}

// Public engine: shards the requests across `num_threads` native poll-loop
// threads (each its own loop over a disjoint socket set) and merges. On
// free-threaded CPython the loops run in true parallel (no Python in the hot
// path), so one native process scales past a single loop's CPU limit — the
// standard sharded-reactor pattern (nginx/envoy). `num_threads<=1` = one loop.
static std::vector<ReqResult> run_batch(const std::string& host, int port,
                                        const std::vector<std::string>& requests,
                                        int concurrency, double timeout_s, bool sse,
                                        const std::vector<double>& dispatch_offsets_ms,
                                        int num_threads) {
  struct rlimit rl;
  getrlimit(RLIMIT_NOFILE, &rl);
  rl.rlim_cur = std::max<rlim_t>(rl.rlim_cur, (rlim_t)(concurrency + 64));
  if (rl.rlim_cur > rl.rlim_max) rl.rlim_cur = rl.rlim_max;
  setrlimit(RLIMIT_NOFILE, &rl);

  int total = (int)requests.size();
  std::vector<ReqResult> results(total);
  for (int i = 0; i < total; i++) results[i].index = i;
  if (total == 0) return results;

  int nthreads = std::max(1, num_threads);
  if (nthreads > total) nthreads = total;
  double t0 = now_ms();

  if (nthreads == 1) {
    std::vector<int> all(total);
    for (int i = 0; i < total; i++) all[i] = i;
    run_batch_worker(host, port, requests, all, concurrency, timeout_s, sse,
                     dispatch_offsets_ms, t0, results);
    return results;
  }

  // Strided sharding spreads the arrival schedule evenly across threads (thread
  // t owns indices t, t+N, t+2N, ...). Split the in-flight budget across shards.
  std::vector<std::vector<int>> shards(nthreads);
  for (int i = 0; i < total; i++) shards[i % nthreads].push_back(i);
  int per_thread_conc = (concurrency + nthreads - 1) / nthreads;
  if (per_thread_conc < 1) per_thread_conc = 1;

  std::vector<std::thread> pool;
  pool.reserve(nthreads);
  for (int t = 0; t < nthreads; t++) {
    pool.emplace_back([&, t]() {
      run_batch_worker(host, port, requests, shards[t], per_thread_conc,
                       timeout_s, sse, dispatch_offsets_ms, t0, results);
    });
  }
  for (auto& th : pool) th.join();
  return results;
}

static std::vector<ReqResult> py_run_batch(
    const std::string& host, int port,
    const std::vector<std::string>& requests, int concurrency, double timeout_s,
    bool sse, const std::vector<double>& dispatch_offsets_ms, int num_threads) {
  py::gil_scoped_release release;
  return run_batch(host, port, requests, concurrency, timeout_s, sse,
                   dispatch_offsets_ms, num_threads);
}

// ===========================================================================
// Native receive->dispatch coupling (P6): closed-loop / dependent workloads.
// Each "chain" is a sequence of turns (turn N+1 depends on turn N completing).
// Native runs many chains concurrently; the instant a turn completes it fires
// the chain's next turn itself — Python is never on the receive->dispatch
// critical path — and records the handoff latency (turn-complete -> next-send).
// This is the coupling the docs (analysis/11 correction) require to be native so
// closed-loop dispatch timing isn't corrupted by Python queue jitter.
// ===========================================================================

struct ChainTurn {
  std::vector<double> offsets_ms;
  std::string content;
  int status = 0;
};

struct ChainResult {
  int index = -1;
  std::vector<ChainTurn> turns;
  std::vector<double> handoff_ms;  // per-inter-turn: complete -> next send
  std::string error;
};

struct ChainConn {
  int chain = -1;
  size_t turn = 0;
  bool connected = false;
  bool headers_done = false;
  bool done = false;
  double send_time = 0.0;
  int status = 0;
  std::string header_buf;
  std::string inbuf;
  std::string content;
  std::vector<double> offsets;
};

static std::vector<ChainResult> run_chains(
    const std::string& host, int port,
    const std::vector<std::vector<std::string>>& chains, int concurrency,
    double timeout_s) {
  struct rlimit rl;
  getrlimit(RLIMIT_NOFILE, &rl);
  rl.rlim_cur = std::max<rlim_t>(rl.rlim_cur, (rlim_t)(concurrency + 64));
  if (rl.rlim_cur > rl.rlim_max) rl.rlim_cur = rl.rlim_max;
  setrlimit(RLIMIT_NOFILE, &rl);

  int total = (int)chains.size();
  std::vector<ChainResult> results(total);
  for (int i = 0; i < total; i++) results[i].index = i;

  std::unordered_map<int, ChainConn> conns;
  std::vector<double> last_complete(total, 0.0);  // when a chain's last turn ended
  int next_chain = 0;
  int chains_done = 0;

  // Launch turn `turn` of chain `ch` on a fresh connection.
  auto launch_turn = [&](int ch, size_t turn) -> bool {
    int fd = make_conn(host, port);
    if (fd < 0) {
      results[ch].error = "connect failed";
      return false;
    }
    ChainConn c;
    c.chain = ch;
    c.turn = turn;
    conns.emplace(fd, std::move(c));
    return true;
  };

  auto start_next_chain = [&]() {
    while ((int)conns.size() < concurrency && next_chain < total) {
      int ch = next_chain++;
      if (chains[ch].empty()) {
        chains_done++;
        continue;
      }
      launch_turn(ch, 0);
    }
  };
  start_next_chain();

  double t0 = now_ms();
  while (chains_done < total && (now_ms() - t0) < timeout_s * 1000.0) {
    std::vector<struct pollfd> pfds;
    pfds.reserve(conns.size());
    for (auto& kv : conns) {
      struct pollfd pp;
      pp.fd = kv.first;
      pp.events = kv.second.connected ? POLLIN : POLLOUT;
      pp.revents = 0;
      pfds.push_back(pp);
    }
    if (pfds.empty()) break;
    int n = poll(pfds.data(), pfds.size(), 50);
    if (n < 0) {
      if (errno == EINTR) continue;
      break;
    }
    std::vector<int> to_close;
    for (auto& pp : pfds) {
      auto it = conns.find(pp.fd);
      if (it == conns.end()) continue;
      ChainConn& c = it->second;
      if (!c.connected && (pp.revents & (POLLOUT | POLLERR | POLLHUP))) {
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(pp.fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
          results[c.chain].error = "connect error";
          chains_done++;
          to_close.push_back(pp.fd);
          continue;
        }
        c.connected = true;
        c.send_time = now_realtime_ms();
        const std::string& req = chains[c.chain][c.turn];
        send(pp.fd, req.data(), req.size(), 0);
      } else if (c.connected && (pp.revents & POLLIN)) {
        char buf[16384];
        double kts;
        ssize_t r = recv_ts(pp.fd, buf, sizeof(buf), &kts);
        if (r > 0) {
          if (!c.headers_done) {
            c.header_buf.append(buf, r);
            size_t term = c.header_buf.find("\r\n\r\n");
            if (term != std::string::npos) {
              size_t sp = c.header_buf.find(' ');
              if (sp != std::string::npos) c.status = atoi(c.header_buf.c_str() + sp + 1);
              c.inbuf = c.header_buf.substr(term + 4);
              c.headers_done = true;
              EngineConn tmp;  // reuse the SSE line parser
              tmp.send_time = c.send_time;
              tmp.inbuf = c.inbuf;
              parse_body_sse(tmp, kts);
              c.inbuf = tmp.inbuf;
              c.content += tmp.content;
              for (double o : tmp.offsets) c.offsets.push_back(o);
              if (tmp.done) c.done = true;
            }
          } else {
            EngineConn tmp;
            tmp.send_time = c.send_time;
            tmp.inbuf = c.inbuf;
            tmp.inbuf.append(buf, r);
            parse_body_sse(tmp, kts);
            c.inbuf = tmp.inbuf;
            c.content += tmp.content;
            for (double o : tmp.offsets) c.offsets.push_back(o);
            if (tmp.done) c.done = true;
          }
          if (c.done) to_close.push_back(pp.fd);
        } else if (r == 0) {
          to_close.push_back(pp.fd);
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          results[c.chain].error = "recv error";
          to_close.push_back(pp.fd);
        }
      } else if (pp.revents & (POLLERR | POLLHUP)) {
        to_close.push_back(pp.fd);
      }
    }
    for (int fd : to_close) {
      auto it = conns.find(fd);
      if (it == conns.end()) continue;
      ChainConn c = std::move(it->second);
      close(fd);
      conns.erase(it);

      double complete_at = now_ms();
      ChainTurn ct;
      ct.offsets_ms = std::move(c.offsets);
      ct.content = std::move(c.content);
      ct.status = c.status;
      results[c.chain].turns.push_back(std::move(ct));

      size_t next_turn = c.turn + 1;
      if (next_turn < chains[c.chain].size()) {
        // COUPLING: fire the dependent next turn immediately, in native, and
        // record the handoff latency (no Python between receive and dispatch).
        last_complete[c.chain] = complete_at;
        if (launch_turn(c.chain, next_turn)) {
          // the new conn's send happens on POLLOUT; approximate handoff as the
          // time from completion to the connect being issued (native-side).
          results[c.chain].handoff_ms.push_back(now_ms() - complete_at);
        } else {
          chains_done++;
        }
      } else {
        chains_done++;
      }
    }
    start_next_chain();
  }
  for (auto& kv : conns) close(kv.first);
  return results;
}

static std::vector<ChainResult> py_run_chains(
    const std::string& host, int port,
    const std::vector<std::vector<std::string>>& chains, int concurrency,
    double timeout_s) {
  py::gil_scoped_release release;
  return run_chains(host, port, chains, concurrency, timeout_s);
}

// ===========================================================================
// Native WebSocket receive + framed send (P3): the interactivity-critical path.
// Connects, performs the HTTP Upgrade handshake, sends caller-provided text
// messages as masked frames, and reads server text/binary frames — timestamping
// each at true socket-read time. Owns concurrency over one poll() loop like the
// SSE engine, so per-frame receive timing is measured with no Python per event.
// ===========================================================================

struct WsResult {
  int index = -1;
  std::vector<double> offsets_ms;  // per data-frame arrival offset (ms from send)
  std::vector<int> sizes;          // per data-frame payload size (bytes)
  std::string content;             // concatenated text/binary payloads
  std::vector<std::string> frames;  // per data-frame payload (for protocol parse)
  std::vector<double> sent_offsets_ms;  // actual send offset of each paced message
  std::string error;
};

struct WsConn {
  int index = -1;
  bool connected = false;
  bool handshaken = false;
  bool sent_init = false;
  bool done = false;
  double send_time = 0.0;   // when the first message was (or should be) sent
  size_t next_send = 0;     // next paced message index to send
  std::string inbuf;
  std::vector<double> offsets;
  std::vector<int> sizes;
  std::string content;
  std::vector<std::string> frames;
  std::vector<double> sent_offsets;
};

// Encode one client->server text frame (opcode 0x1), masked per RFC 6455.
static std::string ws_encode_text(const std::string& payload) {
  std::string frame;
  frame.push_back((char)0x81);  // FIN + text
  size_t n = payload.size();
  if (n < 126) {
    frame.push_back((char)(0x80 | n));  // MASK bit + len
  } else if (n <= 0xFFFF) {
    frame.push_back((char)(0x80 | 126));
    frame.push_back((char)((n >> 8) & 0xFF));
    frame.push_back((char)(n & 0xFF));
  } else {
    frame.push_back((char)(0x80 | 127));
    for (int i = 7; i >= 0; i--) frame.push_back((char)((n >> (8 * i)) & 0xFF));
  }
  // Fixed mask key: masking is required by the RFC, but its value need only be
  // present, not random, for protocol correctness.
  const unsigned char mask[4] = {0x12, 0x34, 0x56, 0x78};
  for (int i = 0; i < 4; i++) frame.push_back((char)mask[i]);
  for (size_t i = 0; i < n; i++)
    frame.push_back((char)(payload[i] ^ mask[i % 4]));
  return frame;
}

// Pull complete server frames out of inbuf, timestamping data frames at read
// time. Handles fragmentation across recv() boundaries and 126/127 extended
// lengths. Sets c.done on a close frame (0x8).
static void ws_parse_frames(WsConn& c) {
  double ts = now_ms();
  while (true) {
    if (c.inbuf.size() < 2) return;
    const unsigned char* p = (const unsigned char*)c.inbuf.data();
    unsigned char b0 = p[0], b1 = p[1];
    int opcode = b0 & 0x0F;
    bool masked = b1 & 0x80;  // server->client must be unmasked
    uint64_t len = b1 & 0x7F;
    size_t offset = 2;
    if (len == 126) {
      if (c.inbuf.size() < 4) return;
      len = ((uint64_t)p[2] << 8) | p[3];
      offset = 4;
    } else if (len == 127) {
      if (c.inbuf.size() < 10) return;
      len = 0;
      for (int i = 0; i < 8; i++) len = (len << 8) | p[2 + i];
      offset = 10;
    }
    size_t mask_len = masked ? 4 : 0;
    if (c.inbuf.size() < offset + mask_len + len) return;  // wait for full frame
    std::string payload = c.inbuf.substr(offset + mask_len, len);
    if (masked) {
      const unsigned char* mk = p + offset;
      for (size_t i = 0; i < payload.size(); i++) payload[i] ^= mk[i % 4];
    }
    c.inbuf.erase(0, offset + mask_len + len);
    if (opcode == 0x8) {  // close
      c.done = true;
      return;
    }
    if (opcode == 0x1 || opcode == 0x2) {  // text / binary data frame
      c.offsets.push_back(ts - c.send_time);
      c.sizes.push_back((int)payload.size());
      c.content += payload;
      c.frames.push_back(payload);
    }
    // ping/pong/continuation: ignored for timing purposes.
  }
}

static std::string ws_handshake(const std::string& host, int port,
                                const std::string& path) {
  std::string req = "GET " + path + " HTTP/1.1\r\n";
  req += "Host: " + host + ":" + std::to_string(port) + "\r\n";
  req += "Upgrade: websocket\r\n";
  req += "Connection: Upgrade\r\n";
  req += "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
  req += "Sec-WebSocket-Version: 13\r\n\r\n";
  return req;
}

// Send any paced messages whose absolute deadline has arrived. Records the
// actual send offset (ms from the pacing base) for each dispatched message —
// the per-dispatch send-drift the interactivity path is sensitive to. Returns
// the ms until the next pending deadline (or a large value if none pending).
static double ws_pump_sends(int fd, WsConn& c,
                            const std::vector<std::string>& messages,
                            const std::vector<double>& send_offsets_ms) {
  const double FAR = 1e9;
  if (c.next_send >= messages.size()) return FAR;
  double now = now_ms();
  while (c.next_send < messages.size()) {
    double due =
        send_offsets_ms.empty() ? 0.0 : send_offsets_ms[c.next_send];
    double deadline = c.send_time + due;
    if (now + 0.05 >= deadline) {
      std::string frame = ws_encode_text(messages[c.next_send]);
      send(fd, frame.data(), frame.size(), 0);
      c.sent_offsets.push_back(now - c.send_time);
      c.next_send++;
      now = now_ms();
    } else {
      return deadline - now;  // wait until this message is due
    }
  }
  return FAR;
}

static std::vector<WsResult> ws_stream(const std::string& host, int port,
                                       const std::string& path,
                                       const std::vector<std::string>& init_messages,
                                       int concurrency, double timeout_s,
                                       const std::vector<double>& send_offsets_ms) {
  struct rlimit rl;
  getrlimit(RLIMIT_NOFILE, &rl);
  rl.rlim_cur = std::max<rlim_t>(rl.rlim_cur, (rlim_t)(concurrency + 64));
  if (rl.rlim_cur > rl.rlim_max) rl.rlim_cur = rl.rlim_max;
  setrlimit(RLIMIT_NOFILE, &rl);

  int total = concurrency;  // one connection per concurrency slot
  std::vector<WsResult> results(total);
  for (int i = 0; i < total; i++) results[i].index = i;
  std::string handshake = ws_handshake(host, port, path);

  std::unordered_map<int, WsConn> conns;
  for (int i = 0; i < total; i++) {
    int fd = make_conn(host, port);
    if (fd < 0) {
      results[i].error = "connect failed";
      continue;
    }
    WsConn c;
    c.index = i;
    conns.emplace(fd, std::move(c));
  }

  int completed = 0;
  double t0 = now_ms();
  while (completed < (int)conns.size() && (now_ms() - t0) < timeout_s * 1000.0) {
    std::vector<struct pollfd> pfds;
    pfds.reserve(conns.size());
    double next_wake = 50.0;  // default poll timeout (ms)
    for (auto& kv : conns) {
      if (kv.second.done) continue;
      struct pollfd pp;
      pp.fd = kv.first;
      pp.events = kv.second.connected ? POLLIN : POLLOUT;
      pp.revents = 0;
      pfds.push_back(pp);
      // pump any due paced sends and shrink the poll timeout to the next deadline
      if (kv.second.handshaken) {
        double wait = ws_pump_sends(kv.first, kv.second, init_messages,
                                    send_offsets_ms);
        if (wait < next_wake) next_wake = wait;
      }
    }
    if (pfds.empty()) break;
    int timeout_ms = (int)std::max(0.0, std::min(50.0, next_wake));
    int n = poll(pfds.data(), pfds.size(), timeout_ms);
    if (n < 0) {
      if (errno == EINTR) continue;
      break;
    }
    std::vector<int> to_close;
    for (auto& pp : pfds) {
      auto it = conns.find(pp.fd);
      if (it == conns.end()) continue;
      WsConn& c = it->second;
      if (!c.connected && (pp.revents & (POLLOUT | POLLERR | POLLHUP))) {
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(pp.fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
          results[c.index].error = "connect error";
          to_close.push_back(pp.fd);
          continue;
        }
        c.connected = true;
        send(pp.fd, handshake.data(), handshake.size(), 0);
      } else if (c.connected && (pp.revents & POLLIN)) {
        char buf[16384];
        ssize_t r = recv(pp.fd, buf, sizeof(buf), 0);
        if (r > 0) {
          c.inbuf.append(buf, r);
          if (!c.handshaken) {
            size_t term = c.inbuf.find("\r\n\r\n");
            if (term != std::string::npos) {
              c.inbuf.erase(0, term + 4);  // drop the 101 response headers
              c.handshaken = true;
              c.send_time = now_ms();  // pacing base = handshake completion
            }
          }
          if (c.handshaken)
            ws_pump_sends(pp.fd, c, init_messages, send_offsets_ms);
          if (c.handshaken) ws_parse_frames(c);
          if (c.done) to_close.push_back(pp.fd);
        } else if (r == 0) {
          to_close.push_back(pp.fd);
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          results[c.index].error = "recv error";
          to_close.push_back(pp.fd);
        }
      } else if (pp.revents & (POLLERR | POLLHUP)) {
        to_close.push_back(pp.fd);
      }
    }
    for (int fd : to_close) {
      auto it = conns.find(fd);
      if (it == conns.end()) continue;
      WsConn& c = it->second;
      WsResult& res = results[c.index];
      res.offsets_ms = std::move(c.offsets);
      res.sizes = std::move(c.sizes);
      res.content = std::move(c.content);
      res.frames = std::move(c.frames);
      res.sent_offsets_ms = std::move(c.sent_offsets);
      close(fd);
      conns.erase(it);
      completed++;
    }
  }
  for (auto& kv : conns) {
    WsResult& res = results[kv.second.index];
    // capture whatever arrived before timeout
    if (res.offsets_ms.empty() && !kv.second.offsets.empty()) {
      res.offsets_ms = std::move(kv.second.offsets);
      res.sizes = std::move(kv.second.sizes);
      res.content = std::move(kv.second.content);
      res.frames = std::move(kv.second.frames);
    }
    if (res.sent_offsets_ms.empty() && !kv.second.sent_offsets.empty())
      res.sent_offsets_ms = std::move(kv.second.sent_offsets);
    if (res.error.empty() && res.offsets_ms.empty()) res.error = "timeout";
    close(kv.first);
  }
  return results;
}

// Per-connection WS batch with a concurrency cap + refill (mirrors run_batch):
// each request has its own message sequence + send schedule, so N distinct audio
// requests (different text / audio) run concurrently over one poll() loop.
static std::vector<WsResult> ws_run_batch(
    const std::string& host, int port, const std::string& path,
    const std::vector<std::vector<std::string>>& req_messages,
    const std::vector<std::vector<double>>& req_offsets, int concurrency,
    double timeout_s) {
  struct rlimit rl;
  getrlimit(RLIMIT_NOFILE, &rl);
  rl.rlim_cur = std::max<rlim_t>(rl.rlim_cur, (rlim_t)(concurrency + 64));
  if (rl.rlim_cur > rl.rlim_max) rl.rlim_cur = rl.rlim_max;
  setrlimit(RLIMIT_NOFILE, &rl);

  int total = (int)req_messages.size();
  std::vector<WsResult> results(total);
  for (int i = 0; i < total; i++) results[i].index = i;
  std::string handshake = ws_handshake(host, port, path);

  std::unordered_map<int, WsConn> conns;
  int launched = 0, completed = 0;

  auto try_launch = [&]() {
    while ((int)conns.size() < concurrency && launched < total) {
      int fd = make_conn(host, port);
      if (fd < 0) {
        results[launched].error = "connect failed";
        completed++;
        launched++;
        continue;
      }
      WsConn c;
      c.index = launched;
      conns.emplace(fd, std::move(c));
      launched++;
    }
  };
  try_launch();

  double t0 = now_ms();
  while (completed < total && (now_ms() - t0) < timeout_s * 1000.0) {
    std::vector<struct pollfd> pfds;
    pfds.reserve(conns.size());
    double next_wake = 50.0;
    for (auto& kv : conns) {
      if (kv.second.done) continue;
      struct pollfd pp;
      pp.fd = kv.first;
      pp.events = kv.second.connected ? POLLIN : POLLOUT;
      pp.revents = 0;
      pfds.push_back(pp);
      if (kv.second.handshaken) {
        double wait = ws_pump_sends(kv.first, kv.second, req_messages[kv.second.index],
                                    req_offsets[kv.second.index]);
        if (wait < next_wake) next_wake = wait;
      }
    }
    if (pfds.empty()) break;
    int timeout_ms = (int)std::max(0.0, std::min(50.0, next_wake));
    int n = poll(pfds.data(), pfds.size(), timeout_ms);
    if (n < 0) {
      if (errno == EINTR) continue;
      break;
    }
    std::vector<int> to_close;
    for (auto& pp : pfds) {
      auto it = conns.find(pp.fd);
      if (it == conns.end()) continue;
      WsConn& c = it->second;
      if (!c.connected && (pp.revents & (POLLOUT | POLLERR | POLLHUP))) {
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(pp.fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
          results[c.index].error = "connect error";
          to_close.push_back(pp.fd);
          continue;
        }
        c.connected = true;
        send(pp.fd, handshake.data(), handshake.size(), 0);
      } else if (c.connected && (pp.revents & POLLIN)) {
        char buf[16384];
        ssize_t r = recv(pp.fd, buf, sizeof(buf), 0);
        if (r > 0) {
          c.inbuf.append(buf, r);
          if (!c.handshaken) {
            size_t term = c.inbuf.find("\r\n\r\n");
            if (term != std::string::npos) {
              c.inbuf.erase(0, term + 4);
              c.handshaken = true;
              c.send_time = now_ms();
            }
          }
          if (c.handshaken)
            ws_pump_sends(pp.fd, c, req_messages[c.index], req_offsets[c.index]);
          if (c.handshaken) ws_parse_frames(c);
          if (c.done) to_close.push_back(pp.fd);
        } else if (r == 0) {
          to_close.push_back(pp.fd);
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          results[c.index].error = "recv error";
          to_close.push_back(pp.fd);
        }
      } else if (pp.revents & (POLLERR | POLLHUP)) {
        to_close.push_back(pp.fd);
      }
    }
    for (int fd : to_close) {
      auto it = conns.find(fd);
      if (it == conns.end()) continue;
      WsConn& c = it->second;
      WsResult& res = results[c.index];
      res.offsets_ms = std::move(c.offsets);
      res.sizes = std::move(c.sizes);
      res.content = std::move(c.content);
      res.frames = std::move(c.frames);
      res.sent_offsets_ms = std::move(c.sent_offsets);
      close(fd);
      conns.erase(it);
      completed++;
    }
    try_launch();
  }
  for (auto& kv : conns) {
    WsResult& res = results[kv.second.index];
    if (res.offsets_ms.empty() && !kv.second.offsets.empty()) {
      res.offsets_ms = std::move(kv.second.offsets);
      res.sizes = std::move(kv.second.sizes);
      res.content = std::move(kv.second.content);
      res.frames = std::move(kv.second.frames);
    }
    if (res.error.empty() && res.offsets_ms.empty()) res.error = "timeout";
    close(kv.first);
  }
  return results;
}

static std::vector<WsResult> py_ws_run_batch(
    const std::string& host, int port, const std::string& path,
    const std::vector<std::vector<std::string>>& req_messages,
    const std::vector<std::vector<double>>& req_offsets, int concurrency,
    double timeout_s) {
  py::gil_scoped_release release;
  return ws_run_batch(host, port, path, req_messages, req_offsets, concurrency,
                      timeout_s);
}

static std::vector<WsResult> py_ws_stream(
    const std::string& host, int port, const std::string& path,
    const std::vector<std::string>& init_messages, int concurrency,
    double timeout_s, const std::vector<double>& send_offsets_ms) {
  py::gil_scoped_release release;
  return ws_stream(host, port, path, init_messages, concurrency, timeout_s,
                   send_offsets_ms);
}

PYBIND11_MODULE(veeksha_native, m, py::mod_gil_not_used()) {
  m.doc() = "Native (C++) streaming receive path for Veeksha.";
  m.def("receive", &py_receive, py::arg("host"), py::arg("port"),
        py::arg("concurrency"), py::arg("num_chunks"), py::arg("total_requests"),
        py::arg("timeout_s") = 120.0,
        "Probe: single-thread native receive loop; per-request chunk offsets.");

  py::class_<ReqResult>(m, "ReqResult")
      .def_readonly("index", &ReqResult::index)
      .def_readonly("status", &ReqResult::status)
      .def_readonly("content", &ReqResult::content)
      .def_readonly("offsets_ms", &ReqResult::offsets_ms)
      .def_readonly("sizes", &ReqResult::sizes)
      .def_readonly("dispatch_offset_ms", &ReqResult::dispatch_offset_ms)
      .def_readonly("error", &ReqResult::error);

  m.def("run_batch", &py_run_batch, py::arg("host"), py::arg("port"),
        py::arg("requests"), py::arg("concurrency"), py::arg("timeout_s") = 120.0,
        py::arg("sse") = true,
        py::arg("dispatch_offsets_ms") = std::vector<double>(),
        py::arg("num_threads") = 1,
        "Real per-request engine: owns connection concurrency over poll() "
        "loop(s), sends caller-built HTTP requests, returns per-request ReqResult "
        "(status, content, kernel-time chunk offsets_ms + sizes, actual "
        "dispatch_offset_ms). With dispatch_offsets_ms it runs OPEN-LOOP: each "
        "request is launched on its arrival deadline (concurrency = max in-flight "
        "cap) so native owns the arrival-dispatch timing too. num_threads>1 shards "
        "the connections across that many native poll-loop threads (true parallel "
        "on free-threaded CPython), splitting the in-flight budget across shards.");

  py::class_<ChainTurn>(m, "ChainTurn")
      .def_readonly("offsets_ms", &ChainTurn::offsets_ms)
      .def_readonly("content", &ChainTurn::content)
      .def_readonly("status", &ChainTurn::status);

  py::class_<ChainResult>(m, "ChainResult")
      .def_readonly("index", &ChainResult::index)
      .def_readonly("turns", &ChainResult::turns)
      .def_readonly("handoff_ms", &ChainResult::handoff_ms)
      .def_readonly("error", &ChainResult::error);

  m.def("run_chains", &py_run_chains, py::arg("host"), py::arg("port"),
        py::arg("chains"), py::arg("concurrency"), py::arg("timeout_s") = 120.0,
        "Closed-loop coupling: each chain is a sequence of dependent turns; "
        "native fires each next turn the instant the prior completes (no Python "
        "on the receive->dispatch path) and records per-turn timelines + the "
        "inter-turn handoff latency.");

  py::class_<WsResult>(m, "WsResult")
      .def_readonly("index", &WsResult::index)
      .def_readonly("offsets_ms", &WsResult::offsets_ms)
      .def_readonly("sizes", &WsResult::sizes)
      .def_readonly("content", &WsResult::content)
      .def_readonly("frames", &WsResult::frames)
      .def_readonly("sent_offsets_ms", &WsResult::sent_offsets_ms)
      .def_readonly("error", &WsResult::error);

  m.def("ws_stream", &py_ws_stream, py::arg("host"), py::arg("port"),
        py::arg("path"), py::arg("init_messages"), py::arg("concurrency"),
        py::arg("timeout_s") = 120.0,
        py::arg("send_offsets_ms") = std::vector<double>(),
        "Native WebSocket receive + timer-wheel send pacing: handshake, send each "
        "init message as a masked frame on its absolute deadline (send_offsets_ms, "
        "ms from handshake), read server frames timestamped at socket-read time. "
        "Returns one WsResult per connection (per-frame offsets_ms + sizes + "
        "content, and sent_offsets_ms = the actual paced send times).");

  m.def("ws_run_batch", &py_ws_run_batch, py::arg("host"), py::arg("port"),
        py::arg("path"), py::arg("req_messages"), py::arg("req_offsets"),
        py::arg("concurrency"), py::arg("timeout_s") = 120.0,
        "Per-connection WS batch with concurrency cap + refill: each request has "
        "its own message sequence + paced send schedule, so N distinct realtime "
        "requests run concurrently over one poll() loop. Returns a WsResult per "
        "request (index-aligned to req_messages).");
}
