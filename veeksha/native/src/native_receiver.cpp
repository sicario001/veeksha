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
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
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

// Scan the body for SSE `data:` lines, timestamping each at read time.
static void parse_body_sse(EngineConn& c) {
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
      c.sizes.push_back((int)d.size());
      c.content += d;
    }
  }
}

static std::vector<ReqResult> run_batch(const std::string& host, int port,
                                        const std::vector<std::string>& requests,
                                        int concurrency, double timeout_s,
                                        bool sse) {
  struct rlimit rl;
  getrlimit(RLIMIT_NOFILE, &rl);
  rl.rlim_cur = std::max<rlim_t>(rl.rlim_cur, (rlim_t)(concurrency + 64));
  if (rl.rlim_cur > rl.rlim_max) rl.rlim_cur = rl.rlim_max;
  setrlimit(RLIMIT_NOFILE, &rl);

  int total = (int)requests.size();
  std::vector<ReqResult> results(total);
  for (int i = 0; i < total; i++) results[i].index = i;

  std::unordered_map<int, EngineConn> conns;
  int launched = 0, completed = 0;
  double t0 = now_ms();

  auto try_launch = [&]() {
    while ((int)conns.size() < concurrency && launched < total) {
      int fd = make_conn(host, port);
      if (fd < 0) {
        results[launched].error = "connect failed";
        completed++;
        launched++;
        continue;
      }
      EngineConn c;
      c.index = launched;
      c.sse = sse;
      conns.emplace(fd, std::move(c));
      launched++;
    }
  };
  try_launch();

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
    int n = poll(pfds.data(), pfds.size(), 50);
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
        c.send_time = now_ms();
        const std::string& req = requests[c.index];
        ssize_t w = send(p.fd, req.data(), req.size(), 0);
        (void)w;
      } else if (c.connected && (p.revents & POLLIN)) {
        char buf[16384];
        ssize_t r = recv(p.fd, buf, sizeof(buf), 0);
        if (r > 0) {
          if (!c.headers_done) {
            c.header_buf.append(buf, r);
            parse_headers(c);
            if (c.headers_done && c.sse) parse_body_sse(c);
            else if (c.headers_done) {
              c.content += c.inbuf;
              c.inbuf.clear();
            }
          } else if (c.sse) {
            c.inbuf.append(buf, r);
            parse_body_sse(c);
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
        // non-streaming: one event at completion time for the whole body
        res.offsets_ms.push_back(now_ms() - c.send_time);
        res.sizes.push_back((int)res.content.size());
      }
      close(fd);
      conns.erase(it);
      completed++;
    }
    try_launch();
  }
  for (auto& kv : conns) {
    // timed out mid-flight
    ReqResult& res = results[kv.second.index];
    if (res.error.empty()) res.error = "timeout";
    close(kv.first);
  }
  return results;
}

static std::vector<ReqResult> py_run_batch(const std::string& host, int port,
                                           const std::vector<std::string>& requests,
                                           int concurrency, double timeout_s,
                                           bool sse) {
  py::gil_scoped_release release;
  return run_batch(host, port, requests, concurrency, timeout_s, sse);
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
    }
    if (res.sent_offsets_ms.empty() && !kv.second.sent_offsets.empty())
      res.sent_offsets_ms = std::move(kv.second.sent_offsets);
    if (res.error.empty() && res.offsets_ms.empty()) res.error = "timeout";
    close(kv.first);
  }
  return results;
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
      .def_readonly("error", &ReqResult::error);

  m.def("run_batch", &py_run_batch, py::arg("host"), py::arg("port"),
        py::arg("requests"), py::arg("concurrency"), py::arg("timeout_s") = 120.0,
        py::arg("sse") = true,
        "Real per-request engine: owns connection concurrency over one poll() "
        "loop, sends caller-built HTTP requests, returns per-request ReqResult "
        "(status, content, kernel-time chunk offsets_ms + sizes).");

  py::class_<WsResult>(m, "WsResult")
      .def_readonly("index", &WsResult::index)
      .def_readonly("offsets_ms", &WsResult::offsets_ms)
      .def_readonly("sizes", &WsResult::sizes)
      .def_readonly("content", &WsResult::content)
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
}
