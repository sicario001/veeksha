// veeksha_native — the native (C++) transport + timing engine for Veeksha.
//
// Event-driven reactor loops (kqueue on BSD/darwin, epoll on Linux, poll()
// fallback) own connection concurrency in native threads: they send
// caller-built requests, parse the stream framing (SSE + chunked
// transfer-encoding, WebSocket), and timestamp every event at socket-read time
// on CLOCK_MONOTONIC — no Python (and no asyncio scheduling lag) anywhere on
// the per-event path. Paced sends and dependent-turn launches are driven by
// per-loop deadline min-heaps, so wakeups are O(ready + due), not O(conns).
//
// Timestamps are taken immediately after each recv() returns. Kernel cmsg
// receive timestamps (SO_TIMESTAMP*) were evaluated and rejected: they are not
// delivered for TCP stream sockets on darwin (verified empirically) nor via
// SO_TIMESTAMP on Linux TCP, so relying on them silently degrades to a
// wall-clock fallback and splits the clock domain. Read-time monotonic
// stamping is what the drift benches actually validated.
//
// Declared `py::mod_gil_not_used()` so it is safe on free-threaded CPython and
// does not re-enable the GIL. Blocking loops release the GIL-equivalent via
// py::gil_scoped_release (a no-op benefit under free-threading, but correct
// either way).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

#if defined(__APPLE__) || defined(__FreeBSD__)
#define VEEKSHA_USE_KQUEUE 1
#include <sys/event.h>
#elif defined(__linux__)
#define VEEKSHA_USE_EPOLL 1
#include <sys/epoll.h>
#endif

#if defined(__APPLE__)
#include <pthread/qos.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Pin the calling thread to a high QoS class on Apple platforms. macOS parks
// default-QoS threads of an idle process on efficiency cores, which adds
// ~2-8 ms of wakeup slop to timer-driven emits/paced sends at LOW load — the
// exact tail the drift benches see on a quiet dev box. Timing-critical loops
// opt into USER_INTERACTIVE so their wakeups stay on performance cores.
static void set_thread_qos() {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

static void raise_nofile(int need) {
  struct rlimit rl;
  getrlimit(RLIMIT_NOFILE, &rl);
  rl.rlim_cur = std::max<rlim_t>(rl.rlim_cur, (rlim_t)(need + 64));
  if (rl.rlim_cur > rl.rlim_max) rl.rlim_cur = rl.rlim_max;
  setrlimit(RLIMIT_NOFILE, &rl);
}

// Resolve host:port once per run. getaddrinfo handles DNS names ("localhost",
// internal hostnames) as well as dotted quads; per-connection connects then
// reuse the sockaddr with no resolver cost. Returns false on failure — callers
// must surface that as a per-request error, never fall through to 0.0.0.0.
static bool resolve_addr(const std::string& host, int port, sockaddr_in* out) {
  memset(out, 0, sizeof(*out));
  out->sin_family = AF_INET;
  out->sin_port = htons(port);
  if (inet_pton(AF_INET, host.c_str(), &out->sin_addr) == 1) return true;
  struct addrinfo hints;
  struct addrinfo* res = nullptr;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || res == nullptr)
    return false;
  out->sin_addr = ((sockaddr_in*)res->ai_addr)->sin_addr;
  freeaddrinfo(res);
  return true;
}

static int make_conn(const sockaddr_in& addr) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  int flags = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  int one = 1;
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  int r = connect(fd, (const struct sockaddr*)&addr, sizeof(addr));
  if (r < 0 && errno != EINPROGRESS) {
    close(fd);
    return -1;
  }
  return fd;
}

// Write as much of (pending outbuf +) data as the socket accepts; buffer the
// unsent tail so the reactor loop can flush it on writability. A request or
// frame larger than the socket send buffer (long prompts, base64 audio) must
// never be silently truncated. Returns false on a hard socket error.
static bool conn_send(int fd, std::string& outbuf, const char* data,
                      size_t len) {
  if (outbuf.empty()) {
    size_t off = 0;
    while (off < len) {
      ssize_t w = ::send(fd, data + off, len - off, 0);
      if (w > 0) {
        off += (size_t)w;
        continue;
      }
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;
      return false;
    }
    if (off < len) outbuf.assign(data + off, len - off);
  } else {
    outbuf.append(data, len);
  }
  return true;
}

static bool flush_outbuf(int fd, std::string& outbuf) {
  size_t off = 0;
  while (off < outbuf.size()) {
    ssize_t w = ::send(fd, outbuf.data() + off, outbuf.size() - off, 0);
    if (w > 0) {
      off += (size_t)w;
      continue;
    }
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
    return false;
  }
  outbuf.erase(0, off);
  return true;
}

// recv() with EINTR retry. Returns like recv(); EAGAIN/EWOULDBLOCK pass through.
static ssize_t recv_retry(int fd, char* buf, size_t len) {
  while (true) {
    ssize_t r = recv(fd, buf, len, 0);
    if (r < 0 && errno == EINTR) continue;
    return r;
  }
}

// ===========================================================================
// Reactor — readiness notification behind one interface. There is no single
// portable kernel API for scalable readiness: kqueue is the BSD/darwin API,
// epoll the Linux one, and POSIX poll() the lowest common denominator — so
// this class IS the portability layer, the same #ifdef seam every mainstream
// event runtime (libuv, asio, tokio/mio) ships. Callers only see
// set_interest/remove/wait. poll() remains fully supported as the portable
// fallback (any other POSIX system compiles to it automatically, and
// VEEKSHA_NATIVE_REACTOR=poll selects it at runtime for A/B measurement).
//
// Why bother: poll() rescans every fd on every tick (O(n) in the kernel AND
// in userspace building the pollfd array); kqueue/epoll wakeups cost
// O(ready). Honest note: on a single box where the co-located mock server
// saturates first, the hot-loop knee is CPU-bound and the two are within
// noise of each other — the O(ready) advantage matters for large mostly-idle
// connection sets and on dedicated-host rigs.
// ===========================================================================

struct ReactorEvent {
  int fd = -1;
  bool readable = false;
  bool writable = false;
  bool error = false;
};

// The kernel-queue backend can be disabled at runtime for A/B measurement:
// VEEKSHA_NATIVE_REACTOR=poll falls back to the portable poll() path.
static bool use_kernel_queue() {
  static const bool enabled = [] {
    const char* v = getenv("VEEKSHA_NATIVE_REACTOR");
    return !(v != nullptr && strcmp(v, "poll") == 0);
  }();
  return enabled;
}

class Reactor {
 public:
  Reactor() : use_kq_(use_kernel_queue()) {
#if defined(VEEKSHA_USE_KQUEUE)
    if (use_kq_) kq_ = kqueue();
#elif defined(VEEKSHA_USE_EPOLL)
    if (use_kq_) ep_ = epoll_create1(0);
#else
    use_kq_ = false;
#endif
  }
  ~Reactor() {
#if defined(VEEKSHA_USE_KQUEUE)
    if (kq_ >= 0) close(kq_);
#elif defined(VEEKSHA_USE_EPOLL)
    if (ep_ >= 0) close(ep_);
#endif
  }

  void set_interest(int fd, bool want_read, bool want_write) {
    Interest& cur = interest_[fd];
#if defined(VEEKSHA_USE_KQUEUE)
    if (use_kq_) {
      struct kevent chs[2];
      int n = 0;
      if (want_read != cur.read) {
        EV_SET(&chs[n++], fd, EVFILT_READ, want_read ? EV_ADD : EV_DELETE, 0, 0,
               nullptr);
      }
      if (want_write != cur.write) {
        EV_SET(&chs[n++], fd, EVFILT_WRITE, want_write ? EV_ADD : EV_DELETE, 0,
               0, nullptr);
      }
      if (n > 0) {
        struct timespec zero = {0, 0};
        kevent(kq_, chs, n, nullptr, 0, &zero);
      }
    }
#elif defined(VEEKSHA_USE_EPOLL)
    if (use_kq_) {
      uint32_t ev = (want_read ? EPOLLIN : 0) | (want_write ? EPOLLOUT : 0);
      struct epoll_event e;
      memset(&e, 0, sizeof(e));
      e.events = ev;
      e.data.fd = fd;
      if (!cur.read && !cur.write) {
        if (ev) epoll_ctl(ep_, EPOLL_CTL_ADD, fd, &e);
      } else if (!ev) {
        epoll_ctl(ep_, EPOLL_CTL_DEL, fd, &e);
      } else {
        epoll_ctl(ep_, EPOLL_CTL_MOD, fd, &e);
      }
    }
#endif
    cur.read = want_read;
    cur.write = want_write;
  }

  void remove(int fd) {
#if defined(VEEKSHA_USE_KQUEUE)
    // closing the fd removes its kevents; drop bookkeeping only.
#elif defined(VEEKSHA_USE_EPOLL)
    if (use_kq_) {
      struct epoll_event e;
      epoll_ctl(ep_, EPOLL_CTL_DEL, fd, &e);
    }
#endif
    interest_.erase(fd);
  }

  int wait(int timeout_ms, std::vector<ReactorEvent>& out) {
    out.clear();
#if defined(VEEKSHA_USE_KQUEUE)
    if (use_kq_) {
      if (evbuf_.size() < interest_.size() * 2 + 8)
        evbuf_.resize(interest_.size() * 2 + 8);
      struct timespec ts;
      ts.tv_sec = timeout_ms / 1000;
      ts.tv_nsec = (long)(timeout_ms % 1000) * 1000000L;
      int n = kevent(kq_, nullptr, 0, evbuf_.data(), (int)evbuf_.size(), &ts);
      for (int i = 0; i < n; i++) {
        ReactorEvent ev;
        ev.fd = (int)evbuf_[i].ident;
        if (evbuf_[i].filter == EVFILT_READ) ev.readable = true;
        if (evbuf_[i].filter == EVFILT_WRITE) ev.writable = true;
        if (evbuf_[i].flags & EV_ERROR) ev.error = true;
        if (evbuf_[i].flags & EV_EOF) ev.readable = true;  // recv() reports 0
        out.push_back(ev);
      }
      return n;
    }
#elif defined(VEEKSHA_USE_EPOLL)
    if (use_kq_) {
      if (epbuf_.size() < interest_.size() + 8)
        epbuf_.resize(interest_.size() + 8);
      int n = epoll_wait(ep_, epbuf_.data(), (int)epbuf_.size(), timeout_ms);
      for (int i = 0; i < n; i++) {
        ReactorEvent ev;
        ev.fd = epbuf_[i].data.fd;
        ev.readable = epbuf_[i].events & (EPOLLIN | EPOLLHUP);
        ev.writable = epbuf_[i].events & EPOLLOUT;
        ev.error = epbuf_[i].events & EPOLLERR;
        out.push_back(ev);
      }
      return n;
    }
#endif
    std::vector<struct pollfd> pfds;
    pfds.reserve(interest_.size());
    for (auto& kv : interest_) {
      struct pollfd p;
      p.fd = kv.first;
      p.events = (short)((kv.second.read ? POLLIN : 0) |
                         (kv.second.write ? POLLOUT : 0));
      p.revents = 0;
      pfds.push_back(p);
    }
    int n = poll(pfds.data(), pfds.size(), timeout_ms);
    if (n <= 0) return n;
    for (auto& p : pfds) {
      if (!p.revents) continue;
      ReactorEvent ev;
      ev.fd = p.fd;
      ev.readable = p.revents & (POLLIN | POLLHUP);
      ev.writable = p.revents & POLLOUT;
      ev.error = p.revents & POLLERR;
      out.push_back(ev);
    }
    return n;
  }

 private:
  struct Interest {
    bool read = false;
    bool write = false;
  };
  bool use_kq_ = false;
  std::unordered_map<int, Interest> interest_;
#if defined(VEEKSHA_USE_KQUEUE)
  int kq_ = -1;
  std::vector<struct kevent> evbuf_;
#elif defined(VEEKSHA_USE_EPOLL)
  int ep_ = -1;
  std::vector<struct epoll_event> epbuf_;
#endif
};

// Deadline min-heap keyed by (deadline_ms, id). Entries are lazily invalidated:
// consumers re-check the owning object's state on pop, so stale entries (a
// closed connection, a rescheduled deadline) cost one pop each.
class TimerHeap {
 public:
  void push(double deadline_ms, int id) { heap_.emplace(-deadline_ms, id); }
  bool empty() const { return heap_.empty(); }
  double next_deadline() const {
    return heap_.empty() ? 1e18 : -heap_.top().first;
  }
  // Pop every entry due at `now`; returns ids (possibly stale — re-validate).
  void pop_due(double now, std::vector<int>& out) {
    out.clear();
    while (!heap_.empty() && -heap_.top().first <= now) {
      out.push_back(heap_.top().second);
      heap_.pop();
    }
  }

 private:
  // max-heap over negated deadlines == min-heap over deadlines
  std::priority_queue<std::pair<double, int>> heap_;
};

// ===========================================================================
// HTTP streaming response state, shared by the batch (run_batch) and chains
// (run_chains) engines: status-line/header parse, chunked transfer-encoding
// decode, and SSE `data:` line framing with read-time timestamps.
// ===========================================================================

struct HttpStreamState {
  bool headers_done = false;
  bool done = false;       // stream finished ([DONE] seen or chunked body ended)
  bool sse = true;
  bool chunked = false;
  long long chunk_left = 0;  // >0: bytes left in chunk; -1: awaiting CRLF
  bool body_done = false;
  int status = 0;
  double send_time = 0.0;
  std::string header_buf;
  std::string rawbuf;   // undecoded chunked bytes
  std::string inbuf;    // decoded body bytes awaiting framing
  std::string content;  // concatenated SSE payloads (or raw body if !sse)
  std::vector<double> offsets;  // per-event arrival offset (ms from send)
  std::vector<int> sizes;       // per-event payload size (bytes)
};

// Decode HTTP/1.1 chunked transfer-encoding: move complete chunk payloads from
// raw into out; leave partial chunks in raw for the next read. Sets body_done
// on the terminal 0-size chunk (trailers, if any, are ignored).
static void dechunk(std::string& raw, std::string& out, long long& chunk_left,
                    bool& body_done) {
  while (true) {
    if (chunk_left > 0) {
      size_t take = std::min((size_t)chunk_left, raw.size());
      out.append(raw, 0, take);
      raw.erase(0, take);
      chunk_left -= (long long)take;
      if (chunk_left > 0) return;  // need more data
      chunk_left = -1;             // expect the CRLF that closes the chunk
    }
    if (chunk_left == -1) {
      if (raw.size() < 2) return;
      raw.erase(0, 2);
      chunk_left = 0;
    }
    size_t eol = raw.find("\r\n");
    if (eol == std::string::npos) return;
    long long sz = strtoll(raw.substr(0, eol).c_str(), nullptr, 16);
    raw.erase(0, eol + 2);
    if (sz <= 0) {
      body_done = true;
      return;
    }
    chunk_left = sz;
  }
}

// Scan decoded body bytes for SSE `data:` lines, stamping each with `ts` — the
// read time of the recv() that delivered these bytes, so the offset reflects
// arrival, not when a later pass got around to parsing.
static void parse_body_sse(HttpStreamState& s, double ts) {
  size_t pos;
  while ((pos = s.inbuf.find('\n')) != std::string::npos) {
    std::string line = s.inbuf.substr(0, pos);
    s.inbuf.erase(0, pos + 1);
    while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
      line.pop_back();
    if (line.rfind("data:", 0) == 0) {
      std::string d = line.substr(5);
      while (!d.empty() && d.front() == ' ') d.erase(0, 1);
      if (d == "[DONE]") {
        s.done = true;
        return;
      }
      s.offsets.push_back(ts - s.send_time);
      s.sizes.push_back((int)d.size());
      s.content += d;
    }
  }
}

static void http_parse_headers(HttpStreamState& s) {
  size_t term = s.header_buf.find("\r\n\r\n");
  if (term == std::string::npos) return;
  std::string head = s.header_buf.substr(0, term);
  std::string rest = s.header_buf.substr(term + 4);
  size_t sp = head.find(' ');
  if (sp != std::string::npos) s.status = atoi(head.c_str() + sp + 1);
  std::string lower(head);
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char ch) { return (char)std::tolower(ch); });
  size_t te = lower.find("transfer-encoding:");
  if (te != std::string::npos) {
    size_t eol = lower.find("\r\n", te);
    size_t ch = lower.find("chunked", te);
    if (ch != std::string::npos && (eol == std::string::npos || ch < eol))
      s.chunked = true;
  }
  s.headers_done = true;
  if (s.chunked)
    s.rawbuf = rest;
  else
    s.inbuf = rest;
  s.header_buf.clear();
}

// Feed freshly read bytes through header parse -> chunked decode -> framing.
static void http_feed(HttpStreamState& s, const char* buf, size_t n, double ts) {
  if (!s.headers_done) {
    s.header_buf.append(buf, n);
    http_parse_headers(s);
    if (!s.headers_done) return;
  } else {
    (s.chunked ? s.rawbuf : s.inbuf).append(buf, n);
  }
  if (s.chunked) dechunk(s.rawbuf, s.inbuf, s.chunk_left, s.body_done);
  if (s.sse) {
    parse_body_sse(s, ts);
  } else {
    s.content += s.inbuf;
    s.inbuf.clear();
  }
  if (s.body_done) s.done = true;
}

// JSON string escaping for content spliced into a request-body template
// (native multi-turn history injection). UTF-8 bytes pass through; quotes,
// backslashes and control characters are escaped per RFC 8259.
static std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 16);
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          char buf[8];
          snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out += (char)c;
        }
    }
  }
  return out;
}

// ===========================================================================
// Per-request receive engine (run_batch): owns connection concurrency (one
// reactor loop over many sockets per thread), sends caller-provided HTTP
// requests, and returns per-request results with the event timeline
// timestamped at read time. Python builds the request bytes and parses the
// returned content; native owns transport + timing.
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
  std::string outbuf;
  HttpStreamState h;
};

// One shard's reactor loop: services the request indices in `indices` with up
// to `concurrency` in-flight, writing into results[idx] (disjoint per shard,
// so threads never touch the same element — no hot-path locks). `t0` is shared
// across shards so open-loop arrival deadlines line up on one global clock.
static void run_batch_worker(const sockaddr_in& addr,
                             const std::vector<std::string>& requests,
                             const std::vector<int>& indices, int concurrency,
                             double timeout_s, bool sse,
                             const std::vector<double>& dispatch_offsets_ms,
                             double t0, std::vector<ReqResult>& results) {
  set_thread_qos();
  int total = (int)indices.size();
  Reactor reactor;
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
      int fd = make_conn(addr);
      if (fd < 0) {
        results[idx].error = "connect failed";
        completed++;
        cur++;
        continue;
      }
      EngineConn c;
      c.index = idx;
      c.h.sse = sse;
      results[idx].dispatch_offset_ms = now_ms() - t0;
      conns.emplace(fd, std::move(c));
      reactor.set_interest(fd, false, true);  // connect completes as writable
      cur++;
    }
    return 1e9;
  };
  double next_dispatch = try_launch();

  auto finish = [&](int fd, EngineConn& c) {
    ReqResult& res = results[c.index];
    res.status = c.h.status;
    res.content = std::move(c.h.content);
    res.offsets_ms = std::move(c.h.offsets);
    res.sizes = std::move(c.h.sizes);
    if (!c.h.sse) {
      res.offsets_ms.push_back(now_ms() - c.h.send_time);
      res.sizes.push_back((int)res.content.size());
    }
    reactor.remove(fd);
    close(fd);
    conns.erase(fd);
    completed++;
  };

  std::vector<ReactorEvent> evs;
  while (completed < total && (now_ms() - t0) < timeout_s * 1000.0) {
    int poll_ms = 50;
    if (open_loop && next_dispatch < poll_ms)
      poll_ms = (int)std::max(0.0, next_dispatch);
    reactor.wait(poll_ms, evs);
    for (auto& ev : evs) {
      auto it = conns.find(ev.fd);
      if (it == conns.end()) continue;
      EngineConn& c = it->second;
      int fd = ev.fd;
      if (!c.connected && (ev.writable || ev.error)) {
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
          results[c.index].error = "connect error";
          finish(fd, c);
          continue;
        }
        c.connected = true;
        c.h.send_time = now_ms();
        const std::string& req = requests[c.index];
        if (!conn_send(fd, c.outbuf, req.data(), req.size())) {
          results[c.index].error = "send error";
          finish(fd, c);
          continue;
        }
        reactor.set_interest(fd, true, !c.outbuf.empty());
        continue;
      }
      if (!c.connected) continue;
      if (ev.writable) {
        if (!flush_outbuf(fd, c.outbuf)) {
          results[c.index].error = "send error";
          finish(fd, c);
          continue;
        }
        if (c.outbuf.empty()) reactor.set_interest(fd, true, false);
      }
      if (ev.readable) {
        char buf[16384];
        ssize_t r = recv_retry(fd, buf, sizeof(buf));
        double ts = now_ms();
        if (r > 0) {
          http_feed(c.h, buf, (size_t)r, ts);
          if (c.h.done) finish(fd, c);
        } else if (r == 0) {
          finish(fd, c);
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          results[c.index].error = "recv error";
          finish(fd, c);
        }
      } else if (ev.error) {
        finish(fd, c);
      }
    }
    next_dispatch = try_launch();
  }
  for (auto& kv : conns) {
    ReqResult& res = results[kv.second.index];
    if (res.error.empty()) res.error = "timeout";
    close(kv.first);
  }
}

// Public engine: shards the requests across `num_threads` native reactor
// threads (each its own loop over a disjoint socket set) and merges. On
// free-threaded CPython the loops run in true parallel (no Python in the hot
// path) — the standard sharded-reactor pattern. `num_threads<=1` = one loop.
static std::vector<ReqResult> run_batch(const std::string& host, int port,
                                        const std::vector<std::string>& requests,
                                        int concurrency, double timeout_s, bool sse,
                                        const std::vector<double>& dispatch_offsets_ms,
                                        int num_threads) {
  raise_nofile(concurrency);

  int total = (int)requests.size();
  std::vector<ReqResult> results(total);
  for (int i = 0; i < total; i++) results[i].index = i;
  if (total == 0) return results;

  sockaddr_in addr;
  if (!resolve_addr(host, port, &addr)) {
    for (auto& r : results) r.error = "resolve failed: " + host;
    return results;
  }

  int nthreads = std::max(1, num_threads);
  if (nthreads > total) nthreads = total;
  double t0 = now_ms();

  if (nthreads == 1) {
    std::vector<int> all(total);
    for (int i = 0; i < total; i++) all[i] = i;
    run_batch_worker(addr, requests, all, concurrency, timeout_s, sse,
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
      run_batch_worker(addr, requests, shards[t], per_thread_conc, timeout_s,
                       sse, dispatch_offsets_ms, t0, results);
    });
  }
  for (auto& th : pool) th.join();
  return results;
}

static std::vector<ReqResult> py_run_batch(
    const std::string& host, int port,
    const std::vector<std::string>& requests, int concurrency, double timeout_s,
    bool sse, const std::vector<double>& dispatch_offsets_ms, int num_threads) {
  if (!dispatch_offsets_ms.empty() &&
      dispatch_offsets_ms.size() < requests.size()) {
    throw std::invalid_argument(
        "dispatch_offsets_ms must cover every request (got " +
        std::to_string(dispatch_offsets_ms.size()) + " offsets for " +
        std::to_string(requests.size()) + " requests)");
  }
  py::gil_scoped_release release;
  return run_batch(host, port, requests, concurrency, timeout_s, sse,
                   dispatch_offsets_ms, num_threads);
}

// ===========================================================================
// Native receive->dispatch coupling with INTER-TURN CONTENT FLOW (run_chains).
// Each "chain" is a sequence of dependent turns. Python pre-builds every
// turn's request as a TEMPLATE: literal body segments interleaved with "holes"
// that reference prior turns of the same chain. When a turn completes, native
// schedules the next turn at complete_time + delay_ms (think time), fills its
// holes with the JSON-escaped content of the referenced prior turns, computes
// Content-Length, and fires — a faithful multi-turn conversation with no
// Python anywhere between receive and next dispatch.
// ===========================================================================

struct ChainTurnSpec {
  // wire = header_prefix + str(body_len) + header_suffix + body
  std::string header_prefix;  // through "Content-Length: "
  std::string header_suffix;  // remaining headers + CRLFCRLF
  // body = seg[0] + fill(ref[0]) + seg[1] + ... + fill(ref[n-1]) + seg[n]
  std::vector<std::string> body_segments;
  std::vector<int> hole_refs;  // indices of prior turns whose content fills holes
  double delay_ms = 0.0;       // think time after the prior turn completes
};

struct ChainTurn {
  std::vector<double> offsets_ms;
  std::string content;
  int status = 0;
  double dispatch_offset_ms = 0.0;  // connect initiation (ms from run start)
};

struct ChainResult {
  int index = -1;
  std::vector<ChainTurn> turns;
  // Per dependent/scheduled launch: how late the dispatch fired vs its
  // deadline (prior-turn completion + delay_ms, or the chain's open-loop
  // start offset). The coupling precision.
  std::vector<double> handoff_ms;
  std::string error;
};

struct ChainConn {
  int chain = -1;
  size_t turn = 0;
  bool connected = false;
  double dispatch_offset_ms = 0.0;
  std::string outbuf;
  HttpStreamState h;
};

// One shard's chains loop. Chain starts (turn 0) are gated by the in-flight
// cap; open-loop start offsets and dependent-turn think times both flow
// through one deadline heap (ids: chain index).
static void run_chains_worker(
    const sockaddr_in& addr,
    const std::vector<std::vector<ChainTurnSpec>>& chains,
    const std::vector<int>& indices, int concurrency, double timeout_s,
    const std::vector<double>& start_offsets_ms, double t0,
    std::vector<ChainResult>& results) {
  set_thread_qos();
  Reactor reactor;
  TimerHeap timers;
  std::unordered_map<int, ChainConn> conns;
  int total = (int)indices.size();
  int chains_done = 0;
  bool open_loop = !start_offsets_ms.empty();

  // Per-chain pending launch: next turn and its deadline (ms from t0).
  std::unordered_map<int, std::pair<size_t, double>> pending;
  std::vector<int> deferred_starts;  // due turn-0 launches held back by the cap
  size_t next_start = 0;             // closed-loop start cursor into `indices`

  auto fail_chain = [&](int ch, const std::string& err) {
    if (results[ch].error.empty()) results[ch].error = err;
    chains_done++;
  };

  // Build turn `turn`'s wire bytes from its template + prior turn contents.
  auto build_wire = [&](int ch, size_t turn) -> std::string {
    const ChainTurnSpec& spec = chains[ch][turn];
    std::string body = spec.body_segments.empty() ? "" : spec.body_segments[0];
    for (size_t h = 0; h < spec.hole_refs.size(); h++) {
      const std::string& prior = results[ch].turns[spec.hole_refs[h]].content;
      body += json_escape(prior);
      body += spec.body_segments[h + 1];
    }
    return spec.header_prefix + std::to_string(body.size()) +
           spec.header_suffix + body;
  };

  auto launch_turn = [&](int ch, size_t turn, double deadline_ms) -> bool {
    int fd = make_conn(addr);
    if (fd < 0) {
      fail_chain(ch, "connect failed");
      return false;
    }
    double launched_at = now_ms() - t0;
    if (deadline_ms >= 0.0)
      results[ch].handoff_ms.push_back(launched_at - deadline_ms);
    ChainConn c;
    c.chain = ch;
    c.turn = turn;
    c.dispatch_offset_ms = launched_at;
    conns.emplace(fd, std::move(c));
    reactor.set_interest(fd, false, true);
    return true;
  };

  auto refill = [&]() {
    if (open_loop) {
      while (!deferred_starts.empty() && (int)conns.size() < concurrency) {
        int ch = deferred_starts.front();
        deferred_starts.erase(deferred_starts.begin());
        auto p = pending.find(ch);
        if (p == pending.end()) continue;
        size_t turn = p->second.first;
        double due = p->second.second;
        pending.erase(p);
        launch_turn(ch, turn, due);
      }
      return;
    }
    while ((int)conns.size() < concurrency && next_start < indices.size()) {
      int ch = indices[next_start++];
      if (chains[ch].empty()) {
        chains_done++;
        continue;
      }
      launch_turn(ch, 0, -1.0);
    }
  };

  if (open_loop) {
    for (int ch : indices) {
      if (chains[ch].empty()) {
        chains_done++;
        continue;
      }
      pending[ch] = {0, start_offsets_ms[ch]};
      timers.push(start_offsets_ms[ch], ch);
    }
  } else {
    refill();
  }

  std::vector<int> due;
  auto fire_due = [&]() {
    double now = now_ms() - t0;
    timers.pop_due(now, due);
    for (int ch : due) {
      auto p = pending.find(ch);
      if (p == pending.end()) continue;  // stale entry
      if (p->second.first == 0 && (int)conns.size() >= concurrency) {
        deferred_starts.push_back(ch);  // at cap: defer the chain start
        continue;
      }
      size_t turn = p->second.first;
      double deadline = p->second.second;
      pending.erase(p);
      launch_turn(ch, turn, deadline);
    }
  };

  auto finish_conn = [&](int fd, ChainConn& c, bool ok,
                         const std::string& err) {
    reactor.remove(fd);
    close(fd);
    int ch = c.chain;
    size_t turn = c.turn;
    double dispatch_offset = c.dispatch_offset_ms;
    HttpStreamState h = std::move(c.h);
    conns.erase(fd);
    if (!ok) {
      fail_chain(ch, err);
      return;
    }
    double complete_at = now_ms() - t0;
    ChainTurn ct;
    ct.offsets_ms = std::move(h.offsets);
    ct.content = std::move(h.content);
    ct.status = h.status;
    ct.dispatch_offset_ms = dispatch_offset;
    results[ch].turns.push_back(std::move(ct));

    size_t next_turn = turn + 1;
    if (next_turn < chains[ch].size()) {
      double delay = chains[ch][next_turn].delay_ms;
      double due_at = complete_at + delay;
      if (delay <= 0.05) {
        // COUPLING: fire immediately — no timer round-trip on the zero-think
        // path, preserving the tens-of-microseconds handoff.
        launch_turn(ch, next_turn, due_at);
      } else {
        pending[ch] = {next_turn, due_at};
        timers.push(due_at, ch);
      }
    } else {
      chains_done++;
    }
  };

  std::vector<ReactorEvent> evs;
  while (chains_done < total && (now_ms() - t0) < timeout_s * 1000.0) {
    double now = now_ms() - t0;
    double until_timer = timers.empty() ? 50.0 : timers.next_deadline() - now;
    int poll_ms = (int)std::max(0.0, std::min(50.0, until_timer));
    reactor.wait(poll_ms, evs);
    for (auto& ev : evs) {
      auto it = conns.find(ev.fd);
      if (it == conns.end()) continue;
      ChainConn& c = it->second;
      int fd = ev.fd;
      if (!c.connected && (ev.writable || ev.error)) {
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
          finish_conn(fd, c, false, "connect error");
          continue;
        }
        c.connected = true;
        c.h.send_time = now_ms();
        std::string wire = build_wire(c.chain, c.turn);
        if (!conn_send(fd, c.outbuf, wire.data(), wire.size())) {
          finish_conn(fd, c, false, "send error");
          continue;
        }
        reactor.set_interest(fd, true, !c.outbuf.empty());
        continue;
      }
      if (!c.connected) continue;
      if (ev.writable) {
        if (!flush_outbuf(fd, c.outbuf)) {
          finish_conn(fd, c, false, "send error");
          continue;
        }
        if (c.outbuf.empty()) reactor.set_interest(fd, true, false);
      }
      if (ev.readable) {
        char buf[16384];
        ssize_t r = recv_retry(fd, buf, sizeof(buf));
        double ts = now_ms();
        if (r > 0) {
          http_feed(c.h, buf, (size_t)r, ts);
          if (c.h.done) finish_conn(fd, c, true, "");
        } else if (r == 0) {
          finish_conn(fd, c, true, "");
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          finish_conn(fd, c, false, "recv error");
        }
      } else if (ev.error) {
        finish_conn(fd, c, true, "");
      }
    }
    fire_due();
    refill();
  }
  for (auto& kv : conns) {
    if (results[kv.second.chain].error.empty())
      results[kv.second.chain].error = "timeout";
    close(kv.first);
  }
}

static std::vector<ChainResult> run_chains(
    const std::string& host, int port,
    const std::vector<std::vector<ChainTurnSpec>>& chains, int concurrency,
    double timeout_s, const std::vector<double>& start_offsets_ms,
    int num_threads) {
  raise_nofile(concurrency);
  int total = (int)chains.size();
  std::vector<ChainResult> results(total);
  for (int i = 0; i < total; i++) results[i].index = i;
  if (total == 0) return results;

  sockaddr_in addr;
  if (!resolve_addr(host, port, &addr)) {
    for (auto& r : results) r.error = "resolve failed: " + host;
    return results;
  }

  int nthreads = std::max(1, num_threads);
  if (nthreads > total) nthreads = total;
  double t0 = now_ms();

  if (nthreads == 1) {
    std::vector<int> all(total);
    for (int i = 0; i < total; i++) all[i] = i;
    run_chains_worker(addr, chains, all, concurrency, timeout_s,
                      start_offsets_ms, t0, results);
    return results;
  }
  std::vector<std::vector<int>> shards(nthreads);
  for (int i = 0; i < total; i++) shards[i % nthreads].push_back(i);
  int per_thread_conc = (concurrency + nthreads - 1) / nthreads;
  if (per_thread_conc < 1) per_thread_conc = 1;
  std::vector<std::thread> pool;
  pool.reserve(nthreads);
  for (int t = 0; t < nthreads; t++) {
    pool.emplace_back([&, t]() {
      run_chains_worker(addr, chains, shards[t], per_thread_conc, timeout_s,
                        start_offsets_ms, t0, results);
    });
  }
  for (auto& th : pool) th.join();
  return results;
}

static std::vector<ChainResult> py_run_chains(
    const std::string& host, int port,
    const std::vector<std::vector<ChainTurnSpec>>& chains, int concurrency,
    double timeout_s, const std::vector<double>& start_offsets_ms,
    int num_threads) {
  for (size_t ci = 0; ci < chains.size(); ci++) {
    for (size_t ti = 0; ti < chains[ci].size(); ti++) {
      const ChainTurnSpec& s = chains[ci][ti];
      if (s.body_segments.size() != s.hole_refs.size() + 1) {
        throw std::invalid_argument(
            "chain " + std::to_string(ci) + " turn " + std::to_string(ti) +
            ": body_segments must be hole_refs+1 segments");
      }
      for (int ref : s.hole_refs) {
        if (ref < 0 || (size_t)ref >= ti) {
          throw std::invalid_argument(
              "chain " + std::to_string(ci) + " turn " + std::to_string(ti) +
              ": hole ref " + std::to_string(ref) +
              " must point to an earlier turn");
        }
      }
    }
  }
  if (!start_offsets_ms.empty() && start_offsets_ms.size() < chains.size()) {
    throw std::invalid_argument(
        "start_offsets_ms must cover every chain when provided");
  }
  py::gil_scoped_release release;
  return run_chains(host, port, chains, concurrency, timeout_s,
                    start_offsets_ms, num_threads);
}

// ===========================================================================
// Native WebSocket receive + framed send: the interactivity-critical path.
// Connects, performs the HTTP Upgrade handshake (validating the 101), sends
// caller-provided text messages as masked frames on absolute deadlines (a
// per-loop deadline heap — no per-tick scan of all connections), and reads
// server frames — timestamping each at socket-read time.
// ===========================================================================

struct WsResult {
  int index = -1;
  std::vector<double> offsets_ms;  // per data-frame arrival offset (ms from send)
  std::vector<int> sizes;          // per data-frame payload size (bytes)
  std::string content;             // concatenated text/binary payloads
  std::vector<std::string> frames;  // per data-frame payload (for protocol parse)
  std::vector<double> sent_offsets_ms;  // actual send offset of each paced message
  double dispatch_offset_ms = 0.0;  // connect initiation (ms from run start)
  std::string error;
};

struct WsConn {
  int index = -1;
  bool connected = false;
  bool handshaken = false;
  bool done = false;
  bool failed = false;
  bool timer_armed = false;
  double send_time = 0.0;  // when the first message was (or should be) sent
  size_t next_send = 0;    // next paced message index to send
  int frag_opcode = 0;     // opcode of an in-progress fragmented message
  std::string frag_buf;    // accumulated fragmented payload
  std::string inbuf;
  std::string outbuf;
  std::vector<double> offsets;
  std::vector<int> sizes;
  std::string content;
  std::vector<std::string> frames;
  std::vector<double> sent_offsets;
  // Terminal event markers: a realtime server may keep the session open after
  // finishing a response (the real OpenAI-realtime contract), so waiting for a
  // close frame would stall until timeout. Finishing on the same terminal event
  // the Python client stops at keeps both paths' notion of "complete" identical.
  const std::vector<std::string>* done_markers = nullptr;
};

// Encode one client->server frame, masked per RFC 6455 (opcode 0x1 = text,
// 0xA = pong). Masking must be present for client frames; the key value need
// not be random for protocol correctness.
static std::string ws_encode_frame(unsigned char opcode,
                                   const std::string& payload) {
  std::string frame;
  frame.push_back((char)(0x80 | opcode));  // FIN + opcode
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
  const unsigned char mask[4] = {0x12, 0x34, 0x56, 0x78};
  for (int i = 0; i < 4; i++) frame.push_back((char)mask[i]);
  for (size_t i = 0; i < n; i++)
    frame.push_back((char)(payload[i] ^ mask[i % 4]));
  return frame;
}

// Pull complete server frames out of inbuf, timestamping data frames at read
// time. Handles fragmentation across recv() boundaries, continuation frames
// (opcode 0x0), 126/127 extended lengths, and ping->pong. Sets c.done on a
// close frame (0x8).
static void ws_parse_frames(WsConn& c) {
  double ts = now_ms();
  auto emit = [&](std::string&& payload) {
    c.offsets.push_back(ts - c.send_time);
    c.sizes.push_back((int)payload.size());
    c.content += payload;
    if (c.done_markers) {
      for (const auto& marker : *c.done_markers) {
        if (payload.find(marker) != std::string::npos) {
          c.done = true;
          break;
        }
      }
    }
    c.frames.push_back(std::move(payload));
  };
  while (true) {
    if (c.inbuf.size() < 2) return;
    const unsigned char* p = (const unsigned char*)c.inbuf.data();
    unsigned char b0 = p[0], b1 = p[1];
    bool fin = b0 & 0x80;
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
    if (opcode == 0x9) {  // ping -> queue pong with the same payload
      c.outbuf += ws_encode_frame(0xA, payload);
    } else if (opcode == 0x1 || opcode == 0x2) {  // text / binary data frame
      if (fin) {
        emit(std::move(payload));
      } else {
        c.frag_opcode = opcode;
        c.frag_buf = std::move(payload);
      }
    } else if (opcode == 0x0 && c.frag_opcode != 0) {  // continuation
      c.frag_buf += payload;
      if (fin) {
        emit(std::move(c.frag_buf));
        c.frag_buf.clear();
        c.frag_opcode = 0;
      }
    }
    // pong (0xA): ignored.
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

// Consume the handshake response once complete. Returns false (and sets the
// error) unless the server answered 101 Switching Protocols; bytes after the
// header terminator remain in inbuf as the first frames. (Sec-WebSocket-Accept
// is not cryptographically verified — status + Upgrade is sufficient for the
// controlled endpoints this engine targets.)
static bool ws_check_handshake(WsConn& c, std::string* error) {
  size_t term = c.inbuf.find("\r\n\r\n");
  if (term == std::string::npos) return true;  // need more bytes
  int status = 0;
  size_t sp = c.inbuf.find(' ');
  if (sp != std::string::npos && sp < term) status = atoi(c.inbuf.c_str() + sp + 1);
  if (status != 101) {
    *error = "ws handshake failed (status " + std::to_string(status) + ")";
    c.failed = true;
    return false;
  }
  c.inbuf.erase(0, term + 4);
  c.handshaken = true;
  c.send_time = now_ms();  // pacing base = handshake completion
  return true;
}

// Send every paced message whose absolute deadline has arrived; record each
// actual send offset (the per-dispatch send-drift the interactivity path is
// sensitive to). Returns the ABSOLUTE next deadline (now_ms domain), or a
// huge value when nothing is pending. A non-empty outbuf (kernel buffer full)
// defers new frames to the writable flush, which re-pumps.
static double ws_pump_sends(int fd, WsConn& c,
                            const std::vector<std::string>& messages,
                            const std::vector<double>& send_offsets_ms) {
  const double FAR = 1e18;
  if (c.next_send >= messages.size()) return FAR;
  if (!c.outbuf.empty()) {
    if (!flush_outbuf(fd, c.outbuf)) {
      c.failed = true;
      return FAR;
    }
    if (!c.outbuf.empty()) return FAR;  // wait for writable; flush re-pumps
  }
  double now = now_ms();
  while (c.next_send < messages.size()) {
    double due = send_offsets_ms.empty() ? 0.0 : send_offsets_ms[c.next_send];
    double deadline = c.send_time + due;
    if (now + 0.05 >= deadline) {
      std::string frame = ws_encode_frame(0x1, messages[c.next_send]);
      if (!conn_send(fd, c.outbuf, frame.data(), frame.size())) {
        c.failed = true;
        return FAR;
      }
      c.sent_offsets.push_back(now - c.send_time);
      c.next_send++;
      if (!c.outbuf.empty()) return FAR;  // backlogged: resume on writable
      now = now_ms();
    } else {
      return deadline;  // absolute next deadline
    }
  }
  return FAR;
}

// Move a finished/failed connection's data into its result slot.
static void ws_finish(WsConn& c, WsResult& res) {
  res.offsets_ms = std::move(c.offsets);
  res.sizes = std::move(c.sizes);
  res.content = std::move(c.content);
  res.frames = std::move(c.frames);
  res.sent_offsets_ms = std::move(c.sent_offsets);
}

// Shared reactor WS loop over an existing connection map. Paced sends are
// deadline-heap driven (ids: fds), so an idle wakeup touches only due
// connections. ws_stream (one shared sequence) and ws_run_batch (per-request
// sequences) share it via the callbacks.
template <typename MsgsFn, typename OffsFn, typename RefillFn>
static void ws_reactor_loop(std::unordered_map<int, WsConn>& conns,
                            Reactor& reactor, std::vector<WsResult>& results,
                            const std::string& handshake, int total,
                            int& completed, double timeout_s, double t0,
                            MsgsFn msgs_for, OffsFn offs_for, RefillFn refill) {
  TimerHeap pacing;

  auto arm = [&](int fd, WsConn& c, double abs_deadline) {
    if (abs_deadline < 1e17 && !c.timer_armed) {
      pacing.push(abs_deadline, fd);
      c.timer_armed = true;
    }
  };
  auto pump = [&](int fd, WsConn& c) {
    double next = ws_pump_sends(fd, c, msgs_for(c.index), offs_for(c.index));
    reactor.set_interest(fd, true, !c.outbuf.empty());
    arm(fd, c, next);
  };
  auto finish = [&](int fd, WsConn& c) {
    ws_finish(c, results[c.index]);
    reactor.remove(fd);
    close(fd);
    conns.erase(fd);
    completed++;
  };

  std::vector<ReactorEvent> evs;
  std::vector<int> due;
  while (completed < total && (now_ms() - t0) < timeout_s * 1000.0) {
    double until = pacing.empty() ? 50.0 : pacing.next_deadline() - now_ms();
    int timeout_ms = (int)std::max(0.0, std::min(50.0, until));
    reactor.wait(timeout_ms, evs);
    for (auto& ev : evs) {
      auto it = conns.find(ev.fd);
      if (it == conns.end()) continue;
      WsConn& c = it->second;
      int fd = ev.fd;
      if (c.failed) {
        finish(fd, c);
        continue;
      }
      if (!c.connected && (ev.writable || ev.error)) {
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
          results[c.index].error = "connect error";
          finish(fd, c);
          continue;
        }
        c.connected = true;
        if (!conn_send(fd, c.outbuf, handshake.data(), handshake.size())) {
          results[c.index].error = "send error";
          finish(fd, c);
          continue;
        }
        reactor.set_interest(fd, true, !c.outbuf.empty());
        continue;
      }
      if (!c.connected) continue;
      if (ev.writable) {
        if (!flush_outbuf(fd, c.outbuf)) {
          results[c.index].error = "send error";
          finish(fd, c);
          continue;
        }
        if (c.handshaken && !c.failed) {
          pump(fd, c);
        } else {
          reactor.set_interest(fd, true, !c.outbuf.empty());
        }
      }
      if (ev.readable) {
        char buf[16384];
        ssize_t r = recv_retry(fd, buf, sizeof(buf));
        if (r > 0) {
          c.inbuf.append(buf, r);
          if (!c.handshaken &&
              !ws_check_handshake(c, &results[c.index].error)) {
            finish(fd, c);
            continue;
          }
          if (c.handshaken) {
            pump(fd, c);
            ws_parse_frames(c);
          }
          if (c.done || c.failed) finish(fd, c);
        } else if (r == 0) {
          finish(fd, c);
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          results[c.index].error = "recv error";
          finish(fd, c);
        }
      } else if (ev.error) {
        finish(fd, c);
      }
    }
    // fire due paced sends (heap-driven: only due connections are touched)
    pacing.pop_due(now_ms(), due);
    for (int fd : due) {
      auto it = conns.find(fd);
      if (it == conns.end()) continue;  // stale
      it->second.timer_armed = false;
      if (it->second.handshaken && !it->second.failed)
        pump(fd, it->second);
    }
    refill();
  }
  for (auto& kv : conns) {
    WsResult& res = results[kv.second.index];
    // capture whatever arrived before timeout
    if (res.offsets_ms.empty()) ws_finish(kv.second, res);
    if (res.error.empty()) res.error = "timeout";
    close(kv.first);
  }
  conns.clear();
}

static std::vector<WsResult> ws_stream(const std::string& host, int port,
                                       const std::string& path,
                                       const std::vector<std::string>& init_messages,
                                       int concurrency, double timeout_s,
                                       const std::vector<double>& send_offsets_ms) {
  raise_nofile(concurrency);

  int total = concurrency;  // one connection per concurrency slot
  std::vector<WsResult> results(total);
  for (int i = 0; i < total; i++) results[i].index = i;
  std::string handshake = ws_handshake(host, port, path);

  sockaddr_in addr;
  if (!resolve_addr(host, port, &addr)) {
    for (auto& r : results) r.error = "resolve failed: " + host;
    return results;
  }

  Reactor reactor;
  std::unordered_map<int, WsConn> conns;
  int completed = 0;
  double t0 = now_ms();
  for (int i = 0; i < total; i++) {
    int fd = make_conn(addr);
    if (fd < 0) {
      results[i].error = "connect failed";
      completed++;  // count failures so the loop still waits for live conns
      continue;
    }
    WsConn c;
    c.index = i;
    results[i].dispatch_offset_ms = now_ms() - t0;
    conns.emplace(fd, std::move(c));
    reactor.set_interest(fd, false, true);
  }

  ws_reactor_loop(
      conns, reactor, results, handshake, total, completed, timeout_s, t0,
      [&](int) -> const std::vector<std::string>& { return init_messages; },
      [&](int) -> const std::vector<double>& { return send_offsets_ms; },
      [] {});
  return results;
}

// One shard's WS batch loop (mirrors run_batch's sharding): serves the request
// indices in `indices` with its own reactor + pacing heap, refilling from its
// own launch cursor. Disjoint result slots — no locks.
static void ws_run_batch_worker(
    const sockaddr_in& addr, const std::string& handshake,
    const std::vector<std::vector<std::string>>& req_messages,
    const std::vector<std::vector<double>>& req_offsets,
    const std::vector<int>& indices, int concurrency, double timeout_s,
    double t0, std::vector<WsResult>& results,
    const std::vector<std::string>& done_markers) {
  set_thread_qos();
  Reactor reactor;
  std::unordered_map<int, WsConn> conns;
  int total = (int)indices.size();
  size_t cur = 0;
  int completed = 0;

  auto try_launch = [&]() {
    while ((int)conns.size() < concurrency && cur < indices.size()) {
      int idx = indices[cur];
      int fd = make_conn(addr);
      if (fd < 0) {
        results[idx].error = "connect failed";
        completed++;
        cur++;
        continue;
      }
      WsConn c;
      c.index = idx;
      c.done_markers = &done_markers;
      results[idx].dispatch_offset_ms = now_ms() - t0;
      conns.emplace(fd, std::move(c));
      reactor.set_interest(fd, false, true);
      cur++;
    }
  };
  try_launch();

  ws_reactor_loop(
      conns, reactor, results, handshake, total, completed, timeout_s, t0,
      [&](int idx) -> const std::vector<std::string>& {
        return req_messages[idx];
      },
      [&](int idx) -> const std::vector<double>& { return req_offsets[idx]; },
      try_launch);
}

// Per-connection WS batch with a concurrency cap + refill, sharded across
// `num_threads` reactor threads: each request has its own message sequence +
// send schedule, so N distinct audio requests run concurrently.
static std::vector<WsResult> ws_run_batch(
    const std::string& host, int port, const std::string& path,
    const std::vector<std::vector<std::string>>& req_messages,
    const std::vector<std::vector<double>>& req_offsets, int concurrency,
    double timeout_s, int num_threads,
    const std::vector<std::string>& done_markers) {
  raise_nofile(concurrency);

  int total = (int)req_messages.size();
  std::vector<WsResult> results(total);
  for (int i = 0; i < total; i++) results[i].index = i;
  std::string handshake = ws_handshake(host, port, path);
  if (total == 0) return results;

  sockaddr_in addr;
  if (!resolve_addr(host, port, &addr)) {
    for (auto& r : results) r.error = "resolve failed: " + host;
    return results;
  }

  int nthreads = std::max(1, num_threads);
  if (nthreads > total) nthreads = total;
  double t0 = now_ms();

  if (nthreads == 1) {
    std::vector<int> all(total);
    for (int i = 0; i < total; i++) all[i] = i;
    ws_run_batch_worker(addr, handshake, req_messages, req_offsets, all,
                        concurrency, timeout_s, t0, results, done_markers);
    return results;
  }
  std::vector<std::vector<int>> shards(nthreads);
  for (int i = 0; i < total; i++) shards[i % nthreads].push_back(i);
  int per_thread_conc = (concurrency + nthreads - 1) / nthreads;
  if (per_thread_conc < 1) per_thread_conc = 1;
  std::vector<std::thread> pool;
  pool.reserve(nthreads);
  for (int t = 0; t < nthreads; t++) {
    pool.emplace_back([&, t]() {
      ws_run_batch_worker(addr, handshake, req_messages, req_offsets,
                          shards[t], per_thread_conc, timeout_s, t0, results,
                          done_markers);
    });
  }
  for (auto& th : pool) th.join();
  return results;
}

static std::vector<WsResult> py_ws_run_batch(
    const std::string& host, int port, const std::string& path,
    const std::vector<std::vector<std::string>>& req_messages,
    const std::vector<std::vector<double>>& req_offsets, int concurrency,
    double timeout_s, int num_threads,
    const std::vector<std::string>& done_markers) {
  py::gil_scoped_release release;
  return ws_run_batch(host, port, path, req_messages, req_offsets, concurrency,
                      timeout_s, num_threads, done_markers);
}

static std::vector<WsResult> py_ws_stream(
    const std::string& host, int port, const std::string& path,
    const std::vector<std::string>& init_messages, int concurrency,
    double timeout_s, const std::vector<double>& send_offsets_ms) {
  py::gil_scoped_release release;
  return ws_stream(host, port, path, init_messages, concurrency, timeout_s,
                   send_offsets_ms);
}

// ---------------------------------------------------------------------------
// NativeSseServer — a native (C++) reference SSE server for drift VALIDATION.
//
// Emits each chunk on an ABSOLUTE monotonic deadline, reactor + emit-heap
// driven (wakeups touch only due connections), from one QoS-pinned thread with
// no GIL / coroutine overhead — so its own emit jitter stays sub-ms even under
// core contention. It is the reference that lets us measure the native
// CLIENT's true fidelity instead of the Python mock's punctuality floor.
// ---------------------------------------------------------------------------
struct SseServerConn {
  bool headers_sent = false;
  bool finished = false;
  double start = 0.0;  // monotonic ms when headers were sent (schedule origin)
  int next_chunk = 0;
  std::string reqbuf;
  std::string outbuf;  // unsent tail of a partially written emit
};

class NativeSseServer {
 public:
  NativeSseServer(int num_chunks, double cadence_ms, double prefill_ms,
                  int chunk_bytes)
      : num_chunks_(num_chunks),
        cadence_ms_(cadence_ms),
        prefill_ms_(prefill_ms),
        chunk_bytes_(chunk_bytes) {}

  ~NativeSseServer() { stop(); }

  int start() {
    listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("socket() failed");
    int one = 1;
    setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    if (bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0)
      throw std::runtime_error("bind() failed");
    socklen_t len = sizeof(addr);
    getsockname(listen_fd_, (sockaddr*)&addr, &len);
    port_ = ntohs(addr.sin_port);
    listen(listen_fd_, 1024);
    set_nonblock(listen_fd_);

    std::string content(std::max(1, chunk_bytes_), 'x');
    data_line_ = "data: {\"choices\":[{\"delta\":{\"content\":\"" + content +
                 "\"}}],\"model\":\"mock\"}\n\n";
    done_line_ = "data: [DONE]\n\n";
    headers_ =
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\nConnection: close\r\n\r\n";

    stop_.store(false);
    thread_ = std::thread([this] { this->run(); });
    return port_;
  }

  void stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    if (listen_fd_ >= 0) {
      close(listen_fd_);
      listen_fd_ = -1;
    }
  }

  // Peak simultaneous connections, the same server-side ground truth the
  // Python mock engine reports — so a preflight workload can be pointed at
  // either server and gated on achieved concurrency identically.
  int max_active_conns() const {
    return max_active_conns_.load(std::memory_order_relaxed);
  }

  void reset_telemetry() {
    std::lock_guard<std::mutex> g(lat_mutex_);
    lateness_.clear();
    max_active_conns_.store(0, std::memory_order_relaxed);
  }

  double server_jitter_p99_ms() {
    std::lock_guard<std::mutex> g(lat_mutex_);
    if (lateness_.empty()) return 0.0;
    std::vector<double> v = lateness_;
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)((v.size() - 1) * 0.99);
    return v[idx];
  }

 private:
  static void set_nonblock(int fd) {
    int f = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, f | O_NONBLOCK);
  }

  void run() {
    set_thread_qos();
    Reactor reactor;
    TimerHeap emits;
    std::unordered_map<int, SseServerConn> conns;
    std::vector<double> local_lat;
    reactor.set_interest(listen_fd_, true, false);

    auto drop = [&](int fd) {
      reactor.remove(fd);
      close(fd);
      conns.erase(fd);
    };

    // Emit every due chunk for one connection; arm its next deadline.
    auto emit_due = [&](int fd, SseServerConn& c) {
      if (c.finished) return;
      if (!c.outbuf.empty() && !flush_outbuf(fd, c.outbuf)) {
        drop(fd);
        return;
      }
      double now = now_ms();
      while (c.next_chunk < num_chunks_ && c.outbuf.empty()) {
        double dl = c.start + prefill_ms_ + c.next_chunk * cadence_ms_;
        if (now < dl) {
          emits.push(dl, fd);
          reactor.set_interest(fd, true, false);
          return;
        }
        if (!conn_send(fd, c.outbuf, data_line_.data(), data_line_.size())) {
          drop(fd);
          return;
        }
        local_lat.push_back(now - dl);  // emit lateness = server jitter
        c.next_chunk++;
        now = now_ms();
      }
      if (!c.outbuf.empty()) {
        reactor.set_interest(fd, true, true);  // resume on writable
        return;
      }
      if (c.next_chunk >= num_chunks_) {
        conn_send(fd, c.outbuf, done_line_.data(), done_line_.size());
        c.finished = true;
        drop(fd);
      }
    };

    std::vector<ReactorEvent> evs;
    std::vector<int> due;
    while (!stop_.load()) {
      double until = emits.empty() ? 5.0 : emits.next_deadline() - now_ms();
      int timeout = (int)std::max(0.0, std::min(5.0, until));
      reactor.wait(timeout, evs);
      for (auto& ev : evs) {
        if (ev.fd == listen_fd_) {
          while (true) {
            int cfd = accept(listen_fd_, nullptr, nullptr);
            if (cfd < 0) break;
            set_nonblock(cfd);
            int one = 1;
            setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
            conns.emplace(cfd, SseServerConn{});
            {
              int active = (int)conns.size();
              int prev = max_active_conns_.load(std::memory_order_relaxed);
              while (active > prev &&
                     !max_active_conns_.compare_exchange_weak(
                         prev, active, std::memory_order_relaxed)) {
              }
            }
            reactor.set_interest(cfd, true, false);
          }
          continue;
        }
        auto it = conns.find(ev.fd);
        if (it == conns.end()) continue;
        SseServerConn& c = it->second;
        if (!c.headers_sent && ev.readable) {
          char buf[4096];
          ssize_t r = recv_retry(ev.fd, buf, sizeof(buf));
          if (r == 0) {
            drop(ev.fd);
            continue;
          }
          if (r > 0) {
            c.reqbuf.append(buf, (size_t)r);
            if (c.reqbuf.find("\r\n\r\n") != std::string::npos) {
              if (!conn_send(ev.fd, c.outbuf, headers_.data(),
                             headers_.size())) {
                drop(ev.fd);
                continue;
              }
              c.headers_sent = true;
              c.start = now_ms();
              emit_due(ev.fd, c);
            }
          }
          continue;
        }
        if (ev.writable && c.headers_sent) emit_due(ev.fd, c);
        if (ev.readable && c.headers_sent) {
          // client went away or sent extra bytes; probe cheaply
          char buf[1024];
          ssize_t r = recv_retry(ev.fd, buf, sizeof(buf));
          if (r == 0) drop(ev.fd);
        }
      }
      emits.pop_due(now_ms(), due);
      for (int fd : due) {
        auto it = conns.find(fd);
        if (it != conns.end()) emit_due(fd, it->second);
      }
      if (local_lat.size() >= 512) {
        std::lock_guard<std::mutex> g(lat_mutex_);
        lateness_.insert(lateness_.end(), local_lat.begin(), local_lat.end());
        local_lat.clear();
      }
    }
    {
      std::lock_guard<std::mutex> g(lat_mutex_);
      lateness_.insert(lateness_.end(), local_lat.begin(), local_lat.end());
    }
    for (auto& kv : conns) close(kv.first);
  }

  int num_chunks_;
  double cadence_ms_, prefill_ms_;
  int chunk_bytes_;
  int listen_fd_ = -1, port_ = 0;
  std::string data_line_, done_line_, headers_;
  std::thread thread_;
  std::atomic<bool> stop_{false};
  std::mutex lat_mutex_;
  std::atomic<int> max_active_conns_{0};
  std::vector<double> lateness_;
};

PYBIND11_MODULE(veeksha_native, m, py::mod_gil_not_used()) {
  m.doc() = "Native (C++) transport + timing engine for Veeksha.";

#if defined(VEEKSHA_USE_KQUEUE)
  m.attr("reactor_backend") = "kqueue";
#elif defined(VEEKSHA_USE_EPOLL)
  m.attr("reactor_backend") = "epoll";
#else
  m.attr("reactor_backend") = "poll";
#endif

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
        "Per-request engine: owns connection concurrency over reactor loop(s) "
        "(kqueue/epoll; poll fallback), sends caller-built HTTP requests "
        "(chunked transfer-encoding decoded), returns per-request ReqResult "
        "(status, content, read-time chunk offsets_ms + sizes, actual "
        "dispatch_offset_ms). With dispatch_offsets_ms it runs OPEN-LOOP: each "
        "request is launched on its arrival deadline. num_threads>1 shards the "
        "connections across that many native reactor threads.");

  py::class_<ChainTurnSpec>(m, "ChainTurnSpec")
      .def(py::init<>())
      .def_readwrite("header_prefix", &ChainTurnSpec::header_prefix)
      .def_readwrite("header_suffix", &ChainTurnSpec::header_suffix)
      .def_readwrite("body_segments", &ChainTurnSpec::body_segments)
      .def_readwrite("hole_refs", &ChainTurnSpec::hole_refs)
      .def_readwrite("delay_ms", &ChainTurnSpec::delay_ms)
      .doc() =
      "One turn's request template: wire = header_prefix + Content-Length + "
      "header_suffix + body, where body interleaves body_segments with the "
      "JSON-escaped content of prior turns (hole_refs). delay_ms = think time "
      "after the prior turn completes.";

  py::class_<ChainTurn>(m, "ChainTurn")
      .def_readonly("offsets_ms", &ChainTurn::offsets_ms)
      .def_readonly("content", &ChainTurn::content)
      .def_readonly("status", &ChainTurn::status)
      .def_readonly("dispatch_offset_ms", &ChainTurn::dispatch_offset_ms);

  py::class_<ChainResult>(m, "ChainResult")
      .def_readonly("index", &ChainResult::index)
      .def_readonly("turns", &ChainResult::turns)
      .def_readonly("handoff_ms", &ChainResult::handoff_ms)
      .def_readonly("error", &ChainResult::error);

  m.def("run_chains", &py_run_chains, py::arg("host"), py::arg("port"),
        py::arg("chains"), py::arg("concurrency"), py::arg("timeout_s") = 120.0,
        py::arg("start_offsets_ms") = std::vector<double>(),
        py::arg("num_threads") = 1,
        "Multi-turn chains with native receive->dispatch coupling AND "
        "inter-turn content flow: each turn is a ChainTurnSpec template whose "
        "holes native fills with prior turns' outputs (JSON-escaped, "
        "Content-Length recomputed). Turn N+1 fires at (turn N complete + "
        "delay_ms); handoff_ms records each scheduled dispatch's lateness vs "
        "its deadline. start_offsets_ms (per chain) makes chain STARTS "
        "open-loop; otherwise chains start closed-loop under the concurrency "
        "cap. num_threads shards chains across reactor threads.");

  py::class_<WsResult>(m, "WsResult")
      .def_readonly("index", &WsResult::index)
      .def_readonly("offsets_ms", &WsResult::offsets_ms)
      .def_readonly("sizes", &WsResult::sizes)
      .def_readonly("content", &WsResult::content)
      .def_readonly("frames", &WsResult::frames)
      .def_readonly("sent_offsets_ms", &WsResult::sent_offsets_ms)
      .def_readonly("dispatch_offset_ms", &WsResult::dispatch_offset_ms)
      .def_readonly("error", &WsResult::error);

  m.def("ws_stream", &py_ws_stream, py::arg("host"), py::arg("port"),
        py::arg("path"), py::arg("init_messages"), py::arg("concurrency"),
        py::arg("timeout_s") = 120.0,
        py::arg("send_offsets_ms") = std::vector<double>(),
        "Native WebSocket receive + paced send: handshake (101 validated), send "
        "each init message as a masked frame on its absolute deadline "
        "(send_offsets_ms, ms from handshake), read server frames (continuation "
        "+ ping/pong handled) timestamped at socket-read time. Returns one "
        "WsResult per connection.");

  m.def("ws_run_batch", &py_ws_run_batch, py::arg("host"), py::arg("port"),
        py::arg("path"), py::arg("req_messages"), py::arg("req_offsets"),
        py::arg("concurrency"), py::arg("timeout_s") = 120.0,
        py::arg("num_threads") = 1,
        py::arg("done_markers") = std::vector<std::string>(),
        "Per-connection WS batch with concurrency cap + refill: each request has "
        "its own message sequence + paced send schedule (deadline-heap driven). "
        "num_threads>1 shards requests across native reactor threads. A stream "
        "ends on a close frame or on the first frame containing any of "
        "done_markers - realtime servers keep the session open after a response, "
        "so waiting for close alone would stall until timeout. Returns a "
        "WsResult per request (index-aligned to req_messages).");

  py::class_<NativeSseServer>(m, "NativeSseServer")
      .def(py::init<int, double, double, int>(), py::arg("num_chunks"),
           py::arg("cadence_ms"), py::arg("prefill_ms") = 20.0,
           py::arg("chunk_bytes") = 1)
      .def("start", &NativeSseServer::start,
           "Bind a loopback port, spawn the emit thread, return the port.")
      .def("stop", &NativeSseServer::stop, "Stop the emit thread and close.")
      .def("max_active_conns", &NativeSseServer::max_active_conns,
           "Peak simultaneous connections observed (server-side ground truth).")
      .def("reset_telemetry", &NativeSseServer::reset_telemetry,
           "Clear emit-lateness samples and the peak-connection counter.")
      .def("server_jitter_p99_ms", &NativeSseServer::server_jitter_p99_ms,
           "p99 of per-chunk emit lateness (actual - scheduled) in ms.")
      .doc() =
      "Native reference SSE server: emits num_chunks per connection on absolute "
      "monotonic deadlines from a QoS-pinned reactor thread, so its emit jitter "
      "is sub-ms. A validation tool to measure the native client's true drift "
      "without the Python reference server's punctuality floor.";
}
