// veeksha_native transport layer — reactor, timers, sockets, HTTP/1.1 (via
// vendored llhttp) + SSE state machine, WebSocket client codec, JSON
// escape/unescape.
//
// Transplanted from the audited veeksha-modality-abstraction-v3 native code
// (native_receiver.cpp): reactor kqueue/epoll/poll behind one class, TimerHeap
// deadline min-heap, HTTP/1.1 + SSE state machine with read-time
// CLOCK_MONOTONIC stamps, chat delta.content extraction with full RFC 8259
// JSON unescape (incl. surrogate pairs), WS client codec (handshake / Upgrade
// validation / fragmentation / ping-pong / masked frames), EINTR /
// partial-send / malformed-framing hardening, set_thread_qos for Apple
// efficiency-core avoidance, raise_nofile.
//
// HTTP response FRAMING (status line, headers, chunked transfer decoding,
// message completion) is delegated to vendored llhttp
// (third_party/llhttp, HTTP_RESPONSE mode). SSE `data:` line framing, chat
// delta extraction, JSON string handling and the WS codec stay in-tree by
// design — they are benchmark-payload concerns, not HTTP framing.
//
// Local adaptations for the benchmark loop (documented inline):
//   - http_feed returns the number of NEW decoded body bytes this call so the
//     TTS_HTTP path can stamp per-chunk arrivals without accumulating audio.
//   - the WS frame parser is generalized to a callback (ws_extract_frames) so
//     the native loop owns per-transport stamping/extraction.
//
// Timestamps are taken immediately after each recv() returns. Kernel cmsg
// receive timestamps (SO_TIMESTAMP*) were evaluated and rejected on the prior
// branch: not delivered for TCP stream sockets on darwin, so read-time
// monotonic stamping is what the drift benches actually validated.
#pragma once

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/socket.h>
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
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "llhttp.h"

namespace veeksha_native_transport {

inline double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// Pin the calling thread to a high QoS class on Apple platforms. macOS parks
// default-QoS threads of an idle process on efficiency cores, which adds
// ~2-8 ms of wakeup slop to timer-driven emits/paced sends at LOW load — the
// exact tail the drift benches see on a quiet dev box.
inline void set_thread_qos() {
#if defined(__APPLE__)
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

inline void raise_nofile(int need) {
  struct rlimit rl;
  getrlimit(RLIMIT_NOFILE, &rl);
  rl.rlim_cur = std::max<rlim_t>(rl.rlim_cur, (rlim_t)(need + 64));
  if (rl.rlim_cur > rl.rlim_max) rl.rlim_cur = rl.rlim_max;
  setrlimit(RLIMIT_NOFILE, &rl);
}

// Resolve host:port once per run. Returns false on failure — callers must
// surface that as a per-request error, never fall through to 0.0.0.0.
// IPv4 only (first A record).
inline bool resolve_addr(const std::string& host, int port, sockaddr_in* out) {
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

inline int make_conn(const sockaddr_in& addr) {
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
// unsent tail so the reactor loop can flush it on writability. Returns false
// on a hard socket error.
inline bool conn_send(int fd, std::string& outbuf, const char* data,
                      size_t len) {
  if (outbuf.empty()) {
    size_t off = 0;
    while (off < len) {
      ssize_t w = ::send(fd, data + off, len - off, 0);
      if (w > 0) {
        off += (size_t)w;
        continue;
      }
      if (w == 0) return false;  // pathological; errno is stale here
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

inline bool flush_outbuf(int fd, std::string& outbuf) {
  size_t off = 0;
  while (off < outbuf.size()) {
    ssize_t w = ::send(fd, outbuf.data() + off, outbuf.size() - off, 0);
    if (w > 0) {
      off += (size_t)w;
      continue;
    }
    if (w == 0) return false;  // pathological; errno is stale here
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) break;
    return false;
  }
  outbuf.erase(0, off);
  return true;
}

// recv() with EINTR retry. Returns like recv(); EAGAIN/EWOULDBLOCK pass
// through.
inline ssize_t recv_retry(int fd, char* buf, size_t len) {
  while (true) {
    ssize_t r = recv(fd, buf, len, 0);
    if (r < 0 && errno == EINTR) continue;
    return r;
  }
}

// ===========================================================================
// Reactor — readiness notification behind one interface (kqueue on
// BSD/darwin, epoll on Linux, POSIX poll() as the portable fallback;
// VEEKSHA_NATIVE_REACTOR=poll selects poll() at runtime for A/B measurement).
// ===========================================================================

struct ReactorEvent {
  int fd = -1;
  bool readable = false;
  bool writable = false;
  bool error = false;
};

inline bool use_kernel_queue() {
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
      struct epoll_event e {};  // ignored by DEL, but must be a valid pointer
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

// Deadline min-heap keyed by (deadline_ms, id). Entries are lazily
// invalidated: consumers re-check the owning object's state on pop, so stale
// entries (a closed connection, a rescheduled deadline) cost one pop each.
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
// HTTP streaming response state. FRAMING (status line, headers, chunked
// transfer decoding, message completion) is llhttp's job (HTTP_RESPONSE
// mode); decoded body bytes land in inbuf where the in-tree SSE `data:` line
// framer consumes them with read-time timestamps.
// ===========================================================================

struct HttpStreamState {
  bool headers_done = false;
  bool done = false;  // stream finished ([DONE] seen or HTTP message ended)
  bool sse = true;
  bool body_done = false;  // llhttp on_message_complete fired
  bool malformed = false;  // llhttp rejected the framing; the loop fails the
                           // request (reason carries llhttp_errno_name)
  int status = 0;
  double send_time = 0.0;
  std::string reason;   // malformed detail: llhttp errno name + reason
  std::string inbuf;    // decoded body bytes awaiting framing
  std::string content;  // extracted assistant text for chat SSE (delta.content
                        // only; raw payloads for non-chat SSE, raw body if
                        // !sse)
  std::vector<double> offsets;  // per-event arrival offset (ms from send)
  std::vector<int> sizes;       // per-event payload size (bytes)

  // llhttp parser. Plain-data C struct: moving HttpStreamState (Conn lives in
  // an unordered_map and is moved on finish) is safe because parser.data is
  // re-pointed at the owning state on every http_feed/http_eof entry, and the
  // settings table is a process-wide static (llhttp_init stores its pointer).
  llhttp_t parser;
  bool parser_init = false;
};

namespace llhttp_detail {

inline int on_headers_complete(llhttp_t* p) {
  auto* s = static_cast<HttpStreamState*>(p->data);
  s->status = (int)llhttp_get_status_code(p);
  s->headers_done = true;
  return 0;
}

// llhttp decodes chunked transfer-encoding internally and delivers only
// decoded body bytes here; the caller attributes them to the read-time stamp
// of the recv() that produced this execute call.
inline int on_body(llhttp_t* p, const char* at, size_t length) {
  auto* s = static_cast<HttpStreamState*>(p->data);
  s->inbuf.append(at, length);
  return 0;
}

inline int on_message_complete(llhttp_t* p) {
  auto* s = static_cast<HttpStreamState*>(p->data);
  s->body_done = true;
  return 0;
}

inline const llhttp_settings_t* settings() {
  static const llhttp_settings_t st = [] {
    llhttp_settings_t v;
    llhttp_settings_init(&v);
    v.on_headers_complete = on_headers_complete;
    v.on_body = on_body;
    v.on_message_complete = on_message_complete;
    return v;
  }();
  return &st;
}

}  // namespace llhttp_detail

// Record an llhttp framing rejection: malformed flag + a reason string built
// from llhttp_errno_name (and the parser's detail message when present).
// Malformed framing always fails the request — never a silent completion.
inline void http_mark_malformed(HttpStreamState& s, llhttp_errno_t err) {
  s.malformed = true;
  s.reason = llhttp_errno_name(err);
  const char* detail = llhttp_get_error_reason(&s.parser);
  if (detail != nullptr && detail[0] != '\0') {
    s.reason += ": ";
    s.reason += detail;
  }
}

// Decode a JSON string literal starting at src[i] (the char after the opening
// quote). Appends the decoded value to `out`; returns the index just past the
// closing quote, or std::string::npos on malformed input. Handles the full
// RFC 8259 escape set including \uXXXX surrogate pairs (encoded to UTF-8).
inline size_t json_unescape_into(const std::string& src, size_t i,
                                 std::string& out) {
  auto hex4 = [&](size_t p, unsigned* v) -> bool {
    if (p + 4 > src.size()) return false;
    unsigned r = 0;
    for (int k = 0; k < 4; k++) {
      char ch = src[p + k];
      r <<= 4;
      if (ch >= '0' && ch <= '9')
        r |= (unsigned)(ch - '0');
      else if (ch >= 'a' && ch <= 'f')
        r |= (unsigned)(ch - 'a' + 10);
      else if (ch >= 'A' && ch <= 'F')
        r |= (unsigned)(ch - 'A' + 10);
      else
        return false;
    }
    *v = r;
    return true;
  };
  auto put_utf8 = [&](unsigned cp) {
    if (cp < 0x80) {
      out.push_back((char)cp);
    } else if (cp < 0x800) {
      out.push_back((char)(0xC0 | (cp >> 6)));
      out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
      out.push_back((char)(0xE0 | (cp >> 12)));
      out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
      out.push_back((char)(0xF0 | (cp >> 18)));
      out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back((char)(0x80 | (cp & 0x3F)));
    }
  };
  while (i < src.size()) {
    char ch = src[i];
    if (ch == '"') return i + 1;
    if (ch != '\\') {
      out.push_back(ch);
      i++;
      continue;
    }
    if (i + 1 >= src.size()) return std::string::npos;
    char esc = src[i + 1];
    i += 2;
    switch (esc) {
      case '"':
        out.push_back('"');
        break;
      case '\\':
        out.push_back('\\');
        break;
      case '/':
        out.push_back('/');
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'u': {
        unsigned cp = 0;
        if (!hex4(i, &cp)) return std::string::npos;
        i += 4;
        if (cp >= 0xD800 && cp <= 0xDBFF && i + 6 <= src.size() &&
            src[i] == '\\' && src[i + 1] == 'u') {
          unsigned lo = 0;
          if (!hex4(i + 2, &lo)) return std::string::npos;
          if (lo >= 0xDC00 && lo <= 0xDFFF) {
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            i += 6;
          }
        }
        put_utf8(cp);
        break;
      }
      default:
        return std::string::npos;
    }
  }
  return std::string::npos;  // unterminated
}

// Append the assistant text carried by one SSE payload to `out`.
//
// The OpenAI chat-completions dialect streams events shaped
// {"choices":[{"delta":{"content":"..."}}]}. The Python client accumulates
// ONLY delta.content — role deltas and finish_reason chunks contribute
// nothing — and multi-turn history splices that accumulated text back into
// the next turn's messages. Payloads that contain no "delta" key at all
// (non-chat servers) pass through raw; payloads WITH a delta but no string
// content yield nothing, matching the Python client.
inline void append_stream_text(const std::string& payload, std::string& out) {
  size_t d = payload.find("\"delta\"");
  if (d == std::string::npos) {
    out += payload;
    return;
  }
  size_t c = payload.find("\"content\"", d);
  if (c == std::string::npos) return;
  size_t i = c + 9;  // past "content"
  while (i < payload.size() && (payload[i] == ' ' || payload[i] == '\t')) i++;
  if (i >= payload.size() || payload[i] != ':') return;
  i++;
  while (i < payload.size() && (payload[i] == ' ' || payload[i] == '\t')) i++;
  if (i >= payload.size() || payload[i] != '"') return;  // null / non-string
  json_unescape_into(payload, i + 1, out);
}

// Scan decoded body bytes for SSE `data:` lines, stamping each with `ts` —
// the read time of the recv() that delivered these bytes, so the offset
// reflects arrival, not when a later pass got around to parsing.
inline void parse_body_sse(HttpStreamState& s, double ts) {
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
      s.sizes.push_back((int)d.size());  // wire payload size, pre-extraction
      append_stream_text(d, s.content);
    }
  }
}

// Feed freshly read bytes through llhttp (headers + chunked decode +
// completion) -> SSE / raw framing. Returns the number of NEW decoded body
// bytes made available by this call (adaptation vs the reference: lets the
// TTS_HTTP path stamp per-chunk arrivals + accumulate recv_bytes without
// keeping the audio body around). `ts` is the read-time stamp captured at
// recv(): every body byte delivered by this execute is attributed to it.
inline size_t http_feed(HttpStreamState& s, const char* buf, size_t n,
                        double ts) {
  if (!s.parser_init) {
    llhttp_init(&s.parser, HTTP_RESPONSE, llhttp_detail::settings());
    s.parser_init = true;
  }
  s.parser.data = &s;
  size_t before = s.inbuf.size();  // leftover partial SSE line, if any
  llhttp_errno_t err = llhttp_execute(&s.parser, buf, n);
  size_t new_bytes = s.inbuf.size() - before;
  if (err != HPE_OK) http_mark_malformed(s, err);
  if (s.sse) {
    parse_body_sse(s, ts);
  } else {
    s.content += s.inbuf;
    s.inbuf.clear();
  }
  if (s.body_done || s.malformed) s.done = true;
  return new_bytes;
}

// EOF from the server after headers arrived. We always send
// `Connection: close`, so EOF is the normal end for read-until-EOF bodies
// (no Content-Length, no chunked): llhttp_finish fires on_message_complete
// for those, and rejects EOF that truncates chunked / Content-Length framing.
// Returns false when the EOF was a framing violation (malformed + reason
// set); the native loop fails the request rather than completing silently.
inline bool http_eof(HttpStreamState& s) {
  if (!s.parser_init) return true;  // no bytes ever parsed; caller handles
  s.parser.data = &s;
  llhttp_errno_t err = llhttp_finish(&s.parser);
  if (err != HPE_OK) {
    http_mark_malformed(s, err);
    s.done = true;
    return false;
  }
  if (s.body_done) s.done = true;
  return true;
}

// JSON string escaping for content spliced into a request-body template
// (native multi-turn history injection). UTF-8 bytes pass through; quotes,
// backslashes and control characters are escaped per RFC 8259.
inline std::string json_escape(const std::string& s) {
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
// WebSocket client codec.
// ===========================================================================

// Encode one client->server frame, masked per RFC 6455 (opcode 0x1 = text,
// 0xA = pong). Masking must be present for client frames; the key value need
// not be random for protocol correctness.
inline std::string ws_encode_frame(unsigned char opcode,
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

// Fragmentation state for an in-progress fragmented server message.
struct WsFragState {
  int frag_opcode = 0;
  std::string frag_buf;
};

// Pull complete server frames out of inbuf. `on_data(std::string&&)` is
// invoked for each complete text/binary message (fragmentation reassembled).
// Ping frames queue a pong into `outbuf`; pongs are ignored. Sets got_close
// on a close frame (0x8). The caller stamps arrival BEFORE invoking this, so
// frame arrival never gets charged with parse cost.
template <typename OnData>
inline void ws_extract_frames(std::string& inbuf, std::string& outbuf,
                              WsFragState& frag, bool& got_close,
                              OnData&& on_data) {
  while (true) {
    if (inbuf.size() < 2) return;
    const unsigned char* p = (const unsigned char*)inbuf.data();
    unsigned char b0 = p[0], b1 = p[1];
    bool fin = b0 & 0x80;
    int opcode = b0 & 0x0F;
    bool masked = b1 & 0x80;  // server->client must be unmasked
    uint64_t len = b1 & 0x7F;
    size_t offset = 2;
    if (len == 126) {
      if (inbuf.size() < 4) return;
      len = ((uint64_t)p[2] << 8) | p[3];
      offset = 4;
    } else if (len == 127) {
      if (inbuf.size() < 10) return;
      len = 0;
      for (int i = 0; i < 8; i++) len = (len << 8) | p[2 + i];
      offset = 10;
    }
    size_t mask_len = masked ? 4 : 0;
    if (inbuf.size() < offset + mask_len + len) return;  // wait for full frame
    std::string payload = inbuf.substr(offset + mask_len, len);
    if (masked) {
      const unsigned char* mk = p + offset;
      for (size_t i = 0; i < payload.size(); i++) payload[i] ^= mk[i % 4];
    }
    inbuf.erase(0, offset + mask_len + len);
    if (opcode == 0x8) {  // close
      got_close = true;
      return;
    }
    if (opcode == 0x9) {  // ping -> queue pong with the same payload
      outbuf += ws_encode_frame(0xA, payload);
    } else if (opcode == 0x1 || opcode == 0x2) {  // text / binary data frame
      if (fin) {
        on_data(std::move(payload));
      } else {
        frag.frag_opcode = opcode;
        frag.frag_buf = std::move(payload);
      }
    } else if (opcode == 0x0 && frag.frag_opcode != 0) {  // continuation
      frag.frag_buf += payload;
      if (fin) {
        on_data(std::move(frag.frag_buf));
        frag.frag_buf.clear();
        frag.frag_opcode = 0;
      }
    }
    // pong (0xA): ignored.
  }
}

inline std::string ws_handshake_request(
    const std::string& host, int port, const std::string& path,
    const std::vector<std::pair<std::string, std::string>>& headers) {
  std::string req = "GET " + path + " HTTP/1.1\r\n";
  req += "Host: " + host + ":" + std::to_string(port) + "\r\n";
  req += "Upgrade: websocket\r\n";
  req += "Connection: Upgrade\r\n";
  req += "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
  req += "Sec-WebSocket-Version: 13\r\n";
  for (const auto& kv : headers) req += kv.first + ": " + kv.second + "\r\n";
  req += "\r\n";
  return req;
}

// Consume the handshake response once complete. Result: 0 = need more bytes,
// 1 = handshake ok (bytes after the header terminator remain in inbuf as the
// first frames), -1 = failed (error set). Sec-WebSocket-Accept is not
// cryptographically verified — status + Upgrade is sufficient for the
// controlled endpoints this transport targets.
inline int ws_check_handshake(std::string& inbuf, std::string* error) {
  size_t term = inbuf.find("\r\n\r\n");
  if (term == std::string::npos) return 0;  // need more bytes
  int status = 0;
  size_t sp = inbuf.find(' ');
  if (sp != std::string::npos && sp < term)
    status = atoi(inbuf.c_str() + sp + 1);
  if (status != 101) {
    *error = "ws handshake failed (status " + std::to_string(status) + ")";
    return -1;
  }
  std::string head(inbuf, 0, term);
  std::transform(head.begin(), head.end(), head.begin(),
                 [](unsigned char ch) { return (char)std::tolower(ch); });
  if (head.find("upgrade: websocket") == std::string::npos) {
    *error = "ws handshake failed (no Upgrade: websocket header)";
    return -1;
  }
  inbuf.erase(0, term + 4);
  return 1;
}

}  // namespace veeksha_native_transport
