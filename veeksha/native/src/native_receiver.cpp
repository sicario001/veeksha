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

PYBIND11_MODULE(veeksha_native, m, py::mod_gil_not_used()) {
  m.doc() = "Native (C++) streaming receive path for Veeksha drift benchmarking.";
  m.def("receive", &py_receive, py::arg("host"), py::arg("port"),
        py::arg("concurrency"), py::arg("num_chunks"), py::arg("total_requests"),
        py::arg("timeout_s") = 120.0,
        "Run a single-thread native receive loop; return per-request chunk "
        "arrival offsets (ms from send).");
}
