// veeksha_native — the native (C++) benchmark loop for Veeksha.
//
// Implements the NativeBenchmarkLoop of docs/design/native_loop_prototype.md
// (§§0-6) and docs/design/preflight_and_native_loop.md §3: a drop-in
// _run_main_loop with the same worker roles, queue topology, traffic
// semantics and five lifecycle timestamps as the Python pipeline, minus
// Python anywhere on the hot path.
//
// Boundary rule (prototype §0): native threads never call Python. The only
// crossings are feed_sessions/feed_intervals/register_blob (Python feeder
// thread pushes plain data in), drain_events (Python drainer pulls plain data
// out), and counters/in_flight snapshots (atomics). Everything the loop needs
// arrives as plain-data structs compiled by Python before/independent of the
// timing window.
//
// Thread topology (mirrors the Python workers 1:1):
//   scheduler ×1        — intake pump + traffic-kind admission (PrefetchWorker
//                         + scheduler admission half)
//   dispatch ×N         — pop due requests off the ready deadline-heap, stamp
//                         scheduler_ready/dispatched, emit DISPATCHED,
//                         power-of-two push to a client reactor shard
//   client reactors ×N  — reactor (kqueue/epoll/poll) loops running the four
//                         transport state machines with read-time stamps and
//                         absolute-deadline paced sends
//   completion-ack ×N   — stamp result_processed, notify the scheduler
//                         (children release / refill / cancel-on-failure)
//                         IMMEDIATELY, then append the COMPLETED event
//
// Declared py::mod_gil_not_used() — free-threaded CPython safe; the module
// never re-enables the GIL.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "transport.h"

namespace py = pybind11;
using namespace veeksha_native_transport;

// ===========================================================================
// §1 input structs (converted once at loop start, or streamed as plain data)
// ===========================================================================

struct NativeRuntimeConfig {
  int max_sessions = -1;             // -1 = unlimited (enforced by the feeder)
  double benchmark_timeout_s = 0.0;  // monitor-owned; whole-run stop arrives
  double post_timeout_grace_s = 0.0;  //   via request_stop() only
  int num_dispatcher_threads = 2;
  int num_completion_threads = 8;
  int num_client_threads = 3;  // resolved in Python: max(3, ceil(target/8))
};

enum class TrafficKind : int { RATE = 0, CONCURRENT = 1, SEQUENTIAL_LAUNCH = 2 };
enum class TicketOrdering : int { DISPATCH = 0, PREFILL = 1, REQUEST = 2 };

struct TrafficPlanConfig {
  TrafficKind kind = TrafficKind::CONCURRENT;
  // CONCURRENT: native reproduces int(target * t/rampup) with a
  // rampup-complete latch (concurrent.py:42-48).
  int target_concurrent_sessions = 0;
  double rampup_seconds = 0.0;
  // SEQUENTIAL_LAUNCH:
  TicketOrdering ordering = TicketOrdering::DISPATCH;
  bool cancel_session_on_failure = true;
  // RATE: interarrival draws are NOT generated natively; Python streams the
  // exact seeded draws via feed_intervals(). Bit-identical schedules across
  // loop implementations by construction.
};

struct EndpointConfig {
  std::string host;
  int port = 0;
  std::string base_path;  // api_base path prefix ("" if none)
  std::vector<std::pair<std::string, std::string>> headers;  // incl. auth
  double request_timeout_s = 0.0;  // per-request budget; <=0 = unlimited
};

enum class TransportKind : int {
  TEXT_SSE = 0,
  TTS_HTTP = 1,
  TTS_REALTIME_WS = 2,
  STT_WS = 3
};

// Large payloads (STT base64 audio frames) are NOT inlined per session: many
// sessions share one clip. Python registers a blob once and frames reference
// (blob_id, offset, len) slices.
struct BlobRef {
  int blob_id = -1;
  long long offset = 0;
  long long len = 0;
};

struct WsFrame {
  std::string payload;   // inline payload (small frames), OR:
  BlobRef blob;          // slice of a registered blob (audio appends)
  double send_offset_ms = -1.0;  // absolute offset from pacing anchor;
                                 // <0 = "asap, in order"
};

struct TransportPlan {
  TransportKind kind = TransportKind::TEXT_SSE;

  // -- TEXT_SSE / TTS_HTTP (templated HTTP/1.1 request) --
  // wire = header_prefix + itoa(body_len) + header_suffix + body
  // body = seg[0] + escape(extract(history_refs[0])) + seg[1] + ...
  std::string header_prefix;  // through "Content-Length: "
  std::string header_suffix;  // remaining headers + CRLFCRLF
  std::vector<std::string> body_segments;
  std::vector<int> history_refs;  // node_ids whose extracted output fills
                                  // holes (empty for single-turn / pre-baked)

  // -- TTS_REALTIME_WS / STT_WS --
  std::string ws_path;
  std::vector<WsFrame> setup_frames;   // ordered, asap
  std::vector<WsFrame> paced_frames;   // on schedule
  std::vector<WsFrame> finish_frames;  // after paced
  std::vector<std::string> done_markers;  // terminal event substrings
  // pacing anchor: TTS_REALTIME_WS -> offsets from post-handshake;
  // STT_WS -> offsets from first paced send.
};

struct RequestPlan {
  long long request_id = 0;
  int node_id = 0;
  double wait_after_ready_s = 0.0;
  std::vector<std::pair<int, bool>> parents;  // (node_id, is_history_parent)
  TransportPlan transport;
};

struct SessionPlan {
  long long session_id = 0;
  std::vector<RequestPlan> requests;  // topological order (Python validates)
  int dispatch_ticket_base = -1;      // SEQUENTIAL_LAUNCH: first root ticket
};

// ===========================================================================
// §2 output structs (drained by Python)
// ===========================================================================

struct ChunkStamp {
  double offset_ms = 0.0;  // read-time, from request send
  int size = 0;
};

struct RawRequestResult {
  long long request_id = 0;
  long long session_id = 0;
  int session_total_requests = 0;
  int status = 0;     // HTTP status / WS close mapping
  std::string error;  // "" = success

  // The five lifecycle stamps, ms from the native-loop epoch (§5), SAME anchor
  // points as the Python loop:
  double scheduler_ready_ms = 0.0;
  double scheduler_dispatched_ms = 0.0;
  double client_picked_up_ms = 0.0;
  double client_completed_ms = 0.0;
  double result_processed_ms = 0.0;

  std::vector<ChunkStamp> recv_stamps;
  std::vector<double> send_offsets_ms;  // per paced send, stamped BEFORE send
  std::string content;   // extracted assistant text / final transcript
  long long recv_bytes = 0;
  std::vector<std::pair<std::string, double>> event_offsets_ms;
};

enum class EventKind : int { DISPATCHED = 0, COMPLETED = 1 };

struct NativeLoopEvent {
  EventKind kind = EventKind::DISPATCHED;
  long long request_id = 0;
  long long session_id = 0;
  int session_total_requests = 0;
  double ready_ms = 0.0, dispatched_ms = 0.0;  // DISPATCHED payload
  RawRequestResult result;  // COMPLETED payload (empty for DISPATCHED)
};

struct NativeLoopCounters {
  long long sessions_completed = 0;
  long long sessions_errored = 0;
  long long sessions_seen = 0;
  long long requests_dispatched = 0;
  long long requests_completed = 0;
  long long in_flight = 0;
  bool intake_exhausted = false;
  bool idle = false;
};

// ===========================================================================
// NativeLoop internals
// ===========================================================================

namespace {

constexpr double kFar = 1e18;
constexpr size_t kIntakeCapacity = 1024;  // bounded intake ring (sessions)

bool is_http_kind(TransportKind k) {
  return k == TransportKind::TEXT_SSE || k == TransportKind::TTS_HTTP;
}

// Extract the value of a top-level-ish "type":"..." key (realtime protocol
// event name). Types never contain escapes in practice; returns "" if absent.
std::string extract_type_string(const std::string& payload) {
  size_t t = payload.find("\"type\"");
  if (t == std::string::npos) return "";
  size_t i = t + 6;
  while (i < payload.size() && (payload[i] == ' ' || payload[i] == '\t')) i++;
  if (i >= payload.size() || payload[i] != ':') return "";
  i++;
  while (i < payload.size() && (payload[i] == ' ' || payload[i] == '\t')) i++;
  if (i >= payload.size() || payload[i] != '"') return "";
  size_t end = payload.find('"', i + 1);
  if (end == std::string::npos) return "";
  return payload.substr(i + 1, end - i - 1);
}

// Extract a top-level "delta":"..." STRING value (STT transcript deltas) and
// append the unescaped text to out. A "delta" that is not a string (the chat
// object shape) contributes nothing here — that shape belongs to TEXT_SSE.
void extract_delta_text(const std::string& payload, std::string& out) {
  size_t d = payload.find("\"delta\"");
  if (d == std::string::npos) return;
  size_t i = d + 7;
  while (i < payload.size() && (payload[i] == ' ' || payload[i] == '\t')) i++;
  if (i >= payload.size() || payload[i] != ':') return;
  i++;
  while (i < payload.size() && (payload[i] == ' ' || payload[i] == '\t')) i++;
  if (i >= payload.size() || payload[i] != '"') return;
  json_unescape_into(payload, i + 1, out);
}

}  // namespace

class NativeBenchmarkLoop {
 public:
  NativeBenchmarkLoop(NativeRuntimeConfig rt, TrafficPlanConfig tp,
                      EndpointConfig ep, double py_monotonic_anchor_s)
      : rt_(rt), tp_(tp), ep_(ep), py_anchor_s_(py_monotonic_anchor_s) {
    if (rt_.num_dispatcher_threads < 1) rt_.num_dispatcher_threads = 1;
    if (rt_.num_completion_threads < 1) rt_.num_completion_threads = 1;
    if (rt_.num_client_threads < 1) rt_.num_client_threads = 1;
    t0_ms_ = now_ms();  // clock handshake: pairs with py_monotonic_anchor_s
    gate_enabled_ = (tp_.kind == TrafficKind::SEQUENTIAL_LAUNCH);
    addr_ok_ = resolve_addr(ep_.host, ep_.port, &addr_);
    if (!addr_ok_) resolve_error_ = "resolve failed: " + ep_.host;
    raise_nofile(4096);

    shards_.reserve(rt_.num_client_threads);
    for (int i = 0; i < rt_.num_client_threads; i++) {
      auto sh = std::make_unique<Shard>();
      int fds[2];
      if (pipe(fds) == 0) {
        fcntl(fds[0], F_SETFL, fcntl(fds[0], F_GETFL, 0) | O_NONBLOCK);
        fcntl(fds[1], F_SETFL, fcntl(fds[1], F_GETFL, 0) | O_NONBLOCK);
        sh->wake_r = fds[0];
        sh->wake_w = fds[1];
      }
      shards_.push_back(std::move(sh));
    }

    live_dispatchers_.store(rt_.num_dispatcher_threads);
    live_reactors_.store(rt_.num_client_threads);
    live_acks_.store(rt_.num_completion_threads);

    threads_.emplace_back([this] { scheduler_loop(); });
    for (int i = 0; i < rt_.num_dispatcher_threads; i++)
      threads_.emplace_back([this, i] { dispatcher_loop(i); });
    for (int i = 0; i < rt_.num_client_threads; i++)
      threads_.emplace_back([this, i] { reactor_loop(i); });
    for (int i = 0; i < rt_.num_completion_threads; i++)
      threads_.emplace_back([this] { ack_loop(); });
  }

  ~NativeBenchmarkLoop() {
    request_stop(0.0);
    {
      std::unique_lock<std::mutex> lk(done_mu_);
      done_cv_.wait_for(lk, std::chrono::seconds(10),
                        [&] { return loop_done_.load(); });
    }
    join_threads();
    for (auto& sh : shards_) {
      if (sh->wake_r >= 0) close(sh->wake_r);
      if (sh->wake_w >= 0) close(sh->wake_w);
    }
  }

  double loop_epoch_py_monotonic_s() const { return py_anchor_s_; }

  // ---- intake (Python feeder thread) ----

  long long register_blob(py::bytes data) {
    std::string s = data;  // copied once
    std::unique_lock<std::shared_mutex> lk(blobs_mu_);
    blobs_.push_back(std::move(s));
    return (long long)blobs_.size() - 1;
  }

  // Validation runs on the feeder thread (Python), errors surface as Python
  // exceptions before any plan reaches native threads.
  void validate_plan(const SessionPlan& p) const {
    for (const auto& r : p.requests) {
      const TransportPlan& t = r.transport;
      if (is_http_kind(t.kind)) {
        if (!t.history_refs.empty() &&
            t.body_segments.size() != t.history_refs.size() + 1) {
          throw std::invalid_argument(
              "session " + std::to_string(p.session_id) + " node " +
              std::to_string(r.node_id) +
              ": body_segments must be history_refs+1 segments");
        }
      } else {
        auto check_frames = [&](const std::vector<WsFrame>& fs) {
          std::shared_lock<std::shared_mutex> lk(blobs_mu_);
          for (const auto& f : fs) {
            if (f.blob.blob_id >= 0) {
              if ((size_t)f.blob.blob_id >= blobs_.size())
                throw std::invalid_argument("unknown blob_id " +
                                            std::to_string(f.blob.blob_id));
              const std::string& b = blobs_[f.blob.blob_id];
              if (f.blob.offset < 0 || f.blob.len < 0 ||
                  (size_t)(f.blob.offset + f.blob.len) > b.size())
                throw std::invalid_argument("blob slice out of range");
            }
          }
        };
        check_frames(t.setup_frames);
        check_frames(t.paced_frames);
        check_frames(t.finish_frames);
      }
    }
  }

  long long feed_sessions(std::vector<SessionPlan> plans) {
    for (const auto& p : plans) validate_plan(p);
    long long accepted = 0;
    for (auto& p : plans) {
      std::unique_lock<std::mutex> lk(intake_mu_);
      intake_space_cv_.wait(lk, [&] {
        return intake_.size() < kIntakeCapacity || stopping_.load() ||
               intake_closed_;
      });
      if (stopping_.load() || intake_closed_) break;
      intake_.push_back(std::move(p));
      accepted++;
      intake_cv_.notify_all();
    }
    return accepted;
  }

  void feed_intervals(std::vector<double> intervals_s) {
    std::lock_guard<std::mutex> lk(sched_mu_);
    for (double v : intervals_s) intervals_.push_back(v);
    sched_cv_.notify_all();
  }

  void close_intake() {
    {
      std::lock_guard<std::mutex> lk(intake_mu_);
      intake_closed_ = true;
      intake_cv_.notify_all();
      intake_space_cv_.notify_all();
    }
    intake_closed_flag_.store(true);
    std::lock_guard<std::mutex> lk(sched_mu_);
    sched_cv_.notify_all();
  }

  // ---- drain (Python drainer thread) ----

  std::vector<NativeLoopEvent> drain_events(int max_items, double timeout_s) {
    std::vector<NativeLoopEvent> out;
    std::unique_lock<std::mutex> lk(ev_mu_);
    if (events_.empty()) {
      ev_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [&] {
        return !events_.empty() || loop_done_.load();
      });
    }
    while (!events_.empty() && (int)out.size() < max_items) {
      out.push_back(std::move(events_.front()));
      events_.pop_front();
    }
    return out;
  }

  // ---- monitor / control (Python monitor thread) ----

  NativeLoopCounters counters() {
    NativeLoopCounters c;
    c.sessions_completed = c_sessions_completed_.load();
    c.sessions_errored = c_sessions_errored_.load();
    c.sessions_seen = c_sessions_seen_.load();
    c.requests_dispatched = c_dispatched_.load();
    c.requests_completed = c_completed_.load();
    c.in_flight = c_in_flight_.load();
    c.intake_exhausted = intake_closed_flag_.load();
    bool sched_idle;
    {
      std::lock_guard<std::mutex> lk(sched_mu_);
      sched_idle = intake_drained_ && pending_sessions_.empty() &&
                   sessions_.empty() && ready_.empty();
    }
    c.idle = sched_idle && c.in_flight == 0;
    return c;
  }

  std::vector<long long> in_flight_request_ids() {
    std::lock_guard<std::mutex> lk(track_mu_);
    return std::vector<long long>(in_flight_ids_.begin(), in_flight_ids_.end());
  }

  std::vector<long long> dispatched_request_ids() {
    std::lock_guard<std::mutex> lk(track_mu_);
    return std::vector<long long>(dispatched_ids_.begin(),
                                  dispatched_ids_.end());
  }

  void request_stop(double grace_s) {
    bool was = stopping_.exchange(true);
    {
      std::lock_guard<std::mutex> lk(sched_mu_);
      stop_dispatch_ = true;
      // stop dispatching new work: drop everything not yet dispatched
      // (pending first so session-finish refill cannot re-activate)
      pending_sessions_.clear();
      while (!ready_.empty()) drop_ready_locked();
      sched_cv_.notify_all();
    }
    {
      std::lock_guard<std::mutex> lk(intake_mu_);
      intake_.clear();
      intake_cv_.notify_all();
      intake_space_cv_.notify_all();
    }
    wake_all_shards();
    if (was) return;  // grace already armed by the first call
    if (grace_s <= 0.0) {
      fail_all_.store(true);
      wake_all_shards();
    } else {
      grace_thread_ = std::thread([this, grace_s] {
        std::unique_lock<std::mutex> lk(done_mu_);
        done_cv_.wait_for(lk, std::chrono::duration<double>(grace_s),
                          [&] { return loop_done_.load(); });
        lk.unlock();
        fail_all_.store(true);
        wake_all_shards();
      });
    }
  }

  bool join(double timeout_s) {
    {
      std::unique_lock<std::mutex> lk(done_mu_);
      if (!done_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s),
                             [&] { return loop_done_.load(); }))
        return false;
    }
    join_threads();
    return true;
  }

 private:
  // ------------------------------------------------------------------
  // scheduler state (one mutex — native-fast, unlike the Python Condition
  // this lock guards microseconds of pointer work, no Python allocation)
  // ------------------------------------------------------------------

  struct SessState {
    SessionPlan plan;
    int total = 0;
    std::unordered_map<int, int> node_index;          // node_id -> request idx
    std::unordered_map<int, double> completions;      // node_id -> ms
    std::unordered_map<int, std::string> contents;    // extracted output; all
                                                      // keys pre-created at
                                                      // activation (value-only
                                                      // writes afterwards)
    std::unordered_map<int, std::vector<int>> children;
    std::unordered_set<int> pending;  // not yet released
    int queued = 0;                   // released, not yet completed
    bool canceled = false;
    bool errored = false;
  };

  struct ReadyItem {
    double ready_ms = 0.0;
    long long seq = 0;
    long long session_id = 0;
    int node_id = 0;
    long long ticket = -1;
  };
  struct ReadyCmp {
    bool operator()(const ReadyItem& a, const ReadyItem& b) const {
      if (a.ready_ms != b.ready_ms) return a.ready_ms > b.ready_ms;
      return a.seq > b.seq;
    }
  };

  struct WorkItem {
    long long request_id = 0;
    long long session_id = 0;
    int total = 0;
    long long ticket = -1;
    double ready_ms = 0.0, dispatched_ms = 0.0;
    const RequestPlan* plan = nullptr;  // owned by SessState (stable)
    SessState* sess = nullptr;          // alive while this request is queued
  };

  struct Shard {
    std::mutex mu;
    std::deque<WorkItem> q;
    std::atomic<int> qlen{0};
    int wake_r = -1, wake_w = -1;
  };

  double rel_now() const { return now_ms() - t0_ms_; }

  void wake_shard(Shard& sh) {
    if (sh.wake_w >= 0) {
      char b = 1;
      ssize_t r = write(sh.wake_w, &b, 1);
      (void)r;  // EAGAIN = wake already pending
    }
  }
  void wake_all_shards() {
    for (auto& sh : shards_) wake_shard(*sh);
  }

  // ---- events ----

  void emit_event(NativeLoopEvent&& ev) {
    std::lock_guard<std::mutex> lk(ev_mu_);
    events_.push_back(std::move(ev));
    ev_cv_.notify_all();
  }

  // ---- ticket gate (DispatchTracker analogue: atomic counter + wakeups) ----

  void gate_advance(long long ticket) {
    long long want = ticket + 1;
    long long cur = gate_counter_.load();
    while (cur < want && !gate_counter_.compare_exchange_weak(cur, want)) {
    }
    wake_all_shards();  // parked items re-check on wake
  }
  bool gate_open(long long ticket) const {
    return ticket < 0 || !gate_enabled_ || ticket <= gate_counter_.load();
  }

  // ---- scheduler core (sched_mu_ held) ----

  int concurrency_cap_locked() {
    if (tp_.kind != TrafficKind::CONCURRENT) return 1 << 30;
    if (rampup_complete_) return tp_.target_concurrent_sessions;
    double t = rel_now() / 1000.0;
    if (tp_.rampup_seconds <= 0.0 || t >= tp_.rampup_seconds) {
      rampup_complete_ = true;
      return tp_.target_concurrent_sessions;
    }
    return (int)(tp_.target_concurrent_sessions * (t / tp_.rampup_seconds));
  }

  void push_ready_locked(double ready_ms, long long sid, int node,
                         long long ticket) {
    ReadyItem it;
    it.ready_ms = ready_ms;
    it.seq = ready_seq_++;
    it.session_id = sid;
    it.node_id = node;
    it.ticket = ticket;
    ready_.push(it);
    sched_cv_.notify_all();
  }

  // request_stop: pop a ready item and account for it so the session state
  // stays consistent (it was released but will never dispatch).
  void drop_ready_locked() {
    ReadyItem it = ready_.top();
    ready_.pop();
    auto sit = sessions_.find(it.session_id);
    if (sit == sessions_.end()) return;
    SessState& st = *sit->second;
    st.queued--;
    st.pending.clear();  // stopping: nothing further will be released
    if (st.queued == 0) finish_session_locked(sit);
  }

  void finish_session_locked(
      std::unordered_map<long long, std::unique_ptr<SessState>>::iterator it) {
    bool errored = it->second->errored;
    sessions_.erase(it);
    if (errored)
      c_sessions_errored_.fetch_add(1);
    else
      c_sessions_completed_.fetch_add(1);
    try_activate_locked();
    sched_cv_.notify_all();
  }

  void activate_locked(SessionPlan&& plan, double start_ms) {
    auto stp = std::make_unique<SessState>();
    SessState& st = *stp;
    st.plan = std::move(plan);
    st.total = (int)st.plan.requests.size();
    for (int i = 0; i < st.total; i++) {
      const RequestPlan& r = st.plan.requests[i];
      st.node_index[r.node_id] = i;
      st.contents[r.node_id];  // pre-create: later writes are value-only
      st.pending.insert(r.node_id);
      for (const auto& pe : r.parents) st.children[pe.first].push_back(r.node_id);
    }
    long long sid = st.plan.session_id;
    if (st.total == 0) {
      c_sessions_completed_.fetch_add(1);
      return;
    }
    int root_ordinal = 0;
    long long base = st.plan.dispatch_ticket_base;
    for (int i = 0; i < st.total; i++) {
      const RequestPlan& r = st.plan.requests[i];
      if (!r.parents.empty()) continue;
      long long ticket = -1;
      if (tp_.kind == TrafficKind::SEQUENTIAL_LAUNCH) {
        if (base >= 0) {
          ticket = base + root_ordinal;
          if (next_ticket_ <= ticket) next_ticket_ = ticket + 1;
        } else {
          ticket = next_ticket_++;
        }
        root_ordinal++;
      }
      st.pending.erase(r.node_id);
      st.queued++;
      push_ready_locked(start_ms + r.wait_after_ready_s * 1000.0, sid,
                        r.node_id, ticket);
    }
    sessions_[sid] = std::move(stp);
  }

  void try_activate_locked() {
    if (tp_.kind != TrafficKind::CONCURRENT) return;
    while (!pending_sessions_.empty() &&
           (int)sessions_.size() < concurrency_cap_locked()) {
      SessionPlan p = std::move(pending_sessions_.front());
      pending_sessions_.pop_front();
      activate_locked(std::move(p), rel_now());
    }
  }

  void admit(SessionPlan&& plan) {
    std::unique_lock<std::mutex> lk(sched_mu_);
    c_sessions_seen_.fetch_add(1);
    switch (tp_.kind) {
      case TrafficKind::RATE: {
        // Consume the deferred interval (spacing to the PREVIOUS session)
        // before assigning this session's start; block until fed. If intake
        // is closed with no interval left, degrade to a 0 gap (the feeder is
        // done; nothing more will arrive).
        while (need_interval_ && intervals_.empty() && !stopping_.load() &&
               !intake_closed_flag_.load()) {
          sched_cv_.wait_for(lk, std::chrono::milliseconds(10));
        }
        if (need_interval_) {
          if (!intervals_.empty()) {
            next_start_ms_ += intervals_.front() * 1000.0;
            intervals_.pop_front();
          }
          need_interval_ = false;
        }
        double start = next_start_ms_;
        if (!intervals_.empty()) {
          next_start_ms_ = start + intervals_.front() * 1000.0;
          intervals_.pop_front();
        } else {
          need_interval_ = true;
        }
        activate_locked(std::move(plan), start);
        break;
      }
      case TrafficKind::CONCURRENT:
        pending_sessions_.push_back(std::move(plan));
        try_activate_locked();
        break;
      case TrafficKind::SEQUENTIAL_LAUNCH:
        activate_locked(std::move(plan), rel_now());
        break;
    }
  }

  // notify_completion — same semantics/order as the Python schedulers:
  // record completion, record extracted content (the history splice source),
  // cancel-on-failure clears un-released nodes (already-queued ones drain
  // normally, matching concurrent.py:182-208), release ready children, erase
  // + refill when the session empties.
  void scheduler_notify_completion(long long sid, int node, double completed_ms,
                                   bool success, const std::string& content) {
    std::unique_lock<std::mutex> lk(sched_mu_);
    auto it = sessions_.find(sid);
    if (it == sessions_.end()) return;
    SessState& st = *it->second;
    st.completions[node] = completed_ms;
    st.queued--;
    if (success) {
      auto cit = st.contents.find(node);
      if (cit != st.contents.end()) cit->second = content;
    } else {
      st.errored = true;
    }

    if (!success && tp_.cancel_session_on_failure) {
      st.canceled = true;
      st.pending.clear();
      if (st.queued == 0) finish_session_locked(it);
      return;
    }

    if (stop_dispatch_) {
      // stopping: nothing further will be released
      st.pending.clear();
    } else {
      auto chit = st.children.find(node);
      if (chit != st.children.end()) {
        for (int child : chit->second) {
          if (!st.pending.count(child)) continue;
          const RequestPlan& r = st.plan.requests[st.node_index[child]];
          double parent_max = -1.0;
          bool all_done = true;
          for (const auto& pe : r.parents) {
            auto cit = st.completions.find(pe.first);
            if (cit == st.completions.end()) {
              all_done = false;
              break;
            }
            parent_max = std::max(parent_max, cit->second);
          }
          if (!all_done) continue;
          st.pending.erase(child);
          st.queued++;
          push_ready_locked(parent_max + r.wait_after_ready_s * 1000.0, sid,
                            child, /*ticket=*/-1);
        }
      }
    }

    if (st.pending.empty() && st.queued == 0) finish_session_locked(it);
  }

  bool sched_finished_locked() {
    return intake_drained_ && pending_sessions_.empty() && sessions_.empty() &&
           ready_.empty();
  }

  // ------------------------------------------------------------------
  // scheduler thread: intake pump + rampup activation timing
  // ------------------------------------------------------------------

  void scheduler_loop() {
    set_thread_qos();
    while (!stopping_.load()) {
      // short wait when a rampup is pending so activation stays timely
      bool rampup_pending;
      {
        std::lock_guard<std::mutex> slk(sched_mu_);
        rampup_pending = !pending_sessions_.empty() && !rampup_complete_;
      }
      std::deque<SessionPlan> batch;
      bool drained;
      {
        std::unique_lock<std::mutex> lk(intake_mu_);
        if (intake_.empty() && !intake_closed_) {
          intake_cv_.wait_for(
              lk, std::chrono::milliseconds(rampup_pending ? 2 : 20));
        }
        while (!intake_.empty()) {
          batch.push_back(std::move(intake_.front()));
          intake_.pop_front();
        }
        if (!batch.empty()) intake_space_cv_.notify_all();
        drained = intake_closed_ && intake_.empty();
      }
      for (auto& p : batch) {
        if (stopping_.load()) break;
        admit(std::move(p));
      }
      {
        std::lock_guard<std::mutex> lk(sched_mu_);
        try_activate_locked();
        if (drained && !stopping_.load()) {
          intake_drained_ = true;
          sched_cv_.notify_all();
          if (pending_sessions_.empty()) return;  // nothing left to admit
        }
      }
    }
    // stopping: mark drained so dispatchers/finish checks converge
    std::lock_guard<std::mutex> lk(sched_mu_);
    intake_drained_ = true;
    sched_cv_.notify_all();
  }

  // ------------------------------------------------------------------
  // dispatcher threads
  // ------------------------------------------------------------------

  void dispatcher_loop(int idx) {
    set_thread_qos();
    std::mt19937 rng((unsigned)(0x9e3779b9u * (idx + 1)));
    std::unique_lock<std::mutex> lk(sched_mu_);
    while (true) {
      if (stop_dispatch_) break;
      if (sched_finished_locked()) break;
      double now = rel_now();
      if (!ready_.empty() && ready_.top().ready_ms <= now) {
        ReadyItem it = ready_.top();
        ready_.pop();
        auto sit = sessions_.find(it.session_id);
        if (sit == sessions_.end()) continue;  // stale (should not happen)
        SessState* st = sit->second.get();
        int ridx = st->node_index[it.node_id];
        WorkItem w;
        w.request_id = st->plan.requests[ridx].request_id;
        w.session_id = it.session_id;
        w.total = st->total;
        w.ticket = it.ticket;
        w.plan = &st->plan.requests[ridx];
        w.sess = st;
        lk.unlock();

        // stamps at the same points/order as dispatch.py:77-78
        w.ready_ms = rel_now();
        w.dispatched_ms = rel_now();
        {
          std::lock_guard<std::mutex> tlk(track_mu_);
          dispatched_ids_.insert(w.request_id);
          in_flight_ids_.insert(w.request_id);
        }
        c_dispatched_.fetch_add(1);
        c_in_flight_.fetch_add(1);

        NativeLoopEvent ev;
        ev.kind = EventKind::DISPATCHED;
        ev.request_id = w.request_id;
        ev.session_id = w.session_id;
        ev.session_total_requests = w.total;
        ev.ready_ms = w.ready_ms;
        ev.dispatched_ms = w.dispatched_ms;
        emit_event(std::move(ev));

        // power-of-two choice among reactor shard queues
        int n = (int)shards_.size();
        int pick = 0;
        if (n > 1) {
          int a = (int)(rng() % n);
          int b = (int)(rng() % (n - 1));
          if (b >= a) b++;
          pick = shards_[a]->qlen.load() <= shards_[b]->qlen.load() ? a : b;
        }
        Shard& sh = *shards_[pick];
        {
          std::lock_guard<std::mutex> qlk(sh.mu);
          sh.q.push_back(w);
          sh.qlen.fetch_add(1);
        }
        wake_shard(sh);

        lk.lock();
        continue;
      }
      double wait_ms = 10.0;
      if (!ready_.empty())
        wait_ms = std::min(wait_ms, std::max(0.1, ready_.top().ready_ms - now));
      sched_cv_.wait_for(lk, std::chrono::duration<double>(wait_ms / 1000.0));
    }
    lk.unlock();
    if (live_dispatchers_.fetch_sub(1) == 1) {
      dispatchers_done_.store(true);
      wake_all_shards();
    }
  }

  // ------------------------------------------------------------------
  // client reactor threads: the four transport state machines
  // ------------------------------------------------------------------

  struct Conn {
    WorkItem w;
    TransportKind kind = TransportKind::TEXT_SSE;
    bool connected = false;
    bool handshaken = false;  // WS
    bool finish_sent = false;
    bool dispatched_fired = false;  // HTTP 200 / handshake-ok
    bool sent_fired = false;        // first content chunk
    bool status_checked = false;
    size_t next_paced = 0;
    double t_start_abs = 0.0;        // pre-connect anchor (recv stamp base)
    double pace_anchor_abs = -1.0;   // TTS realtime: post-handshake
    double audio_anchor_abs = -1.0;  // STT: first paced send
    double deadline_abs = kFar;      // request timeout budget
    double next_wake_armed = kFar;   // earliest pending timer for this conn
    HttpStreamState h;  // HTTP kinds
    std::string inbuf, outbuf;
    WsFragState frag;
    bool ws_done = false;
    RawRequestResult res;
  };

  struct CompItem {
    int node_id = -1;
    RawRequestResult res;
  };

  std::string blob_slice(const BlobRef& b) {
    std::shared_lock<std::shared_mutex> lk(blobs_mu_);
    return blobs_[b.blob_id].substr((size_t)b.offset, (size_t)b.len);
  }

  std::string frame_payload(const WsFrame& f) {
    if (f.blob.blob_id >= 0) return blob_slice(f.blob);
    return f.payload;
  }

  // body = seg[0] + escape(content[ref[0]]) + seg[1] + ... — the chains
  // build_wire approach; contents entries are pre-created at activation so
  // the lock-free read here is a value read ordered by the dispatch chain.
  std::string build_http_wire(const WorkItem& w) {
    const TransportPlan& t = w.plan->transport;
    std::string body = t.body_segments.empty() ? "" : t.body_segments[0];
    for (size_t hi = 0; hi < t.history_refs.size(); hi++) {
      const auto cit = w.sess->contents.find(t.history_refs[hi]);
      if (cit != w.sess->contents.end()) body += json_escape(cit->second);
      if (hi + 1 < t.body_segments.size()) body += t.body_segments[hi + 1];
    }
    return t.header_prefix + std::to_string(body.size()) + t.header_suffix +
           body;
  }

  class ReactorWorker {
   public:
    ReactorWorker(NativeBenchmarkLoop& eng, int idx)
        : eng_(eng), sh_(*eng.shards_[idx]) {}

    void run() {
      set_thread_qos();
      if (sh_.wake_r >= 0) reactor_.set_interest(sh_.wake_r, true, false);
      std::vector<ReactorEvent> evs;
      std::vector<int> due;
      while (true) {
        if (eng_.fail_all_.load() && !failed_all_) fail_everything();
        intake_pump();
        if (eng_.dispatchers_done_.load() && conns_.empty() && gated_.empty() &&
            sh_.qlen.load() == 0)
          break;
        double now = now_ms();
        double until = timers_.empty() ? 50.0 : timers_.next_deadline() - now;
        int timeout_ms = (int)std::max(0.0, std::min(50.0, until));
        reactor_.wait(timeout_ms, evs);
        for (auto& ev : evs) {
          if (ev.fd == sh_.wake_r) {
            char buf[256];
            while (read(sh_.wake_r, buf, sizeof(buf)) > 0) {
            }
            continue;
          }
          handle_event(ev);
        }
        timers_.pop_due(now_ms(), due);
        for (int fd : due) handle_timers(fd);
      }
    }

   private:
    NativeBenchmarkLoop& eng_;
    Shard& sh_;
    Reactor reactor_;
    TimerHeap timers_;
    std::unordered_map<int, Conn> conns_;
    std::deque<WorkItem> gated_;
    bool failed_all_ = false;

    // ---------------- lifecycle ----------------

    void init_result(RawRequestResult& r, const WorkItem& w) {
      r.request_id = w.request_id;
      r.session_id = w.session_id;
      r.session_total_requests = w.total;
      r.scheduler_ready_ms = w.ready_ms;
      r.scheduler_dispatched_ms = w.dispatched_ms;
    }

    // Arm a wakeup for this connection. Multiple heap entries are allowed
    // (lazy invalidation); the strictly-earlier check bounds duplication.
    void arm(int fd, Conn& c, double t) {
      if (t >= kFar) return;
      if (t < c.next_wake_armed - 1e-9) {
        timers_.push(t, fd);
        c.next_wake_armed = t;
      }
    }

    // Fail an item that never got a connection (gate parked / connect fail /
    // stop). Errors always carry reason strings.
    void fail_unstarted(const WorkItem& w, const std::string& err) {
      RawRequestResult r;
      init_result(r, w);
      double now = eng_.rel_now();
      r.client_picked_up_ms = now;
      r.client_completed_ms = now;
      r.error = err;
      if (eng_.gate_enabled_ && w.ticket >= 0) eng_.gate_advance(w.ticket);
      eng_.push_completion(w.plan->node_id, std::move(r));
    }

    void start_item(const WorkItem& w) {
      Conn c;
      c.w = w;
      c.kind = w.plan->transport.kind;
      c.t_start_abs = now_ms();
      init_result(c.res, w);
      // client_picked_up stamped AFTER the ticket gate (client_runner.py:116)
      c.res.client_picked_up_ms = c.t_start_abs - eng_.t0_ms_;
      if (!eng_.addr_ok_) {
        c.res.error = eng_.resolve_error_;
        finalize(std::move(c));
        return;
      }
      int fd = make_conn(eng_.addr_);
      if (fd < 0) {
        c.res.error = "connect failed";
        finalize(std::move(c));
        return;
      }
      if (eng_.ep_.request_timeout_s > 0)
        c.deadline_abs = c.t_start_abs + eng_.ep_.request_timeout_s * 1000.0;
      if (is_http_kind(c.kind)) {
        c.h.sse = (c.kind == TransportKind::TEXT_SSE);
        // Timing origin = request initiation, BEFORE connect — same anchor
        // as the Python client's t_start just before its POST.
        c.h.send_time = c.t_start_abs;
      }
      auto it = conns_.emplace(fd, std::move(c)).first;
      arm(fd, it->second, it->second.deadline_abs);
      reactor_.set_interest(fd, false, true);  // connect completes as writable
    }

    // advance points --------------------------------------------------

    void fire_dispatched_point(Conn& c) {
      if (c.dispatched_fired) return;
      c.dispatched_fired = true;
      if (eng_.gate_enabled_ && c.w.ticket >= 0 &&
          eng_.tp_.ordering == TicketOrdering::DISPATCH)
        eng_.gate_advance(c.w.ticket);
    }
    void fire_sent_point(Conn& c) {
      if (c.sent_fired) return;
      c.sent_fired = true;
      if (eng_.gate_enabled_ && c.w.ticket >= 0 &&
          eng_.tp_.ordering == TicketOrdering::PREFILL)
        eng_.gate_advance(c.w.ticket);
    }

    // finalize: move the (possibly partial) result out and hand it to the
    // completion-ack threads. Socket errors never silently complete.
    void finish(int fd, Conn& c, const std::string& error) {
      c.res.error = error;
      Conn moved = std::move(c);
      reactor_.remove(fd);
      close(fd);
      conns_.erase(fd);
      finalize(std::move(moved));
    }

    void finalize(Conn&& c) {
      RawRequestResult& r = c.res;
      if (is_http_kind(c.kind)) {
        r.status = c.h.status;
        if (c.kind == TransportKind::TEXT_SSE) {
          r.content = std::move(c.h.content);
          for (size_t i = 0; i < c.h.offsets.size(); i++) {
            r.recv_stamps.push_back({c.h.offsets[i], c.h.sizes[i]});
            r.recv_bytes += c.h.sizes[i];
          }
        }
      }
      r.client_completed_ms = eng_.rel_now();
      if (eng_.gate_enabled_ && c.w.ticket >= 0 &&
          (eng_.tp_.ordering == TicketOrdering::REQUEST || !r.error.empty()))
        eng_.gate_advance(c.w.ticket);  // request-end / advance-on-error
      eng_.push_completion(c.w.plan->node_id, std::move(r));
    }

    // ---------------- intake / gate ----------------

    void intake_pump() {
      std::deque<WorkItem> items;
      {
        std::lock_guard<std::mutex> lk(sh_.mu);
        items.swap(sh_.q);
        sh_.qlen.store(0);
      }
      for (auto& w : items) {
        if (failed_all_) {
          fail_unstarted(w, "timeout");
          continue;
        }
        if (!eng_.gate_open(w.ticket))
          gated_.push_back(std::move(w));
        else
          start_item(w);  // client_picked_up stamped after the gate
      }
      // re-check parked items (counter may have advanced)
      while (!gated_.empty()) {
        // find any openable item, preserving order among same-gate items
        bool started = false;
        for (auto it = gated_.begin(); it != gated_.end(); ++it) {
          if (eng_.gate_open(it->ticket)) {
            WorkItem w = std::move(*it);
            gated_.erase(it);
            start_item(w);
            started = true;
            break;
          }
        }
        if (!started) break;
      }
    }

    void fail_everything() {
      failed_all_ = true;
      // straggler in-flight connections: fail as timeout, KEEPING partials
      std::vector<int> fds;
      fds.reserve(conns_.size());
      for (auto& kv : conns_) fds.push_back(kv.first);
      for (int fd : fds) {
        auto it = conns_.find(fd);
        if (it != conns_.end()) finish(fd, it->second, "timeout");
      }
      for (auto& w : gated_) fail_unstarted(w, "timeout");
      gated_.clear();
    }

    // ---------------- event handling ----------------

    void handle_event(const ReactorEvent& ev) {
      auto it = conns_.find(ev.fd);
      if (it == conns_.end()) return;
      Conn& c = it->second;
      int fd = ev.fd;
      if (!c.connected && (ev.writable || ev.error)) {
        int err = 0;
        socklen_t len = sizeof(err);
        getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) {
          finish(fd, c, "connect error");
          return;
        }
        c.connected = true;
        if (is_http_kind(c.kind)) {
          std::string wire = eng_.build_http_wire(c.w);
          if (!conn_send(fd, c.outbuf, wire.data(), wire.size())) {
            finish(fd, c, "send error");
            return;
          }
        } else {
          std::string path = eng_.ep_.base_path + c.w.plan->transport.ws_path;
          std::string hs = ws_handshake_request(eng_.ep_.host, eng_.ep_.port,
                                                path, eng_.ep_.headers);
          if (!conn_send(fd, c.outbuf, hs.data(), hs.size())) {
            finish(fd, c, "send error");
            return;
          }
        }
        reactor_.set_interest(fd, true, !c.outbuf.empty());
        return;
      }
      if (!c.connected) return;
      if (ev.writable) {
        if (!flush_outbuf(fd, c.outbuf)) {
          finish(fd, c, "send error");
          return;
        }
        if (!is_http_kind(c.kind) && c.handshaken) {
          if (!ws_pump(fd, c)) return;  // finished inside
        } else {
          reactor_.set_interest(fd, true, !c.outbuf.empty());
        }
      }
      if (ev.readable || ev.error) {
        char buf[16384];
        ssize_t r = recv_retry(fd, buf, sizeof(buf));
        double ts = now_ms();  // read-time stamp, before ANY parsing
        if (r > 0) {
          if (is_http_kind(c.kind))
            http_readable(fd, c, buf, (size_t)r, ts);
          else
            ws_readable(fd, c, buf, (size_t)r, ts);
        } else if (r == 0) {
          on_eof(fd, c);
        } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          finish(fd, c, "recv error");
        } else if (ev.error) {
          finish(fd, c, "socket error");
        }
      }
    }

    void on_eof(int fd, Conn& c) {
      if (is_http_kind(c.kind)) {
        // server closed the stream: llhttp_finish validates message
        // completion (we always send Connection: close, so EOF legitimately
        // ends read-until-EOF bodies / SSE without [DONE]); EOF that
        // truncates chunked or Content-Length framing is an error, as is
        // headers never arriving
        if (!c.h.headers_done)
          finish(fd, c, "connection closed before response headers");
        else if (!http_eof(c.h))
          finish(fd, c, "malformed http response (" + c.h.reason + ")");
        else
          finish(fd, c, "");
      } else {
        if (!c.handshaken)
          finish(fd, c, "connection closed during ws handshake");
        else if (!c.w.plan->transport.done_markers.empty() && !c.ws_done)
          finish(fd, c, "connection closed before done marker");
        else
          finish(fd, c, "");
      }
    }

    // ---------------- HTTP kinds (TEXT_SSE / TTS_HTTP) ----------------

    void http_readable(int fd, Conn& c, const char* buf, size_t n, double ts) {
      size_t prev_events = c.h.offsets.size();
      size_t nb = http_feed(c.h, buf, n, ts);
      if (c.h.headers_done && !c.status_checked) {
        c.status_checked = true;
        if (c.h.status == 200) {
          fire_dispatched_point(c);
        } else {
          finish(fd, c, "http status " + std::to_string(c.h.status));
          return;
        }
      }
      if (c.kind == TransportKind::TTS_HTTP && nb > 0) {
        // per-chunk read-time stamps; audio bytes are counted, not kept
        c.res.recv_stamps.push_back({ts - c.h.send_time, (int)nb});
        c.res.recv_bytes += (long long)nb;
        c.h.content.clear();
        fire_sent_point(c);
      }
      if (c.kind == TransportKind::TEXT_SSE &&
          c.h.offsets.size() > prev_events)
        fire_sent_point(c);
      if (c.h.malformed) {
        finish(fd, c, "malformed http response (" + c.h.reason + ")");
        return;
      }
      if (c.h.done) finish(fd, c, "");
    }

    // ---------------- WS kinds (TTS_REALTIME_WS / STT_WS) ----------------

    void ws_readable(int fd, Conn& c, const char* buf, size_t n, double ts) {
      c.inbuf.append(buf, n);
      if (!c.handshaken) {
        std::string err;
        int hs = ws_check_handshake(c.inbuf, &err);
        if (hs == 0) return;  // need more bytes
        if (hs < 0) {
          finish(fd, c, err);
          return;
        }
        c.handshaken = true;
        c.res.status = 101;
        fire_dispatched_point(c);  // handshake-ok = dispatched
        // pacing anchor: TTS realtime -> post-handshake
        if (c.kind == TransportKind::TTS_REALTIME_WS)
          c.pace_anchor_abs = now_ms();
        // setup frames: asap, in order
        for (const auto& f : c.w.plan->transport.setup_frames) {
          std::string frame = ws_encode_frame(0x1, eng_.frame_payload(f));
          if (!conn_send(fd, c.outbuf, frame.data(), frame.size())) {
            finish(fd, c, "send error");
            return;
          }
        }
        if (!ws_pump(fd, c)) return;
      }
      bool got_close = false;
      ws_extract_frames(c.inbuf, c.outbuf, c.frag, got_close,
                        [&](std::string&& payload) {
                          on_ws_data(c, std::move(payload), ts);
                        });
      if (c.ws_done) {
        finish(fd, c, "");
        return;
      }
      if (got_close) {
        on_eof(fd, c);
        return;
      }
      if (!c.outbuf.empty()) reactor_.set_interest(fd, true, true);
    }

    void on_ws_data(Conn& c, std::string&& payload, double ts) {
      double off = ts - c.t_start_abs;
      std::string type = extract_type_string(payload);
      if (!type.empty()) c.res.event_offsets_ms.push_back({type, off});
      // content deltas get recv stamps; protocol events only event offsets
      bool is_delta = type.empty() || type.find("delta") != std::string::npos;
      if (is_delta) {
        c.res.recv_stamps.push_back({off, (int)payload.size()});
        c.res.recv_bytes += (long long)payload.size();
        if (c.kind == TransportKind::STT_WS)
          extract_delta_text(payload, c.res.content);
        fire_sent_point(c);  // first content chunk = sent
      }
      for (const auto& m : c.w.plan->transport.done_markers) {
        if (payload.find(m) != std::string::npos) {
          c.ws_done = true;
          break;
        }
      }
    }

    // Absolute-deadline paced sends; per-loop deadline heap. Returns false if
    // the connection was finished (caller must stop touching it).
    bool ws_pump(int fd, Conn& c) {
      bool failed = false;
      double next = pump_paced(fd, c, &failed);
      if (failed) {
        // A paced-send failure is a FAILED result: silently completing with
        // undelivered messages would fake a healthy paced stream.
        finish(fd, c, "paced send error");
        return false;
      }
      reactor_.set_interest(fd, true, !c.outbuf.empty());
      arm(fd, c, next);
      return true;
    }

    // Send every paced frame whose absolute deadline arrived; stamp the send
    // offset BEFORE the socket send. Returns the ABSOLUTE next deadline, or
    // kFar when nothing is pending (finish frames flushed). Never finishes
    // the connection itself — sets *failed instead (the caller owns Conn
    // lifetime).
    double pump_paced(int fd, Conn& c, bool* failed) {
      const auto& t = c.w.plan->transport;
      if (!c.outbuf.empty()) {
        if (!flush_outbuf(fd, c.outbuf)) {
          *failed = true;
          return kFar;
        }
        if (!c.outbuf.empty()) return kFar;  // resume on writable
      }
      double now = now_ms();
      while (c.next_paced < t.paced_frames.size()) {
        const WsFrame& f = t.paced_frames[c.next_paced];
        double due = 0.0;  // asap
        if (f.send_offset_ms >= 0.0) {
          if (c.kind == TransportKind::STT_WS) {
            // anchor = the first paced send (stt.py audio_started_at); the
            // first paced frame goes out asap and DEFINES the schedule zero
            due = (c.audio_anchor_abs < 0)
                      ? 0.0
                      : c.audio_anchor_abs + f.send_offset_ms;
          } else {
            due = c.pace_anchor_abs + f.send_offset_ms;
          }
        }
        if (due > 0.0 && now + 0.05 < due) return due;
        double stamp = now_ms();
        if (c.kind == TransportKind::STT_WS && c.audio_anchor_abs < 0)
          c.audio_anchor_abs = stamp;
        double base = (c.kind == TransportKind::STT_WS) ? c.audio_anchor_abs
                                                        : c.pace_anchor_abs;
        c.res.send_offsets_ms.push_back(stamp - base);  // BEFORE send()
        std::string frame = ws_encode_frame(0x1, eng_.frame_payload(f));
        if (!conn_send(fd, c.outbuf, frame.data(), frame.size())) {
          *failed = true;
          return kFar;
        }
        c.next_paced++;
        if (!c.outbuf.empty()) return kFar;  // backlogged: resume on writable
        now = now_ms();
      }
      if (!c.finish_sent) {
        for (const auto& f : t.finish_frames) {
          std::string frame = ws_encode_frame(0x1, eng_.frame_payload(f));
          if (!conn_send(fd, c.outbuf, frame.data(), frame.size())) {
            *failed = true;
            return kFar;
          }
        }
        c.finish_sent = true;
      }
      return kFar;
    }

    // timer pops: request-timeout budget first, then paced sends; always
    // re-arm the surviving earliest deadline (entries are one-shot)
    void handle_timers(int fd) {
      auto it = conns_.find(fd);
      if (it == conns_.end()) return;  // stale timer (fd closed/reused)
      Conn& c = it->second;
      c.next_wake_armed = kFar;
      double now = now_ms();
      if (now >= c.deadline_abs) {
        finish(fd, c, "timeout");  // budget exceeded; partials kept
        return;
      }
      if (!is_http_kind(c.kind) && c.handshaken) {
        if (!ws_pump(fd, c)) return;  // arms the next pace deadline
      }
      auto it2 = conns_.find(fd);
      if (it2 != conns_.end()) arm(fd, it2->second, it2->second.deadline_abs);
    }
  };

  void reactor_loop(int idx) {
    ReactorWorker w(*this, idx);
    w.run();
    if (live_reactors_.fetch_sub(1) == 1) {
      reactors_done_.store(true);
      std::lock_guard<std::mutex> lk(comp_mu_);
      comp_cv_.notify_all();
    }
  }

  // ------------------------------------------------------------------
  // completion-ack threads
  // ------------------------------------------------------------------

  void push_completion(int node_id, RawRequestResult&& r) {
    std::lock_guard<std::mutex> lk(comp_mu_);
    comp_q_.push_back(CompItem{node_id, std::move(r)});
    comp_cv_.notify_one();
  }

  void ack_loop() {
    set_thread_qos();
    while (true) {
      CompItem item;
      {
        std::unique_lock<std::mutex> lk(comp_mu_);
        if (comp_q_.empty()) {
          if (reactors_done_.load()) break;
          comp_cv_.wait_for(lk, std::chrono::milliseconds(50));
          if (comp_q_.empty()) continue;
        }
        item = std::move(comp_q_.front());
        comp_q_.pop_front();
      }
      RawRequestResult& r = item.res;
      // result_processed stamped at native ack (completion.py:50 analogue)
      r.result_processed_ms = rel_now();
      bool success = r.error.empty();
      // scheduler FIRST — children release / refill / cancel-on-failure must
      // never wait on event consumers (they are off-path by construction)
      scheduler_notify_completion(r.session_id, item.node_id,
                                  r.client_completed_ms, success, r.content);
      {
        std::lock_guard<std::mutex> tlk(track_mu_);
        in_flight_ids_.erase(r.request_id);
      }
      c_in_flight_.fetch_sub(1);
      c_completed_.fetch_add(1);

      NativeLoopEvent ev;
      ev.kind = EventKind::COMPLETED;
      ev.request_id = r.request_id;
      ev.session_id = r.session_id;
      ev.session_total_requests = r.session_total_requests;
      ev.ready_ms = r.scheduler_ready_ms;
      ev.dispatched_ms = r.scheduler_dispatched_ms;
      ev.result = std::move(r);
      emit_event(std::move(ev));
    }
    if (live_acks_.fetch_sub(1) == 1) {
      loop_done_.store(true);
      {
        std::lock_guard<std::mutex> lk(done_mu_);
        done_cv_.notify_all();
      }
      std::lock_guard<std::mutex> lk(ev_mu_);
      ev_cv_.notify_all();
    }
  }

  void join_threads() {
    std::lock_guard<std::mutex> lk(join_mu_);
    for (auto& t : threads_)
      if (t.joinable()) t.join();
    if (grace_thread_.joinable()) grace_thread_.join();
  }

  // ------------------------------------------------------------------
  // members
  // ------------------------------------------------------------------

  NativeRuntimeConfig rt_;
  TrafficPlanConfig tp_;
  EndpointConfig ep_;
  double py_anchor_s_;
  double t0_ms_ = 0.0;
  sockaddr_in addr_{};
  bool addr_ok_ = false;
  std::string resolve_error_;
  bool gate_enabled_ = false;

  // blobs
  mutable std::shared_mutex blobs_mu_;
  std::deque<std::string> blobs_;

  // intake ring (bounded MPSC)
  std::mutex intake_mu_;
  std::condition_variable intake_cv_, intake_space_cv_;
  std::deque<SessionPlan> intake_;
  bool intake_closed_ = false;
  std::atomic<bool> intake_closed_flag_{false};

  // scheduler
  std::mutex sched_mu_;
  std::condition_variable sched_cv_;
  std::priority_queue<ReadyItem, std::vector<ReadyItem>, ReadyCmp> ready_;
  long long ready_seq_ = 0;
  std::unordered_map<long long, std::unique_ptr<SessState>> sessions_;
  std::deque<SessionPlan> pending_sessions_;  // CONCURRENT
  std::deque<double> intervals_;              // RATE draws (seconds)
  double next_start_ms_ = 0.0;
  bool need_interval_ = false;
  bool rampup_complete_ = false;
  long long next_ticket_ = 0;
  bool stop_dispatch_ = false;
  bool intake_drained_ = false;

  // ticket gate
  std::atomic<long long> gate_counter_{0};

  // reactors
  std::vector<std::unique_ptr<Shard>> shards_;

  // completion queue
  std::mutex comp_mu_;
  std::condition_variable comp_cv_;
  std::deque<CompItem> comp_q_;

  // ordered event log
  std::mutex ev_mu_;
  std::condition_variable ev_cv_;
  std::deque<NativeLoopEvent> events_;

  // id snapshots
  std::mutex track_mu_;
  std::unordered_set<long long> in_flight_ids_, dispatched_ids_;

  // counters (atomics; lock-free monitor snapshot)
  std::atomic<long long> c_sessions_completed_{0}, c_sessions_errored_{0},
      c_sessions_seen_{0}, c_dispatched_{0}, c_completed_{0}, c_in_flight_{0};

  // lifecycle
  std::atomic<bool> stopping_{false};
  std::atomic<bool> fail_all_{false};
  std::atomic<int> live_dispatchers_{0}, live_reactors_{0}, live_acks_{0};
  std::atomic<bool> dispatchers_done_{false}, reactors_done_{false};
  std::atomic<bool> loop_done_{false};
  std::mutex done_mu_, join_mu_;
  std::condition_variable done_cv_;
  std::vector<std::thread> threads_;
  std::thread grace_thread_;

  friend class ReactorWorker;
};

// ===========================================================================
// §3 pybind surface
// ===========================================================================

PYBIND11_MODULE(veeksha_native, m, py::mod_gil_not_used()) {
  m.doc() = "Native (C++) benchmark loop for Veeksha.";

#if defined(VEEKSHA_USE_KQUEUE)
  m.attr("reactor_backend") = use_kernel_queue() ? "kqueue" : "poll";
#elif defined(VEEKSHA_USE_EPOLL)
  m.attr("reactor_backend") = use_kernel_queue() ? "epoll" : "poll";
#else
  m.attr("reactor_backend") = "poll";
#endif

  py::enum_<TrafficKind>(m, "TrafficKind")
      .value("RATE", TrafficKind::RATE)
      .value("CONCURRENT", TrafficKind::CONCURRENT)
      .value("SEQUENTIAL_LAUNCH", TrafficKind::SEQUENTIAL_LAUNCH);

  py::enum_<TicketOrdering>(m, "TicketOrdering")
      .value("DISPATCH", TicketOrdering::DISPATCH)
      .value("PREFILL", TicketOrdering::PREFILL)
      .value("REQUEST", TicketOrdering::REQUEST);

  py::enum_<TransportKind>(m, "TransportKind")
      .value("TEXT_SSE", TransportKind::TEXT_SSE)
      .value("TTS_HTTP", TransportKind::TTS_HTTP)
      .value("TTS_REALTIME_WS", TransportKind::TTS_REALTIME_WS)
      .value("STT_WS", TransportKind::STT_WS);

  py::enum_<EventKind>(m, "EventKind")
      .value("DISPATCHED", EventKind::DISPATCHED)
      .value("COMPLETED", EventKind::COMPLETED);

  py::class_<NativeRuntimeConfig>(m, "NativeRuntimeConfig")
      .def(py::init<>())
      .def_readwrite("max_sessions", &NativeRuntimeConfig::max_sessions)
      .def_readwrite("benchmark_timeout_s",
                     &NativeRuntimeConfig::benchmark_timeout_s)
      .def_readwrite("post_timeout_grace_s",
                     &NativeRuntimeConfig::post_timeout_grace_s)
      .def_readwrite("num_dispatcher_threads",
                     &NativeRuntimeConfig::num_dispatcher_threads)
      .def_readwrite("num_completion_threads",
                     &NativeRuntimeConfig::num_completion_threads)
      .def_readwrite("num_client_threads",
                     &NativeRuntimeConfig::num_client_threads);

  py::class_<TrafficPlanConfig>(m, "TrafficPlanConfig")
      .def(py::init<>())
      .def_readwrite("kind", &TrafficPlanConfig::kind)
      .def_readwrite("target_concurrent_sessions",
                     &TrafficPlanConfig::target_concurrent_sessions)
      .def_readwrite("rampup_seconds", &TrafficPlanConfig::rampup_seconds)
      .def_readwrite("ordering", &TrafficPlanConfig::ordering)
      .def_readwrite("cancel_session_on_failure",
                     &TrafficPlanConfig::cancel_session_on_failure);

  py::class_<EndpointConfig>(m, "EndpointConfig")
      .def(py::init<>())
      .def_readwrite("host", &EndpointConfig::host)
      .def_readwrite("port", &EndpointConfig::port)
      .def_readwrite("base_path", &EndpointConfig::base_path)
      .def_readwrite("headers", &EndpointConfig::headers)
      .def_readwrite("request_timeout_s", &EndpointConfig::request_timeout_s);

  py::class_<BlobRef>(m, "BlobRef")
      .def(py::init<>())
      .def_readwrite("blob_id", &BlobRef::blob_id)
      .def_readwrite("offset", &BlobRef::offset)
      .def_readwrite("len", &BlobRef::len);

  py::class_<WsFrame>(m, "WsFrame")
      .def(py::init<>())
      .def_readwrite("payload", &WsFrame::payload)
      .def_readwrite("blob", &WsFrame::blob)
      .def_readwrite("send_offset_ms", &WsFrame::send_offset_ms);

  py::class_<TransportPlan>(m, "TransportPlan")
      .def(py::init<>())
      .def_readwrite("kind", &TransportPlan::kind)
      .def_readwrite("header_prefix", &TransportPlan::header_prefix)
      .def_readwrite("header_suffix", &TransportPlan::header_suffix)
      .def_readwrite("body_segments", &TransportPlan::body_segments)
      .def_readwrite("history_refs", &TransportPlan::history_refs)
      .def_readwrite("ws_path", &TransportPlan::ws_path)
      .def_readwrite("setup_frames", &TransportPlan::setup_frames)
      .def_readwrite("paced_frames", &TransportPlan::paced_frames)
      .def_readwrite("finish_frames", &TransportPlan::finish_frames)
      .def_readwrite("done_markers", &TransportPlan::done_markers);

  py::class_<RequestPlan>(m, "RequestPlan")
      .def(py::init<>())
      .def_readwrite("request_id", &RequestPlan::request_id)
      .def_readwrite("node_id", &RequestPlan::node_id)
      .def_readwrite("wait_after_ready_s", &RequestPlan::wait_after_ready_s)
      .def_readwrite("parents", &RequestPlan::parents)
      .def_readwrite("transport", &RequestPlan::transport);

  py::class_<SessionPlan>(m, "SessionPlan")
      .def(py::init<>())
      .def_readwrite("session_id", &SessionPlan::session_id)
      .def_readwrite("requests", &SessionPlan::requests)
      .def_readwrite("dispatch_ticket_base", &SessionPlan::dispatch_ticket_base);

  py::class_<ChunkStamp>(m, "ChunkStamp")
      .def(py::init<>())
      .def_readwrite("offset_ms", &ChunkStamp::offset_ms)
      .def_readwrite("size", &ChunkStamp::size);

  py::class_<RawRequestResult>(m, "RawRequestResult")
      .def(py::init<>())
      .def_readwrite("request_id", &RawRequestResult::request_id)
      .def_readwrite("session_id", &RawRequestResult::session_id)
      .def_readwrite("session_total_requests",
                     &RawRequestResult::session_total_requests)
      .def_readwrite("status", &RawRequestResult::status)
      .def_readwrite("error", &RawRequestResult::error)
      .def_readwrite("scheduler_ready_ms", &RawRequestResult::scheduler_ready_ms)
      .def_readwrite("scheduler_dispatched_ms",
                     &RawRequestResult::scheduler_dispatched_ms)
      .def_readwrite("client_picked_up_ms",
                     &RawRequestResult::client_picked_up_ms)
      .def_readwrite("client_completed_ms",
                     &RawRequestResult::client_completed_ms)
      .def_readwrite("result_processed_ms",
                     &RawRequestResult::result_processed_ms)
      .def_readwrite("recv_stamps", &RawRequestResult::recv_stamps)
      .def_readwrite("send_offsets_ms", &RawRequestResult::send_offsets_ms)
      .def_readwrite("content", &RawRequestResult::content)
      .def_readwrite("recv_bytes", &RawRequestResult::recv_bytes)
      .def_readwrite("event_offsets_ms", &RawRequestResult::event_offsets_ms);

  py::class_<NativeLoopEvent>(m, "NativeLoopEvent")
      .def(py::init<>())
      .def_readwrite("kind", &NativeLoopEvent::kind)
      .def_readwrite("request_id", &NativeLoopEvent::request_id)
      .def_readwrite("session_id", &NativeLoopEvent::session_id)
      .def_readwrite("session_total_requests",
                     &NativeLoopEvent::session_total_requests)
      .def_readwrite("ready_ms", &NativeLoopEvent::ready_ms)
      .def_readwrite("dispatched_ms", &NativeLoopEvent::dispatched_ms)
      .def_readwrite("result", &NativeLoopEvent::result);

  py::class_<NativeLoopCounters>(m, "NativeLoopCounters")
      .def(py::init<>())
      .def_readwrite("sessions_completed", &NativeLoopCounters::sessions_completed)
      .def_readwrite("sessions_errored", &NativeLoopCounters::sessions_errored)
      .def_readwrite("sessions_seen", &NativeLoopCounters::sessions_seen)
      .def_readwrite("requests_dispatched",
                     &NativeLoopCounters::requests_dispatched)
      .def_readwrite("requests_completed", &NativeLoopCounters::requests_completed)
      .def_readwrite("in_flight", &NativeLoopCounters::in_flight)
      .def_readwrite("intake_exhausted", &NativeLoopCounters::intake_exhausted)
      .def_readwrite("idle", &NativeLoopCounters::idle);

  py::class_<NativeBenchmarkLoop>(m, "NativeBenchmarkLoop")
      .def(py::init<NativeRuntimeConfig, TrafficPlanConfig, EndpointConfig,
                    double>(),
           py::arg("runtime"), py::arg("traffic"), py::arg("endpoint"),
           py::arg("py_monotonic_anchor_s"))
      .def_property_readonly("loop_epoch_py_monotonic_s",
                             &NativeBenchmarkLoop::loop_epoch_py_monotonic_s)
      // ---- intake (called by the Python feeder thread) ----
      .def("register_blob", &NativeBenchmarkLoop::register_blob,
           py::arg("data"),
           "Register shared payload bytes (e.g. one audio clip's pre-encoded "
           "append frames); returns blob_id for WsFrame.blob references.")
      .def("feed_sessions", &NativeBenchmarkLoop::feed_sessions,
           py::arg("plans"), py::call_guard<py::gil_scoped_release>(),
           "Enqueue compiled SessionPlans. Blocks (GIL released) if the "
           "intake ring is full — natural backpressure replacing "
           "PrefetchWorker's throttle. Returns number accepted.")
      .def("feed_intervals", &NativeBenchmarkLoop::feed_intervals,
           py::arg("intervals_s"), py::call_guard<py::gil_scoped_release>(),
           "RATE only: stream the next batch of seeded interarrival draws.")
      .def("close_intake", &NativeBenchmarkLoop::close_intake,
           "Generator exhausted / max_sessions reached; loop may finish.")
      // ---- drain (called by the Python drainer thread) ----
      .def("drain_events", &NativeBenchmarkLoop::drain_events,
           py::arg("max_items") = 256, py::arg("timeout_s") = 0.1,
           py::call_guard<py::gil_scoped_release>(),
           "Pop up to max_items ordered NativeLoopEvents; blocks up to timeout_s "
           "(GIL released while waiting). Empty list = nothing pending.")
      // ---- monitor / control (called by the Python monitor thread) ----
      .def("counters", &NativeBenchmarkLoop::counters)
      .def("in_flight_request_ids",
           &NativeBenchmarkLoop::in_flight_request_ids)
      .def("dispatched_request_ids",
           &NativeBenchmarkLoop::dispatched_request_ids)
      .def("request_stop", &NativeBenchmarkLoop::request_stop,
           py::arg("grace_s") = 0.0,
           "Benchmark-timeout path: stop dispatching new work; after grace_s, "
           "fail remaining in-flight as timed out (keeping partial streams).")
      .def("join", &NativeBenchmarkLoop::join, py::arg("timeout_s"),
           py::call_guard<py::gil_scoped_release>());
}
