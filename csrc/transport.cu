// Transfers between processes on one host whose GPUs cannot reach each other's memory (no peer access): a ring of
// slots per direction in shared page-locked host memory, filled and drained by the copy engines. Sequence flags in
// the same memory tell the other side what is there. Each direction of a link has a host thread of its own that
// watches the flags and issues the copies only when they can run at once, so no GPU stream ever waits for the other
// process (a kernel spinning on a flag can deadlock two GPUs through lazy module loading or shared hardware queues),
// and the caller's thread never waits either, except for data it asks to use before it has arrived.
//
//   send of message seq to slot k = (seq - 1) % n:  wait (host) until read[k] >= seq - n; copy the data into the
//                                                   slot on the send stream; when the copy is done, written[k] = seq
//   receipt of seq:                                 wait (host) until written[k] >= seq; copy the slot out on the
//                                                   receive stream; when the copy is done, read[k] = seq
//
// Both ends number a direction's messages identically (they issue them in the same order). A message larger than a
// slot takes consecutive slots, so the sender's copy of one piece overlaps the receiver's copy of the previous one.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

constexpr int64_t kHeader = 64 * 1024;      // flags: written[slots], read[slots]; data after the header

struct Ring {
  std::string name;
  size_t bytes = 0;
  int64_t slot_bytes = 0, slots = 0;
  unsigned char* host = nullptr;
  uint32_t* written = nullptr;
  uint32_t* read = nullptr;
  unsigned char* data = nullptr;
};

size_t ring_bytes(int64_t slot_bytes, int64_t slots) { return kHeader + static_cast<size_t>(slot_bytes) * slots; }

Ring map_ring(const std::string& name, int64_t slot_bytes, int64_t slots) {
  const size_t bytes = ring_bytes(slot_bytes, slots);
  const int fd = shm_open(name.c_str(), O_RDWR, 0600);
  TORCH_CHECK(fd >= 0, "link: shm_open(", name, ") failed (the receiving side creates the ring)");
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  TORCH_CHECK(p != MAP_FAILED, "link: mmap failed");
  C10_CUDA_CHECK(cudaHostRegister(p, bytes, cudaHostRegisterPortable));
  Ring r;
  r.name = name;
  r.bytes = bytes;
  r.slot_bytes = slot_bytes;
  r.slots = slots;
  r.host = static_cast<unsigned char*>(p);
  r.written = reinterpret_cast<uint32_t*>(r.host);
  r.read = r.written + slots;
  r.data = r.host + kHeader;
  return r;
}

uint32_t load(const uint32_t* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
void store(uint32_t* p, uint32_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
bool reached(uint32_t have, uint32_t want) { return static_cast<int32_t>(have - want) >= 0; }

// back-off for polling loops: spin briefly, then yield, then sleep in growing steps (at most 50 us)
struct Backoff {
  int n = 0;
  void reset() { n = 0; }
  void pause() {
    if (n < 64) {
      ++n;
    } else if (n < 256) {
      ++n;
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(std::min(50, n++ - 255)));
    }
  }
};

// CUDA calls of the link threads: a failure is recorded (and raised by the caller's next send / take) instead of
// thrown from a thread, which would terminate the process
#define BL_LINK_CUDA(call)                                         \
  do {                                                             \
    const cudaError_t e_ = (call);                                 \
    if (e_ != cudaSuccess) {                                       \
      fail(cudaGetErrorString(e_));                                \
      return;                                                      \
    }                                                              \
  } while (0)

struct Piece {
  int64_t slot;
  uint32_t seq;
  cudaEvent_t done;
};

struct Incoming {
  at::Tensor out;
  uint32_t seq;
  cudaEvent_t ready = nullptr;
  std::atomic<bool> issued{false};
};

class Link {
 public:
  Link(const std::string& in_name, const std::string& out_name, int64_t slot_bytes, int64_t slots, int device)
      : device_(device) {
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
    in_ = map_ring(in_name, slot_bytes, slots);
    out_ = map_ring(out_name, slot_bytes, slots);
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&send_stream_, cudaStreamNonBlocking));
    C10_CUDA_CHECK(cudaStreamCreateWithFlags(&recv_stream_, cudaStreamNonBlocking));
    send_thread_ = std::thread([this] { send_loop(); });
    recv_thread_ = std::thread([this] { recv_loop(); });
  }

  ~Link() { close(false); }

  void close(bool unlink_in) {
    if (closed_.exchange(true)) return;
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    taken_.notify_all();
    cv_.notify_all();
    send_thread_.join();
    recv_thread_.join();
    c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device_));
    cudaStreamSynchronize(send_stream_);
    cudaStreamSynchronize(recv_stream_);
    for (Ring* r : {&in_, &out_}) {
      cudaHostUnregister(r->host);
      munmap(r->host, r->bytes);
    }
    if (unlink_in) shm_unlink(in_.name.c_str());
    cudaStreamDestroy(send_stream_);
    cudaStreamDestroy(recv_stream_);
    for (cudaEvent_t e : events_) cudaEventDestroy(e);
  }

  // after the work issued so far on the current stream, send x (kept alive until it has been copied)
  void send(const at::Tensor& x) {
    check();
    cudaEvent_t ready = event();
    C10_CUDA_CHECK(cudaEventRecord(ready, at::cuda::getCurrentCUDAStream()));
    std::lock_guard<std::mutex> lk(mu_);
    sends_.push_back({x, ready, seq_out_});
    seq_out_ += pieces(x);
    cv_.notify_all();
  }

  // post the receipt of the next message into out; returns a ticket for take()
  int64_t recv(const at::Tensor& out) {
    auto in = std::make_shared<Incoming>();
    in->out = out;
    in->ready = event();
    std::lock_guard<std::mutex> lk(mu_);
    in->seq = seq_in_;
    seq_in_ += pieces(out);
    const int64_t t = next_ticket_++;
    tickets_[t] = in;
    recvs_.push_back(in);
    cv_.notify_all();
    return t;
  }

  // the current stream waits (on the GPU) for the copy of the ticket's message; the caller waits only until that
  // copy has been issued, i.e. until the data has arrived in host memory
  void take(int64_t ticket) {
    std::shared_ptr<Incoming> in;
    {
      std::unique_lock<std::mutex> lk(mu_);
      auto it = tickets_.find(ticket);
      TORCH_CHECK(it != tickets_.end(), "link: unknown ticket ", ticket);
      in = it->second;
      tickets_.erase(it);
      taken_.wait(lk, [&] { return in->issued.load() || stop_.load(); });
    }
    check();
    TORCH_CHECK(in->issued.load(), "link closed before the message arrived");
    C10_CUDA_CHECK(cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream(), in->ready, 0));
    give(in->ready);
  }

  int64_t pieces(const at::Tensor& t) const {
    const int64_t n = t.numel() * t.element_size();
    return std::max<int64_t>(1, (n + out_.slot_bytes - 1) / out_.slot_bytes);
  }

 private:
  void fail(const char* what) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (error_.empty()) error_ = what;
      stop_ = true;
    }
    cv_.notify_all();
    taken_.notify_all();
  }

  void check() {
    std::lock_guard<std::mutex> lk(mu_);
    TORCH_CHECK(error_.empty(), "link failed: ", error_);
  }

  struct Outgoing {
    at::Tensor x;
    cudaEvent_t ready;
    uint32_t seq;
  };

  cudaEvent_t event() {
    std::lock_guard<std::mutex> lk(ev_mu_);
    if (!free_events_.empty()) {
      cudaEvent_t e = free_events_.back();
      free_events_.pop_back();
      return e;
    }
    cudaEvent_t e = nullptr;
    if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) return nullptr;
    events_.push_back(e);
    return e;
  }

  void give(cudaEvent_t e) {
    std::lock_guard<std::mutex> lk(ev_mu_);
    free_events_.push_back(e);
  }

  // completed copies publish their flags, in order
  void publish(std::deque<Piece>& inflight, uint32_t* flags) {
    while (!inflight.empty() && cudaEventQuery(inflight.front().done) == cudaSuccess) {
      store(flags + inflight.front().slot, inflight.front().seq);
      give(inflight.front().done);
      inflight.pop_front();
    }
  }

  void send_loop() {
    cudaSetDevice(device_);
    std::deque<Piece> inflight;
    std::deque<std::pair<at::Tensor, cudaEvent_t>> holding;     // sent tensors until their last copy is done
    Backoff bo;
    for (;;) {
      Outgoing req;
      {
        std::unique_lock<std::mutex> lk(mu_);
        if (sends_.empty()) {
          if (inflight.empty() && holding.empty()) cv_.wait(lk, [&] { return stop_.load() || !sends_.empty(); });
          if (stop_.load() && sends_.empty() && inflight.empty()) return;
        }
        if (!sends_.empty()) {
          req = sends_.front();
          sends_.pop_front();
        }
      }
      if (!req.x.defined()) {
        publish(inflight, out_.written);
        while (!holding.empty() && cudaEventQuery(holding.front().second) == cudaSuccess) {
          give(holding.front().second);
          holding.pop_front();
        }
        bo.pause();
        continue;
      }
      bo.reset();
      BL_LINK_CUDA(cudaStreamWaitEvent(send_stream_, req.ready, 0));
      const auto* src = static_cast<const unsigned char*>(req.x.data_ptr());
      const int64_t n = req.x.numel() * req.x.element_size();
      uint32_t s = req.seq;
      for (int64_t off = 0; off < n || (n == 0 && off == 0); off += out_.slot_bytes, ++s) {
        const int64_t k = (s - 1) % out_.slots;
        Backoff wait;
        while (s > out_.slots && !reached(load(out_.read + k), s - static_cast<uint32_t>(out_.slots))) {
          publish(inflight, out_.written);
          wait.pause();
        }
        const int64_t len = std::min<int64_t>(out_.slot_bytes, n - off);
        if (len > 0)
          BL_LINK_CUDA(cudaMemcpyAsync(out_.data + k * out_.slot_bytes, src + off, len, cudaMemcpyDeviceToHost,
                                         send_stream_));
        cudaEvent_t e = event();
        BL_LINK_CUDA(cudaEventRecord(e, send_stream_));
        inflight.push_back({k, s, e});
        publish(inflight, out_.written);
        if (n == 0) break;
      }
      cudaEvent_t last = event();
      BL_LINK_CUDA(cudaEventRecord(last, send_stream_));
      holding.emplace_back(req.x, last);
      give(req.ready);
    }
  }

  void recv_loop() {
    cudaSetDevice(device_);
    std::deque<Piece> acks;
    Backoff bo;
    for (;;) {
      std::shared_ptr<Incoming> in;
      {
        std::unique_lock<std::mutex> lk(mu_);
        if (recvs_.empty()) {
          if (acks.empty()) cv_.wait(lk, [&] { return stop_.load() || !recvs_.empty(); });
          if (stop_.load() && recvs_.empty()) {
            taken_.notify_all();
            return;
          }
        }
        if (!recvs_.empty()) {
          in = recvs_.front();
          recvs_.pop_front();
        }
      }
      if (!in) {
        publish(acks, in_.read);
        bo.pause();
        continue;
      }
      bo.reset();
      auto* dst = static_cast<unsigned char*>(in->out.data_ptr());
      const int64_t n = in->out.numel() * in->out.element_size();
      uint32_t s = in->seq;
      for (int64_t off = 0; off < n || (n == 0 && off == 0); off += in_.slot_bytes, ++s) {
        const int64_t k = (s - 1) % in_.slots;
        Backoff wait;
        while (!reached(load(in_.written + k), s)) {
          publish(acks, in_.read);
          if (stop_.load()) return;
          wait.pause();
        }
        const int64_t len = std::min<int64_t>(in_.slot_bytes, n - off);
        if (len > 0)
          BL_LINK_CUDA(cudaMemcpyAsync(dst + off, in_.data + k * in_.slot_bytes, len, cudaMemcpyHostToDevice,
                                         recv_stream_));
        cudaEvent_t e = event();
        BL_LINK_CUDA(cudaEventRecord(e, recv_stream_));
        acks.push_back({k, s, e});
        if (n == 0) break;
      }
      BL_LINK_CUDA(cudaEventRecord(in->ready, recv_stream_));
      {
        std::lock_guard<std::mutex> lk(mu_);
        in->issued = true;
      }
      taken_.notify_all();
      publish(acks, in_.read);
    }
  }

  int device_;
  Ring in_, out_;
  cudaStream_t send_stream_ = nullptr, recv_stream_ = nullptr;
  std::thread send_thread_, recv_thread_;
  std::mutex mu_, ev_mu_;
  std::condition_variable cv_, taken_;
  std::deque<Outgoing> sends_;
  std::deque<std::shared_ptr<Incoming>> recvs_;
  std::unordered_map<int64_t, std::shared_ptr<Incoming>> tickets_;
  std::vector<cudaEvent_t> events_, free_events_;
  uint32_t seq_out_ = 1, seq_in_ = 1;
  int64_t next_ticket_ = 0;
  std::atomic<bool> stop_{false};
  std::atomic<bool> closed_{false};
  std::string error_;
};

std::mutex g_mu;
// never destroyed: links are closed explicitly (bitlattice.transport closes them at interpreter exit), and a static
// destructor would run after the CUDA runtime has been unloaded
auto& g_links = *new std::vector<std::unique_ptr<Link>>();

Link& link(int64_t h) {
  std::lock_guard<std::mutex> lk(g_mu);
  TORCH_CHECK(h >= 0 && h < static_cast<int64_t>(g_links.size()) && g_links[h], "unknown or closed link ", h);
  return *g_links[h];
}

}  // namespace

// Creates (or resets) the shared ring ``name``: the receiving side calls it before either side opens a link.
void bl_ring_create(std::string name, int64_t slot_bytes, int64_t slots) {
  TORCH_CHECK(slots > 0 && slots <= 4096 && slot_bytes > 0 && slot_bytes % 256 == 0,
              "ring: 1..4096 slots of a positive multiple of 256 bytes");
  const size_t bytes = ring_bytes(slot_bytes, slots);
  const int fd = shm_open(name.c_str(), O_RDWR | O_CREAT, 0600);
  TORCH_CHECK(fd >= 0, "ring: shm_open(", name, ") failed");
  TORCH_CHECK(ftruncate(fd, static_cast<off_t>(bytes)) == 0, "ring: ftruncate failed");
  void* p = mmap(nullptr, kHeader, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  TORCH_CHECK(p != MAP_FAILED, "ring: mmap failed");
  std::memset(p, 0, kHeader);                         // a name left over by an earlier run starts clean
  munmap(p, kHeader);
}

int64_t bl_link_open(std::string in_name, std::string out_name, int64_t slot_bytes, int64_t slots, int64_t device) {
  auto l = std::make_unique<Link>(in_name, out_name, slot_bytes, slots, static_cast<int>(device));
  std::lock_guard<std::mutex> lk(g_mu);
  g_links.push_back(std::move(l));
  return static_cast<int64_t>(g_links.size()) - 1;
}

void bl_link_close(int64_t h, bool unlink_in) {
  std::unique_ptr<Link> l;
  {
    std::lock_guard<std::mutex> lk(g_mu);
    TORCH_CHECK(h >= 0 && h < static_cast<int64_t>(g_links.size()), "unknown link");
    l = std::move(g_links[h]);
  }
  if (l) l->close(unlink_in);
}

void bl_link_send(int64_t h, at::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "link send: a contiguous CUDA tensor");
  const c10::cuda::CUDAGuard guard(x.device());
  link(h).send(x);
}

int64_t bl_link_recv(int64_t h, at::Tensor out) {
  TORCH_CHECK(out.is_cuda() && out.is_contiguous(), "link recv: a contiguous CUDA tensor");
  return link(h).recv(out);
}

void bl_link_take(int64_t h, int64_t ticket) {
  link(h).take(ticket);
}
