// Shared device helpers: counter-based RNG, scale codecs, packed code access, lattice parameters.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace bl {

constexpr int kEPT = 8;            // elements per thread in element-wise kernels
constexpr int kMaxLevels = 256;

enum Kind : int { kTable = 0, kUniform = 1 };
enum ScaleDtype : int { kF32 = 0, kF16 = 1, kBF16 = 2, kE4M3 = 3, kE8M0 = 4 };
enum Rounding : int { kSR = 0, kStrat = 1, kRTN = 2 };
enum ScaleRule : int { kFixed = 0, kFitRepresented = 1, kFitLatent = 2 };

// RNG: identical to bitlattice.reference.hash_uniform
__device__ __forceinline__ float hash_uniform(uint64_t idx, uint32_t seed, uint32_t step) {
  uint32_t key = seed * 0x9E3779B1u + step * 0x85EBCA77u;
  uint32_t x = static_cast<uint32_t>(idx) * 0x27D4EB2Fu + key;
  x = (x ^ (x >> 16)) * 0x7FEB352Du;
  x = (x ^ (x >> 15)) * 0x846CA68Bu;
  x ^= x >> 16;
  return static_cast<float>(x >> 8) * 5.9604644775390625e-08f;  // 2^-24
}

struct RoundParams {
  int mode;        // Rounding
  uint32_t seed;
  uint32_t step;
  int strat_j;     // bit-reversed step for stratified rounding
  int strat_k;     // number of strata (power of two)
  uint64_t offset; // index of the first element (row tiles of a larger tensor)
};

__device__ __forceinline__ float rounding_u(const RoundParams& rp, uint64_t idx) {
  idx += rp.offset;
  if (rp.mode == kSR) return hash_uniform(idx, rp.seed, rp.step);
  if (rp.mode == kStrat) {
    float v = (static_cast<float>(rp.strat_j) + hash_uniform(idx, rp.seed, 0u)) / static_cast<float>(rp.strat_k);
    return v - floorf(v);
  }
  return 0.5f;
}

// scale codecs; scales are addressed as raw bytes of their storage dtype
__device__ __forceinline__ float decode_scale(const void* p, int64_t i, int dt) {
  switch (dt) {
    case kF32: return static_cast<const float*>(p)[i];
    case kF16: return __half2float(static_cast<const __half*>(p)[i]);
    case kBF16: return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
    case kE4M3: {
      __nv_fp8_e4m3 v;
      v.__x = static_cast<const __nv_fp8_storage_t*>(p)[i];
      return static_cast<float>(v);
    }
    default: return exp2f(static_cast<float>(static_cast<const uint8_t*>(p)[i]) - 127.0f);
  }
}

// Encode a (global-normalised) scale and return the decoded value; ``out`` receives the storage bits.
__device__ __forceinline__ float encode_scale(float s, int dt, uint32_t* out) {
  switch (dt) {
    case kF32: {
      float v = fmaxf(s, 1e-30f);
      *out = __float_as_uint(v);
      return v;
    }
    case kF16: {
      __half h = __float2half_rn(fminf(fmaxf(s, 5.9604644775390625e-08f), 65504.0f));
      *out = __half_as_ushort(h);
      return __half2float(h);
    }
    case kBF16: {
      __nv_bfloat16 h = __float2bfloat16_rn(fmaxf(s, 1e-30f));
      *out = __bfloat16_as_ushort(h);
      return __bfloat162float(h);
    }
    case kE4M3: {
      __nv_fp8_e4m3 v(fminf(fmaxf(s, 0.001953125f), 448.0f));
      *out = v.__x;
      return static_cast<float>(v);
    }
    default: {
      float e = rintf(log2f(fmaxf(s, 1.1754943508222875e-38f))) + 127.0f;
      e = fminf(fmaxf(e, 0.0f), 254.0f);
      *out = static_cast<uint32_t>(e);
      return exp2f(e - 127.0f);
    }
  }
}

__device__ __forceinline__ void store_scale(void* p, int64_t i, int dt, uint32_t bits) {
  switch (dt) {
    case kF32: static_cast<uint32_t*>(p)[i] = bits; break;
    case kF16:
    case kBF16: static_cast<uint16_t*>(p)[i] = static_cast<uint16_t>(bits); break;
    default: static_cast<uint8_t*>(p)[i] = static_cast<uint8_t>(bits); break;
  }
}

// packed access for a chunk of kEPT consecutive elements starting at a multiple of kEPT
// container bits cb in {1,2,4,8,16,32}; returns raw (unsigned) patterns
__device__ __forceinline__ void load_codes(const uint8_t* base, int64_t chunk, int cb, uint32_t out[kEPT]) {
  if (cb <= 8) {
    const uint8_t* p = base + chunk * cb;          // kEPT*cb/8 == cb bytes per chunk
    uint64_t v = 0;
    switch (cb) {
      case 1: v = *p; break;
      case 2: v = *reinterpret_cast<const uint16_t*>(p); break;
      case 4: v = *reinterpret_cast<const uint32_t*>(p); break;
      default: v = *reinterpret_cast<const uint64_t*>(p); break;
    }
    const uint32_t mask = (1u << cb) - 1u;
#pragma unroll
    for (int i = 0; i < kEPT; ++i) out[i] = static_cast<uint32_t>(v >> (i * cb)) & mask;
  } else if (cb == 16) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(base) + chunk * kEPT;
#pragma unroll
    for (int i = 0; i < kEPT; ++i) out[i] = p[i];
  } else {
    const uint32_t* p = reinterpret_cast<const uint32_t*>(base) + chunk * kEPT;
#pragma unroll
    for (int i = 0; i < kEPT; ++i) out[i] = p[i];
  }
}

__device__ __forceinline__ void store_codes(uint8_t* base, int64_t chunk, int cb, const uint32_t in[kEPT]) {
  if (cb <= 8) {
    uint64_t v = 0;
    const uint64_t mask = (1ull << cb) - 1ull;
#pragma unroll
    for (int i = 0; i < kEPT; ++i) v |= (static_cast<uint64_t>(in[i]) & mask) << (i * cb);
    uint8_t* p = base + chunk * cb;
    switch (cb) {
      case 1: *p = static_cast<uint8_t>(v); break;
      case 2: *reinterpret_cast<uint16_t*>(p) = static_cast<uint16_t>(v); break;
      case 4: *reinterpret_cast<uint32_t*>(p) = static_cast<uint32_t>(v); break;
      default: *reinterpret_cast<uint64_t*>(p) = v; break;
    }
  } else if (cb == 16) {
    uint16_t* p = reinterpret_cast<uint16_t*>(base) + chunk * kEPT;
#pragma unroll
    for (int i = 0; i < kEPT; ++i) p[i] = static_cast<uint16_t>(in[i]);
  } else {
    uint32_t* p = reinterpret_cast<uint32_t*>(base) + chunk * kEPT;
#pragma unroll
    for (int i = 0; i < kEPT; ++i) p[i] = in[i];
  }
}

__device__ __forceinline__ int sign_extend(uint32_t raw, int bits) {
  if (bits >= 32) return static_cast<int>(raw);
  raw &= (1u << bits) - 1u;
  return (raw >> (bits - 1)) ? static_cast<int>(raw) - (1 << bits) : static_cast<int>(raw);
}

// lattice description passed by value to kernels
struct Lattice {
  int kind;            // Kind
  int bits;            // logical code bits
  int cb;              // code container bits
  int qmin, qmax;      // uniform grid
  int zero_point;      // uniform with per-group zero
  int levels;          // number of levels (table)
  int scale_dt;        // ScaleDtype
  int64_t group;       // elements per scale group along the flattened tensor (>0), or -1 (row), 0 (tensor)
  int64_t rows, cols;
  const float* sorted;       // [levels]
  const int* rank_of_code;   // [2^bits]
  const int* sorted_code;    // [levels]
  const void* scales_in;
  void* scales_out;
  const float* global_scale; // nullable
  const __half* zeros;       // nullable
};

__device__ __forceinline__ int64_t group_of(const Lattice& L, int64_t e) {
  if (L.group == 0) return 0;
  if (L.group == -1) return e / L.cols;
  return e / L.group;  // groups tile rows exactly (cols % group == 0)
}

__device__ __forceinline__ int num_ranks(const Lattice& L) {
  return L.kind == kTable ? L.levels : L.qmax - L.qmin + 1;
}

__device__ __forceinline__ int raw_to_rank(const Lattice& L, const int* s_rank, uint32_t raw) {
  if (L.kind == kTable) return max(s_rank[raw], 0);
  const int k = L.zero_point ? static_cast<int>(raw) : sign_extend(raw, L.bits);
  return k - L.qmin;
}

__device__ __forceinline__ uint32_t rank_to_raw(const Lattice& L, const int* s_code, int rank) {
  if (L.kind == kTable) return static_cast<uint32_t>(s_code[rank]);
  const int k = rank + L.qmin;
  if (L.zero_point) return static_cast<uint32_t>(k);
  return L.bits >= 32 ? static_cast<uint32_t>(k) : static_cast<uint32_t>(k) & ((1u << L.bits) - 1u);
}

__device__ __forceinline__ float rank_level(const Lattice& L, const float* s_sorted, int rank, float zero) {
  if (L.kind == kTable) return s_sorted[rank];
  return static_cast<float>(rank + L.qmin) - zero;
}

// nearest rank; ties to the lower level (count of midpoints strictly below x), same as torch.searchsorted
__device__ __forceinline__ int nearest_rank(const Lattice& L, const float* s_sorted, float x, float zero) {
  if (L.kind == kTable) {
    int lo = 0, hi = L.levels - 1;   // search count of mids < x over mids[0..levels-2]
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      const float m = (s_sorted[mid + 1] + s_sorted[mid]) * 0.5f;
      if (m < x) lo = mid + 1; else hi = mid;
    }
    return lo;
  }
  float k = rintf(x + zero);
  k = fminf(fmaxf(k, static_cast<float>(L.qmin)), static_cast<float>(L.qmax));
  return static_cast<int>(k) - L.qmin;
}

// phase packed as two's complement pb-bit values inside pcb-bit containers
__device__ __forceinline__ void load_phase(const uint8_t* base, int64_t chunk, int pb, int pcb, int out[kEPT]) {
  if (pb == 0) {
#pragma unroll
    for (int i = 0; i < kEPT; ++i) out[i] = 0;
    return;
  }
  uint32_t raw[kEPT];
  load_codes(base, chunk, pcb, raw);
#pragma unroll
  for (int i = 0; i < kEPT; ++i) out[i] = sign_extend(raw[i], pb);
}

__device__ __forceinline__ void store_phase(uint8_t* base, int64_t chunk, int pb, int pcb, const int in[kEPT]) {
  if (pb == 0) return;
  uint32_t raw[kEPT];
#pragma unroll
  for (int i = 0; i < kEPT; ++i) raw[i] = static_cast<uint32_t>(in[i]) & ((1u << pb) - 1u);
  store_codes(base, chunk, pcb, raw);
}

// latent position from phase code c in [-half, half)
__device__ __forceinline__ float latent(float w, float lo, float hi, int code, int pb) {
  if (pb == 0) return w;
  const float c = static_cast<float>(code) / static_cast<float>(1 << (pb - 1));
  return c < 0.0f ? w + c * (w - lo) * 0.5f : w + c * (hi - w) * 0.5f;
}

// projection of target t around nearest value w with neighbours lo/hi; returns move (-1,0,1), writes phase code
template <typename T>
__device__ __forceinline__ int project(T t, T w, T lo, T hi, T u, int pb, int* code) {
  if (pb == 0) {
    const bool up = t >= w;
    const T a = up ? w : lo;
    const T b = up ? hi : w;
    const T gap = b - a;
    const T p = gap > T(0) ? (t - a) / gap : T(0);
    const bool take_b = u < p;
    *code = 0;
    return up ? (take_b ? 1 : 0) : (take_b ? 0 : -1);
  }
  const T half = static_cast<T>(1 << (pb - 1));
  const bool left = t < w;
  const T width = left ? (w - lo) * T(0.5) : (hi - w) * T(0.5);
  T c = width > T(0) ? (t - w) / width : T(0);
  c = c < T(-1) ? T(-1) : (c > T(1) ? T(1) : c);
  const T q = c * half;
  const T fl = floor(q);
  T k = fl + ((u < (q - fl)) ? T(1) : T(0));
  int move = 0;
  if (k >= half) {
    if (hi > w) { move = 1; k = -half; } else { k = half - T(1); }
  }
  *code = static_cast<int>(k);
  return move;
}

}  // namespace bl
