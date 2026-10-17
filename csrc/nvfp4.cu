// NVFP4 quantization kernels for the FP4 GEMM paths and caches (reference: bitlattice/fp4ref.py).
//
// One configurable pipeline per 16-element block:
//   [rotation]  randomized Hadamard transform over 128-element chunks (8 lanes x 16 values, warp shuffles)
//   scale       block amax / 6 (or / 4: four over six, choosing the lower MSE or MAE of the round-to-nearest codes)
//   codes       round to nearest (ties to even) | stochastic rounding | round to nearest with MS-EDEN scale correction
//   MS-EDEN     S = <x, x> / <x, x_rtn> per 128 elements; every block scale times S, stochastically rounded to E4M3
// Layouts: rows of a [rows, cols] tensor along cols ("row"), or rows of its transpose ("col", via a shared-memory
// tile); the input may be bf16 / fp16 / fp32 or an NVFP4 tensor given by its codes and scales (re-quantization of
// a transposed weight). The global scale is either given or computed by a first pass over the (rotated) input.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <optional>
#include <type_traits>
#include <vector>

#include "common.cuh"

// architecture-specific targets with the E2M1 conversion instructions (sm_100a / sm_120a and relatives)
#if defined(__CUDA_ARCH__) &&                                                                                   \
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL) ||                              \
     defined(__CUDA_ARCH_FEAT_SM103_ALL) || defined(__CUDA_ARCH_FEAT_SM110_ALL) ||                              \
     defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define BL_HW_E2M1 1
#endif

namespace {

constexpr int kRTN = 0, kSR = 1, kEDEN = 2;

__device__ __forceinline__ float e2m1_level(uint32_t m) {           // magnitude of code m in 0..7
  return m < 4 ? 0.5f * static_cast<float>(m) : (m == 4 ? 2.0f : m == 5 ? 3.0f : m == 6 ? 4.0f : 6.0f);
}

struct Opts {
  int mode;          // kRTN | kSR | kEDEN
  bool fos;          // four over six
  bool mae;          // four-over-six selection by mean absolute error (else squared error)
  bool rotate;       // randomized Hadamard over 128-element chunks
  uint32_t seed;     // stochastic draws
  uint32_t rseed;    // rotation signs (both operands of a GEMM use the same)
};

// nearest E2M1 level, ties to the even code (as the hardware conversion); |v| > 6 saturates
__device__ __forceinline__ uint32_t code_rtn(float v) {
  const float a = fabsf(v);
  const uint32_t m = (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.0f);
  return (v < 0.0f && m ? 8u : 0u) | m;
}

__device__ __forceinline__ uint32_t code_sr(float v, float u) {
  const float a = fminf(fabsf(v), 6.0f);
  uint32_t m = (a >= 0.5f) + (a >= 1.0f) + (a >= 1.5f) + (a >= 2.0f) + (a >= 3.0f) + (a >= 4.0f) + (a >= 6.0f);
  if (m < 7) m += u * (e2m1_level(m + 1) - e2m1_level(m)) < a - e2m1_level(m);
  return (v < 0.0f && m ? 8u : 0u) | m;
}

__device__ __forceinline__ float level(uint32_t code) {
  const float m = e2m1_level(code & 7u);
  return code & 8u ? -m : m;
}

__device__ __forceinline__ float e4m3_round(float v) {
  return static_cast<float>(__nv_fp8_e4m3(fminf(fmaxf(v, 0.001953125f), 448.0f)));
}

// stochastic rounding of a positive value to E4M3 (clamped to its range)
__device__ __forceinline__ uint8_t e4m3_sr(float v, float u) {
  v = fminf(fmaxf(v, 0.001953125f), 448.0f);
  __nv_fp8_e4m3 r(v);
  uint32_t bits = r.__x;
  __nv_fp8_e4m3 t;
  t.__x = bits;
  float lo = static_cast<float>(t);
  if (lo > v) {
    bits = bits > 1 ? bits - 1 : 1;
    t.__x = bits;
    lo = static_cast<float>(t);
  }
  const uint32_t hb = bits < 0x7E ? bits + 1 : 0x7E;
  t.__x = hb;
  const float hi = static_cast<float>(t);
  const float p = hi > lo ? (v - lo) / (hi - lo) : 0.0f;
  return static_cast<uint8_t>(u < p ? hb : bits);
}

// in-register 16-point FWHT, then 3 butterfly stages across the 8 lanes holding one 128-element chunk; the lanes of
// a chunk are 8 consecutive lanes of a warp
__device__ __forceinline__ void hadamard128(float v[16], int lane8) {
#pragma unroll
  for (int h = 1; h < 16; h <<= 1) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if ((i & h) == 0) {
        const float a = v[i], b = v[i + h];
        v[i] = a + b;
        v[i + h] = a - b;
      }
    }
  }
#pragma unroll
  for (int m = 1; m < 8; m <<= 1) {
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float p = __shfl_xor_sync(0xffffffffu, v[i], m);
      v[i] = (lane8 & m) ? p - v[i] : v[i] + p;
    }
  }
  constexpr float inv = 0.08838834764831845f;           // 1/sqrt(128)
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] *= inv;
}

__device__ __forceinline__ void rotate_block(float v[16], int lane8, uint32_t seed) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float u = bl::hash_uniform(static_cast<uint64_t>(lane8 * 16 + i), seed, 0x5151u);
    v[i] = u < 0.5f ? -v[i] : v[i];
  }
  hadamard128(v, lane8);
}

__device__ __forceinline__ float lane8_sum(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  x += __shfl_xor_sync(0xffffffffu, x, 4);
  return x;
}

// quantize one block: returns packed codes, writes the scale byte. ``elem0`` / ``block`` index the stochastic draws.
// round-to-nearest codes of v * inv (16 values), and the dequantized values of codes c times s
__device__ __forceinline__ void codes16(const float v[16], float inv, uint32_t c[16]) {
#ifdef BL_HW_E2M1
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    uint32_t r;
    asm("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u32.u8 %0, t;\n}"
        : "=r"(r) : "f"(v[2 * k + 1] * inv), "f"(v[2 * k] * inv));
    c[2 * k] = r & 15u;
    c[2 * k + 1] = r >> 4;
  }
#else
#pragma unroll
  for (int i = 0; i < 16; ++i) c[i] = code_rtn(v[i] * inv);
#endif
}

__device__ __forceinline__ void values16(const uint32_t c[16], float s, float out[16]) {
#ifdef BL_HW_E2M1
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    uint32_t h2;
    const uint16_t b = static_cast<uint16_t>(c[2 * k] | (c[2 * k + 1] << 4));
    asm("{\n .reg .b8 t;\n cvt.u8.u16 t, %1;\n cvt.rn.f16x2.e2m1x2 %0, t;\n}" : "=r"(h2) : "h"(b));
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
    out[2 * k] = f.x * s;
    out[2 * k + 1] = f.y * s;
  }
#else
#pragma unroll
  for (int i = 0; i < 16; ++i) out[i] = level(c[i]) * s;
#endif
}

// quantize one block: returns packed codes, writes the scale byte. ``elem0`` / ``block`` index the stochastic draws.
__device__ __forceinline__ uint64_t quantize16(const float v[16], float g, const Opts& o, uint64_t elem0,
                                               uint64_t block, int lane8, uint8_t* scale_out) {
  // one division per block; elements are scaled by the reciprocal (the reference does the same)
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  float s = e4m3_round(amax / 6.0f / g) * g;
  float inv = s > 0.0f ? 1.0f / s : 0.0f;
  uint32_t c[16];
  codes16(v, inv, c);
  if (o.fos) {
    const float s4 = e4m3_round(amax / 4.0f / g) * g;
    const float inv4 = s4 > 0.0f ? 1.0f / s4 : 0.0f;
    uint32_t c4[16];
    codes16(v, inv4, c4);
    float d6[16], d4[16];
    values16(c, s, d6);
    values16(c4, s4, d4);
    float e6 = 0.0f, e4 = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float a6 = d6[i] - v[i], a4 = d4[i] - v[i];
      e6 += o.mae ? fabsf(a6) : a6 * a6;
      e4 += o.mae ? fabsf(a4) : a4 * a4;
    }
    if (e4 < e6) {
      s = s4;
      inv = inv4;
#pragma unroll
      for (int i = 0; i < 16; ++i) c[i] = c4[i];
    }
  }
  uint8_t s8 = __nv_fp8_e4m3(s / g).__x;
  if (o.mode == kSR) {
#pragma unroll
    for (int i = 0; i < 16; ++i) c[i] = code_sr(v[i] * inv, bl::hash_uniform(elem0 + i, o.seed, 0u));
  } else if (o.mode == kEDEN) {
    float d[16];
    values16(c, s, d);
    float xx = 0.0f, xr = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      xx += v[i] * v[i];
      xr += v[i] * d[i];
    }
    xx = lane8_sum(xx);
    xr = lane8_sum(xr);
    const float S = xr != 0.0f ? xx / xr : 1.0f;
    s8 = e4m3_sr(s / g * S, bl::hash_uniform(block, o.seed ^ 0x9E3779B9u, 0u));
  }
  uint64_t out = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) out |= static_cast<uint64_t>(c[i]) << (4 * i);
  *scale_out = s8;
  return out;
}

template <typename T> __device__ __forceinline__ float tof(T v);
template <> __device__ __forceinline__ float tof<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float tof<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float tof<float>(float v) { return v; }
template <> __device__ __forceinline__ float tof<uint8_t>(uint8_t) { return 0.0f; }   // NVFP4 input: col layout only

template <typename T>
__device__ __forceinline__ void load_row16(const T* p, float v[16]) {
  constexpr int per = 16 / sizeof(T);
  const uint4* q = reinterpret_cast<const uint4*>(p);
#pragma unroll
  for (int j = 0; j < 16 / per; ++j) {
    const uint4 raw = q[j];
    const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
    for (int i = 0; i < per; ++i) v[j * per + i] = tof<T>(e[i]);
  }
}

__device__ __forceinline__ void atomic_max_pos(float* addr, float v) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(v));
}

// Block scales [rows, cols] row-major, or directly in the 128x4-tiled layout of block-scaled GEMMs (``nc``: columns
// padded to a multiple of 4, divided by 4); see bitlattice.backend.to_blocked
struct ScaleOut {
  uint8_t* p;
  int64_t cols;
  int64_t nc;
  bool blocked;
  __device__ __forceinline__ void put(int64_t r, int64_t c, uint8_t v) const {
    if (blocked) p[((r >> 7) * nc + (c >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (c & 3)] = v;
    else p[r * cols + c] = v;
  }
};

// row layout: blocks along the last dimension
// thread b handles block b; 8 consecutive threads hold one 128-element chunk (cols % 128 == 0 when rotating or EDEN)

template <typename T, bool kAmax>
__global__ void row_kernel(const T* x, int64_t nblocks, int64_t cols, Opts o, float* amax, const float* gptr,
                           uint8_t* packed, ScaleOut scales) {
  const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool live = b < nblocks;
  const int lane8 = static_cast<int>(b & 7);
  float v[16];
  if (live) load_row16(x + b * 16, v);
  else {
#pragma unroll
    for (int i = 0; i < 16; ++i) v[i] = 0.0f;
  }
  if (o.rotate) rotate_block(v, lane8, o.rseed);
  if (kAmax) {
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(v[i]));
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
    if ((threadIdx.x & 31) == 0) atomic_max_pos(amax, m);
    return;
  }
  const float g = *gptr;
  uint8_t s8;
  const uint64_t out = quantize16(v, g, o, static_cast<uint64_t>(b) * 16, static_cast<uint64_t>(b), lane8, &s8);
  if (live) {
    reinterpret_cast<uint64_t*>(packed)[b] = out;
    scales.put(b / (cols / 16), b % (cols / 16), s8);
  }
}

// column layout: blocks along the first dimension
// A CTA of 256 threads takes a tile of 128 rows x 32 columns of the input (coalesced loads into shared memory);
// thread t then owns column t / 8 and the 16 rows (t % 8) * 16 .. + 15, so 8 consecutive lanes hold one 128-element
// chunk of a row of the transpose. Output row = input column.

struct Fp4Input {                                     // NVFP4 tensor [rows, cols]: codes along cols
  const uint8_t* packed;
  const uint8_t* scales;
  const float* g;
};

template <typename T>
__device__ __forceinline__ float load_elem(const T* x, int64_t cols, int64_t r, int64_t c, Fp4Input) {
  return tof<T>(x[r * cols + c]);
}

template <>
__device__ __forceinline__ float load_elem<uint8_t>(const uint8_t*, int64_t cols, int64_t r, int64_t c, Fp4Input w) {
  const uint32_t code = (w.packed[r * (cols / 2) + c / 2] >> (4 * (c & 1))) & 0xFu;
  __nv_fp8_e4m3 s8;
  s8.__x = w.scales[r * (cols / 16) + c / 16];
  return level(code) * static_cast<float>(s8) * (*w.g);
}

template <typename T, bool kAmax>
__global__ void col_kernel(const T* x, int64_t rows, int64_t cols, Fp4Input w, Opts o, float* amax,
                           const float* gptr, uint8_t* packed, ScaleOut scales) {
  __shared__ float tile[128][33];
  const int t = threadIdx.x;
  const int64_t r0 = static_cast<int64_t>(blockIdx.y) * 128, c0 = static_cast<int64_t>(blockIdx.x) * 32;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const int idx = i * 256 + t, rr = idx >> 5, cc = idx & 31;
    const int64_t r = r0 + rr, c = c0 + cc;
    tile[rr][cc] = (r < rows && c < cols) ? load_elem<T>(x, cols, r, c, w) : 0.0f;
  }
  __syncthreads();
  const int col = t >> 3, lane8 = t & 7;
  float v[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = tile[lane8 * 16 + i][col];
  if (o.rotate) rotate_block(v, lane8, o.rseed);
  const int64_t oc = c0 + col;                          // output row
  if (kAmax) {
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) m = fmaxf(m, fabsf(v[i]));
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
    if ((t & 31) == 0) atomic_max_pos(amax, m);
    return;
  }
  const float g = *gptr;
  const int64_t ob = r0 / 16 + lane8;                    // block index inside the output row
  const int64_t nb = rows / 16;
  uint8_t s8;
  const uint64_t out = quantize16(v, g, o, static_cast<uint64_t>(oc * rows + ob * 16),
                                  static_cast<uint64_t>(oc * nb + ob), lane8, &s8);
  if (oc < cols) {
    reinterpret_cast<uint64_t*>(packed + oc * (rows / 2))[ob] = out;
    scales.put(oc, ob, s8);
  }
}

// both layouts from one read (cast-transpose)
// A CTA of 256 threads stages a 128 x 128 tile of x [M, K] in shared memory and quantizes its rows (operand rows of
// x, blocks along K) and its columns (operand rows of x^T, blocks along M), each side with its own options. As in
// the other kernels a thread owns one 16-element block and 8 consecutive lanes one 128-element chunk; the rotation
// signs of a lane are computed once. Element (r, c) of the tile lives at r * 128 + (c ^ swz(r)) with swz(r) =
// ((r / 16) % 8) * (elements per 4 bank words): 16-byte row reads stay contiguous and the column reads of a warp (4
// columns x 8 row groups) hit distinct banks.

struct Side {
  bool on;
  int mode;
  bool fos;
  bool rotate;
  uint32_t seed, rseed;
  bool pow2;         // global scale rounded up to a power of two (operands that are later concatenated exactly)
};

template <typename T> struct Stage { using type = T; static constexpr int kSwz = 3; };
template <> struct Stage<float> { using type = float; static constexpr int kSwz = 2; };

template <int kSwz> __device__ __forceinline__ int swz(int r) { return ((r >> 4) & 7) << kSwz; }

__device__ __forceinline__ uint32_t sign_mask(int lane8, uint32_t rseed) {
  uint32_t m = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i)
    m |= (bl::hash_uniform(static_cast<uint64_t>(lane8 * 16 + i), rseed, 0x5151u) < 0.5f ? 1u : 0u) << i;
  return m;
}

__device__ __forceinline__ void rotate_masked(float v[16], int lane8, uint32_t mask) {
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = (mask >> i) & 1u ? -v[i] : v[i];
  hadamard128(v, lane8);
}

__device__ __forceinline__ float global_of(const float* amax, const float* given, int side, bool fos, bool pow2) {
  if (given) return given[side];
  const float g = fmaxf(amax[side] / (6.0f * (fos ? 256.0f : 448.0f)), 1e-12f);
  return pow2 ? exp2f(ceilf(log2f(g))) : g;
}

template <typename T, bool kAmax>
__global__ void __launch_bounds__(256) dual_kernel(const T* __restrict__ x, int64_t M, int64_t K, Side row, Side col,
                                                   bool mae, float* amax, const float* given, float* g_out,
                                                   uint8_t* rp, ScaleOut rs, uint8_t* cp, ScaleOut cs,
                                                   float* amax_out) {
  using E = typename Stage<T>::type;
  constexpr int kSwz = Stage<T>::kSwz;
  constexpr int kPer = 16 / sizeof(E);                       // elements per 16-byte vector
  extern __shared__ __align__(16) unsigned char smem_raw[];
  E* tile = reinterpret_cast<E*>(smem_raw);
  const int64_t r0 = static_cast<int64_t>(blockIdx.y) * 128, k0 = static_cast<int64_t>(blockIdx.x) * 128;
  for (int i = threadIdx.x; i < 128 * 128 / kPer; i += 256) {
    const int r = i / (128 / kPer), c = (i % (128 / kPer)) * kPer;
    *reinterpret_cast<uint4*>(tile + r * 128 + (c ^ swz<kSwz>(r))) =
        *reinterpret_cast<const uint4*>(x + (r0 + r) * K + k0 + c);
  }
  const int lane8 = threadIdx.x & 7, sub = threadIdx.x >> 3;   // sub: 0..31
  const uint32_t rmask = row.on && row.rotate ? sign_mask(lane8, row.rseed) : 0u;
  const uint32_t cmask = col.on && col.rotate ? sign_mask(lane8, col.rseed) : 0u;
  float gr = 0.0f, gc = 0.0f;
  Opts ro{row.mode, row.fos, mae, false, row.seed, row.rseed}, co{col.mode, col.fos, mae, false, col.seed, col.rseed};
  if (!kAmax) {
    if (row.on) gr = global_of(amax, given, 0, row.fos, row.pow2);
    if (col.on) gc = global_of(amax, given, 1, col.fos, col.pow2);
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
      g_out[0] = gr;
      g_out[1] = gc;
    }
  }
  __syncthreads();
  float mr = 0.0f, mc = 0.0f;
  float v[16];
  if (row.on) {
#pragma unroll 1
    for (int k = 0; k < 4; ++k) {
      const int r = sub + 32 * k;
      const E* src = tile + r * 128;
      const int sw = swz<kSwz>(r);
#pragma unroll
      for (int h = 0; h < 16 / kPer; ++h) {
        const uint4 raw = *reinterpret_cast<const uint4*>(src + ((lane8 * 16 + h * kPer) ^ sw));
        const E* e = reinterpret_cast<const E*>(&raw);
#pragma unroll
        for (int t = 0; t < kPer; ++t) v[h * kPer + t] = tof<E>(e[t]);
      }
      if (row.rotate) rotate_masked(v, lane8, rmask);
      const int64_t orow = r0 + r;
#pragma unroll
      for (int i = 0; i < 16; ++i) mr = fmaxf(mr, fabsf(v[i]));
      if (!kAmax) {
        const int64_t b = orow * (K / 16) + k0 / 16 + lane8;
        uint8_t s8;
        const uint64_t out = quantize16(v, gr, ro, static_cast<uint64_t>(b) * 16, static_cast<uint64_t>(b), lane8,
                                        &s8);
        reinterpret_cast<uint64_t*>(rp)[b] = out;
        rs.put(orow, k0 / 16 + lane8, s8);
      }
    }
  }
  if (col.on) {
#pragma unroll 1
    for (int k = 0; k < 4; ++k) {
      const int c = sub + 32 * k;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const int r = lane8 * 16 + i;
        v[i] = tof<E>(tile[r * 128 + (c ^ swz<kSwz>(r))]);
      }
      if (col.rotate) rotate_masked(v, lane8, cmask);
      const int64_t ocol = k0 + c;
#pragma unroll
      for (int i = 0; i < 16; ++i) mc = fmaxf(mc, fabsf(v[i]));
      if (!kAmax) {
        const int64_t b = ocol * (M / 16) + r0 / 16 + lane8;
        uint8_t s8;
        const uint64_t out = quantize16(v, gc, co, static_cast<uint64_t>(b) * 16, static_cast<uint64_t>(b), lane8,
                                        &s8);
        reinterpret_cast<uint64_t*>(cp)[b] = out;
        cs.put(ocol, r0 / 16 + lane8, s8);
      }
    }
  }
  // the amax pass reduces into ``amax``; the quantization pass reports the amax it saw into ``amax_out`` (delayed
  // scaling: the next call's global scales)
  float* dst = kAmax ? amax : amax_out;
  if (dst) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
      mr = fmaxf(mr, __shfl_xor_sync(0xffffffffu, mr, m));
      mc = fmaxf(mc, __shfl_xor_sync(0xffffffffu, mc, m));
    }
    if ((threadIdx.x & 31) == 0) {
      if (row.on) atomic_max_pos(dst, mr);
      if (col.on) atomic_max_pos(dst + 1, mc);
    }
  }
}

}  // namespace

namespace {
// scale buffer for an operand [rows, cols / 16]: row-major, or tiled (zero padding to 128 rows x 4 columns)
std::pair<at::Tensor, ScaleOut> scale_buffer(int64_t rows, int64_t scols, bool blocked, const at::TensorOptions& opt) {
  if (!blocked) {
    at::Tensor t = at::empty({rows, scols}, opt);
    return {t, ScaleOut{t.data_ptr<uint8_t>(), scols, 0, false}};
  }
  const int64_t nr = (rows + 127) / 128, nc = (scols + 3) / 4;
  const bool pad = nr * 128 != rows || nc * 4 != scols;
  at::Tensor t = pad ? at::zeros({nr * 128 * nc * 4}, opt) : at::empty({nr * 128 * nc * 4}, opt);
  return {t, ScaleOut{t.data_ptr<uint8_t>(), scols, nc, true}};
}
}  // namespace

// x: [rows, cols] bf16 / fp16 / fp32, or uint8 NVFP4 codes [rows, cols/2] with ``w_scales`` / ``w_global`` (col
// layout only). transpose: quantize rows of x^T. mode: 0 round to nearest, 1 stochastic, 2 MS-EDEN.
// global_scale: given, or computed as amax / (6 * (four_over_six ? 256 : 448)) of the (rotated) input.
// rotate_seed: seed of the rotation signs (-1: ``seed``).
std::tuple<at::Tensor, at::Tensor, at::Tensor> bl_nvfp4_quantize(at::Tensor x, bool transpose, int64_t mode,
                                                                 bool four_over_six, bool mae, bool rotate,
                                                                 int64_t seed, std::optional<at::Tensor> global_scale,
                                                                 std::optional<at::Tensor> w_scales,
                                                                 std::optional<at::Tensor> w_global,
                                                                 int64_t rotate_seed, bool blocked, bool pow2_global) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2, "x must be a contiguous 2-D CUDA tensor");
  const c10::cuda::CUDAGuard guard(x.device());
  const bool fp4_in = x.scalar_type() == at::kByte;
  TORCH_CHECK(!fp4_in || (transpose && w_scales.has_value() && w_global.has_value()),
              "an NVFP4 input needs transpose and its scales");
  const int64_t rows = x.size(0), cols = fp4_in ? x.size(1) * 2 : x.size(1);
  const int64_t out_rows = transpose ? cols : rows, out_cols = transpose ? rows : cols;
  const int64_t chunk = (rotate || mode == kEDEN) ? 128 : 16;
  TORCH_CHECK(out_cols % chunk == 0, "the quantized dimension must be a multiple of ", chunk);
  Opts o{static_cast<int>(mode), four_over_six, mae, rotate, static_cast<uint32_t>(seed),
         static_cast<uint32_t>(rotate_seed < 0 ? seed : rotate_seed)};
  auto opt = x.options();
  at::Tensor packed = at::empty({out_rows, out_cols / 2}, opt.dtype(at::kByte));
  auto [scales, sout] = scale_buffer(out_rows, out_cols / 16, blocked, opt.dtype(at::kByte));
  at::Tensor amax = at::zeros({1}, opt.dtype(at::kFloat));
  at::Tensor global;
  const bool given = global_scale.has_value();
  auto stream = at::cuda::getCurrentCUDAStream();
  Fp4Input w{nullptr, nullptr, nullptr};
  at::Tensor wg;
  if (fp4_in) {
    wg = w_global->to(at::kFloat).contiguous();
    w = Fp4Input{x.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(w_scales->data_ptr()), wg.data_ptr<float>()};
  }
  const float top = 6.0f * (four_over_six ? 256.0f : 448.0f);
  auto finish_global = [&]() {
    global = given ? global_scale->to(at::kFloat).reshape({1}).contiguous() : (amax / top).clamp_min(1e-12);
    if (!given && pow2_global) global = at::exp2(at::ceil(at::log2(global)));
  };
  auto launch = [&](auto* ptr) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;
    if (!transpose) {
      const int64_t nb = rows * cols / 16;
      const int blocks = static_cast<int>((nb + 255) / 256);
      if (!given) row_kernel<T, true><<<blocks, 256, 0, stream>>>(ptr, nb, cols, o, amax.data_ptr<float>(), nullptr,
                                                                  nullptr, sout);
      finish_global();
      row_kernel<T, false><<<blocks, 256, 0, stream>>>(ptr, nb, cols, o, nullptr, global.data_ptr<float>(),
                                                       packed.data_ptr<uint8_t>(), sout);
    } else {
      TORCH_CHECK(rows % 128 == 0, "transposed quantization needs rows % 128 == 0 (pad first)");
      dim3 grid(static_cast<unsigned>((cols + 31) / 32), static_cast<unsigned>(rows / 128));
      if (!given) col_kernel<T, true><<<grid, 256, 0, stream>>>(ptr, rows, cols, w, o, amax.data_ptr<float>(),
                                                                nullptr, nullptr, sout);
      finish_global();
      col_kernel<T, false><<<grid, 256, 0, stream>>>(ptr, rows, cols, w, o, nullptr, global.data_ptr<float>(),
                                                     packed.data_ptr<uint8_t>(), sout);
    }
  };
  if (fp4_in) launch(x.data_ptr<uint8_t>());
  else if (x.scalar_type() == at::kBFloat16) launch(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
  else if (x.scalar_type() == at::kHalf) launch(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()));
  else {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be bf16, fp16, fp32 or NVFP4 codes");
    launch(x.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {packed, scales.view(at::kFloat8_e4m3fn), global};
}


// Both layouts of x [M, K] (M, K multiples of 128) from one read per pass. row / col = {on, mode, four_over_six,
// rotate, seed, rotate_seed}; a side that is off returns empty tensors. Global scales: given ([2]: row, col) or the
// amax of each (rotated) side / (6 * (four_over_six ? 256 : 448)), computed by a first pass over x. ``amax_out``
// (zeroed [2]): receives the amax of each (rotated) side as seen by the quantization pass.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> bl_nvfp4_quantize2(
    at::Tensor x, std::vector<int64_t> row, std::vector<int64_t> col, bool mae, std::optional<at::Tensor> global_scales,
    bool blocked, std::optional<at::Tensor> amax_out) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2, "x must be a contiguous 2-D CUDA tensor");
  TORCH_CHECK((row.size() == 6 || row.size() == 7) && (col.size() == 6 || col.size() == 7),
              "row / col: {on, mode, four_over_six, rotate, seed, rotate_seed[, pow2_global]}");
  const int64_t M = x.size(0), K = x.size(1);
  TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "both dimensions must be multiples of 128");
  TORCH_CHECK(!amax_out.has_value() || (amax_out->is_cuda() && amax_out->scalar_type() == at::kFloat &&
                                        amax_out->numel() == 2 && amax_out->is_contiguous()),
              "amax_out: zero-initialized float32 [2] (row, column)");
  const c10::cuda::CUDAGuard guard(x.device());
  auto side = [](const std::vector<int64_t>& v) {
    return Side{v[0] != 0, static_cast<int>(v[1]), v[2] != 0, v[3] != 0, static_cast<uint32_t>(v[4]),
                static_cast<uint32_t>(v[5] < 0 ? v[4] : v[5]), v.size() > 6 && v[6] != 0};
  };
  const Side rs = side(row), cs = side(col);
  auto opt = x.options().dtype(at::kByte);
  at::Tensor rp = at::empty({rs.on ? M : 0, K / 2}, opt), cp = at::empty({cs.on ? K : 0, M / 2}, opt);
  auto [rsc, rso] = scale_buffer(rs.on ? M : 0, K / 16, blocked, opt);
  auto [csc, cso] = scale_buffer(cs.on ? K : 0, M / 16, blocked, opt);
  at::Tensor g = at::empty({2}, x.options().dtype(at::kFloat));
  at::Tensor amax = at::zeros({2}, x.options().dtype(at::kFloat));
  at::Tensor given;
  if (global_scales.has_value()) given = global_scales->to(at::kFloat).reshape({2}).contiguous();
  const float* gp = global_scales.has_value() ? given.data_ptr<float>() : nullptr;
  dim3 grid(static_cast<unsigned>(K / 128), static_cast<unsigned>(M / 128));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto* ptr) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;
    const size_t smem = 128 * 128 * sizeof(typename Stage<T>::type);
    auto k_amax = dual_kernel<T, true>;
    auto k_quant = dual_kernel<T, false>;
    if (smem > 48 * 1024) {
      cudaFuncSetAttribute(k_amax, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
      cudaFuncSetAttribute(k_quant, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    }
    float* ao = amax_out.has_value() ? amax_out->data_ptr<float>() : nullptr;
    if (!gp) k_amax<<<grid, 256, smem, stream>>>(ptr, M, K, rs, cs, mae, amax.data_ptr<float>(), nullptr, nullptr,
                                                 nullptr, rso, nullptr, cso, nullptr);
    k_quant<<<grid, 256, smem, stream>>>(ptr, M, K, rs, cs, mae, amax.data_ptr<float>(), gp, g.data_ptr<float>(),
                                         rp.data_ptr<uint8_t>(), rso, cp.data_ptr<uint8_t>(), cso, ao);
  };
  if (x.scalar_type() == at::kBFloat16) launch(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
  else if (x.scalar_type() == at::kHalf) launch(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()));
  else {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be bf16, fp16 or fp32");
    launch(x.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {rp, rsc.view(at::kFloat8_e4m3fn), g.narrow(0, 0, 1), cp, csc.view(at::kFloat8_e4m3fn), g.narrow(0, 1, 1)};
}
