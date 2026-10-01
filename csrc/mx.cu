// MX quantization (OCP microscaling): blocks of 32 elements share a power-of-two (E8M0) scale, elements are FP8
// (E4M3 or E5M2). The scale is the block maximum divided by the largest element value, rounded up to a power of two
// (as in the NVIDIA MXFP8 training recipe: rounding down would clip the block maximum); elements round to nearest
// even with saturation. Both operand layouts of a GEMM come from one read of a 128 x 128 tile in shared memory
// (rows of x: blocks along K; rows of x^T: blocks along M); scales can be written in the tiled layout of
// block-scaled GEMMs.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <tuple>
#include <type_traits>

namespace {

template <typename T> __device__ __forceinline__ float tof(T v);
template <> __device__ __forceinline__ float tof<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <> __device__ __forceinline__ float tof<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float tof<float>(float v) { return v; }

struct MxOut {
  uint8_t* codes;     // [rows, cols]
  uint8_t* scales;    // [rows, cols / 32], or tiled
  int64_t cols;
  int64_t nc;         // tiled: ceil(cols / 32 / 4)
  bool blocked;
  __device__ __forceinline__ void put_scale(int64_t r, int64_t c, uint8_t v) const {
    if (blocked) scales[((r >> 7) * nc + (c >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (c & 3)] = v;
    else scales[r * (cols / 32) + c] = v;
  }
};

// one 32-element block held as 4 values per lane by 8 consecutive lanes; returns the E8M0 code, packs 4 codes
template <bool kE5M2>
__device__ __forceinline__ uint32_t quantize32(const float v[4], uint8_t* e8) {
  float m = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
  m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 4));
  constexpr float kMax = kE5M2 ? 57344.0f : 448.0f;
  int e = 127;                                                     // an all-zero block gets scale 1
  if (m > 0.0f) e = min(max(static_cast<int>(ceilf(log2f(m / kMax))) + 127, 0), 254);
  const float inv = exp2f(static_cast<float>(127 - e));
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint8_t c;
    if constexpr (kE5M2) c = __nv_fp8_e5m2(v[i] * inv).__x;
    else c = __nv_fp8_e4m3(v[i] * inv).__x;
    packed |= static_cast<uint32_t>(c) << (8 * i);
  }
  *e8 = static_cast<uint8_t>(e);
  return packed;
}

constexpr int kLd = 136;     // tile row stride (elements): 16-byte aligned rows, shifted banks

template <typename T, bool kE5M2>
__global__ void __launch_bounds__(256) mx_dual(const T* __restrict__ x, int64_t M, int64_t K, bool do_row,
                                               bool do_col, MxOut ro, MxOut co) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* tile = reinterpret_cast<T*>(smem_raw);
  const int64_t r0 = static_cast<int64_t>(blockIdx.y) * 128, k0 = static_cast<int64_t>(blockIdx.x) * 128;
  constexpr int kPer = 16 / sizeof(T);
  for (int i = threadIdx.x; i < 128 * 128 / kPer; i += 256) {
    const int r = i / (128 / kPer), c = (i % (128 / kPer)) * kPer;
    *reinterpret_cast<uint4*>(tile + r * kLd + c) = *reinterpret_cast<const uint4*>(x + (r0 + r) * K + k0 + c);
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int lane8 = lane & 7, grp = lane >> 3;
  // a warp quantizes 4 vectors at a time (one per group of 8 lanes), each as 4 blocks of 32; 16 vectors per warp
  for (int it = 0; it < 4; ++it) {
    const int vec = warp * 16 + it * 4 + grp;
    float v[4];
    uint8_t e8;
    if (do_row) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = tof<T>(tile[vec * kLd + b * 32 + lane8 * 4 + i]);
        const uint32_t packed = quantize32<kE5M2>(v, &e8);
        const int64_t row = r0 + vec, col = k0 + b * 32 + lane8 * 4;
        *reinterpret_cast<uint32_t*>(ro.codes + row * K + col) = packed;
        if (lane8 == 0) ro.put_scale(row, (k0 + b * 32) / 32, e8);
      }
    }
    if (do_col) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = tof<T>(tile[(b * 32 + lane8 * 4 + i) * kLd + vec]);
        const uint32_t packed = quantize32<kE5M2>(v, &e8);
        const int64_t row = k0 + vec, col = r0 + b * 32 + lane8 * 4;
        *reinterpret_cast<uint32_t*>(co.codes + row * M + col) = packed;
        if (lane8 == 0) co.put_scale(row, (r0 + b * 32) / 32, e8);
      }
    }
  }
}

std::pair<at::Tensor, MxOut> mx_buffers(int64_t rows, int64_t cols, bool blocked, const at::TensorOptions& opt,
                                        at::Tensor& codes) {
  codes = at::empty({rows, cols}, opt);
  const int64_t sc = cols / 32;
  at::Tensor scales;
  int64_t nc = 0;
  if (blocked) {
    const int64_t nr = (rows + 127) / 128;
    nc = (sc + 3) / 4;
    const bool pad = nr * 128 != rows || nc * 4 != sc;
    scales = pad ? at::full({nr * 128 * nc * 4}, 127, opt) : at::empty({nr * 128 * nc * 4}, opt);
  } else {
    scales = at::empty({rows, sc}, opt);
  }
  return {scales, MxOut{codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), cols, nc, blocked}};
}

}  // namespace

// x [M, K] (M, K multiples of 128) -> (row codes [M, K], row scales, column codes [K, M], column scales); a layout
// that is not requested comes back empty. Codes are E4M3 (or E5M2) bytes, scales E8M0 bytes.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> bl_mx_quantize2(at::Tensor x, bool row, bool col,
                                                                           bool e5m2, bool blocked) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2, "x must be a contiguous 2-D CUDA tensor");
  const int64_t M = x.size(0), K = x.size(1);
  TORCH_CHECK(M % 128 == 0 && K % 128 == 0, "both dimensions must be multiples of 128 (pad first)");
  const c10::cuda::CUDAGuard guard(x.device());
  auto opt = x.options().dtype(at::kByte);
  at::Tensor rc, cc;
  auto [rs, ro] = mx_buffers(row ? M : 0, K, blocked, opt, rc);
  auto [cs, co] = mx_buffers(col ? K : 0, M, blocked, opt, cc);
  dim3 grid(static_cast<unsigned>(K / 128), static_cast<unsigned>(M / 128));
  auto stream = at::cuda::getCurrentCUDAStream();
  auto launch = [&](auto* ptr) {
    using T = std::remove_const_t<std::remove_pointer_t<decltype(ptr)>>;
    const size_t smem = 128 * kLd * sizeof(T);
    auto k = e5m2 ? mx_dual<T, true> : mx_dual<T, false>;
    if (smem > 48 * 1024) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    k<<<grid, 256, smem, stream>>>(ptr, M, K, row, col, ro, co);
  };
  if (x.scalar_type() == at::kBFloat16) launch(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()));
  else if (x.scalar_type() == at::kHalf) launch(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()));
  else {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be bf16, fp16 or fp32");
    launch(x.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const auto f8 = e5m2 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
  return {rc.view(f8), rs.view(at::kFloat8_e8m0fnu), cc.view(f8), cs.view(at::kFloat8_e8m0fnu)};
}

// ---------------------------------------------------------------- flat packing (saved activations)
// Contiguous values in groups of 32 -> E4M3 codes and E8M0 scales (rounded up), and back; no layout requirements
// beyond the length being a multiple of 32. Eight lanes hold one group (4 values each).
namespace {

template <typename T>
__global__ void mx_pack_flat(const T* __restrict__ x, int64_t groups, uint8_t* __restrict__ codes,
                             uint8_t* __restrict__ scales) {
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t g = t >> 3;
  const int lane8 = threadIdx.x & 7;
  const bool live = g < groups;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (live) {
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = tof<T>(x[g * 32 + lane8 * 4 + i]);
  }
  uint8_t e8;
  const uint32_t packed = quantize32<false>(v, &e8);
  if (live) {
    *reinterpret_cast<uint32_t*>(codes + g * 32 + lane8 * 4) = packed;
    if (lane8 == 0) scales[g] = e8;
  }
}

template <typename T> __device__ __forceinline__ T from_f(float v);
template <> __device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }
template <> __device__ __forceinline__ __half from_f<__half>(float v) { return __float2half_rn(v); }
template <> __device__ __forceinline__ float from_f<float>(float v) { return v; }

template <typename T>
__global__ void mx_unpack_flat(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int64_t n,
                               T* __restrict__ out) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
  if (i >= n) return;
  const float s = exp2f(static_cast<float>(scales[i / 32]) - 127.0f);
  const uint32_t raw = *reinterpret_cast<const uint32_t*>(codes + i);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    __nv_fp8_e4m3 c;
    c.__x = static_cast<uint8_t>(raw >> (8 * k));
    out[i + k] = from_f<T>(static_cast<float>(c) * s);
  }
}

}  // namespace

// x contiguous, numel % 32 == 0 -> (E4M3 codes [numel], E8M0 scales [numel / 32])
std::tuple<at::Tensor, at::Tensor> bl_mx_pack(at::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.numel() % 32 == 0, "x: contiguous CUDA, numel % 32 == 0");
  const c10::cuda::CUDAGuard guard(x.device());
  const int64_t groups = x.numel() / 32;
  auto opt = x.options().dtype(at::kByte);
  at::Tensor codes = at::empty({x.numel()}, opt), scales = at::empty({groups}, opt);
  if (groups == 0) return {codes, scales};
  const int64_t threads = groups * 8;
  const unsigned blocks = static_cast<unsigned>((threads + 255) / 256);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (x.scalar_type() == at::kBFloat16)
    mx_pack_flat<__nv_bfloat16><<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()), groups, codes.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>());
  else if (x.scalar_type() == at::kHalf)
    mx_pack_flat<__half><<<blocks, 256, 0, stream>>>(reinterpret_cast<const __half*>(x.data_ptr<at::Half>()), groups,
                                                     codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>());
  else {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be bf16, fp16 or fp32");
    mx_pack_flat<float><<<blocks, 256, 0, stream>>>(x.data_ptr<float>(), groups, codes.data_ptr<uint8_t>(),
                                                    scales.data_ptr<uint8_t>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {codes.view(at::kFloat8_e4m3fn), scales.view(at::kFloat8_e8m0fnu)};
}

at::Tensor bl_mx_unpack(at::Tensor codes, at::Tensor scales, at::ScalarType dtype) {
  TORCH_CHECK(codes.is_cuda() && codes.is_contiguous() && codes.numel() % 32 == 0, "codes: contiguous CUDA");
  const c10::cuda::CUDAGuard guard(codes.device());
  at::Tensor out = at::empty({codes.numel()}, codes.options().dtype(dtype));
  const int64_t n = codes.numel();
  if (n == 0) return out;
  const unsigned blocks = static_cast<unsigned>((n / 4 + 255) / 256);
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint8_t* c = reinterpret_cast<const uint8_t*>(codes.data_ptr());
  const uint8_t* s = reinterpret_cast<const uint8_t*>(scales.data_ptr());
  if (dtype == at::kBFloat16)
    mx_unpack_flat<__nv_bfloat16><<<blocks, 256, 0, stream>>>(c, s, n,
                                                              reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  else if (dtype == at::kHalf)
    mx_unpack_flat<__half><<<blocks, 256, 0, stream>>>(c, s, n, reinterpret_cast<__half*>(out.data_ptr()));
  else {
    TORCH_CHECK(dtype == at::kFloat, "dtype must be bf16, fp16 or fp32");
    mx_unpack_flat<float><<<blocks, 256, 0, stream>>>(c, s, n, out.data_ptr<float>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
