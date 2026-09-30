// Update directions computed from the gradient (ATen, device-agnostic). All return float32 [rows, cols] with the
// RMS clipped to ``clip``.
#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <optional>

using at::Tensor;

namespace {

enum Direction : int64_t { kRaw = 0, kSign = 1, kColnorm = 2, kSinkhorn = 3, kFactored = 4, kMuon = 5 };

Tensor view2d(const Tensor& g) { return g.dim() == 1 ? g.reshape({1, -1}) : g.reshape({g.size(0), -1}); }

// Newton-Schulz coefficient schedules: (a, b, c) of the odd quintic a x + b x^3 + c x^5 per iteration.
//   kMuonCoeffs    fixed quintic of the Muon reference implementation (singular values end in ~[0.7, 1.2])
//   kPolarExpress  per-iteration minimax schedule (Amsel et al., 2025) with a 1.05 safety factor; iterations past the
//                  table repeat its last entry
enum Coeffs : int64_t { kMuonCoeffs = 0, kPolarExpress = 1 };
enum NsAlgo : int64_t { kNsAuto = 0, kNsStandard = 1, kNsGram = 2 };

struct Abc { double a, b, c; };

Abc coeffs(int64_t kind, int64_t i) {
  if (kind == kMuonCoeffs) return {3.4445, -4.7750, 2.0315};
  static const Abc pe[5] = {{8.28721201814563, -23.595886519098837, 17.300387312530933},
                            {4.107059111542203, -2.9478499167379106, 0.5448431082926601},
                            {3.9486908534822946, -2.908902115962949, 0.5518191394370137},
                            {3.3184196573706015, -2.488488024314874, 0.51004894012372},
                            {2.300652019954817, -1.6689039845747493, 0.4188073119525673}};
  const Abc& e = pe[std::min<int64_t>(i, 4)];
  constexpr double f = 1.05;
  return {e.a / f, e.b / (f * f * f), e.c / (f * f * f * f * f)};
}

// Standard iteration on a wide batch x [B, n, m] (n <= m): per step 3 GEMMs, two of them n x n x m.
Tensor ns_standard(Tensor x, int64_t steps, int64_t ck) {
  for (int64_t i = 0; i < steps; ++i) {
    const Abc k = coeffs(ck, i);
    Tensor s = at::bmm(x, x.transpose(-2, -1));
    Tensor B = at::baddbmm(s, s, s, /*beta=*/k.b, /*alpha=*/k.c);        // b*S + c*S^2
    x = at::baddbmm(x, B, x, /*beta=*/k.a, /*alpha=*/1.0);              // a*X + B X
  }
  return x;
}

// Gram iteration (Zhang, Amsel, Chen, Dao, 2026): iterate on the n x n Gram matrix R = X X^T and accumulate the
// polynomial Q, so only the first Gram product and the final Q X touch the rectangular [n, m] matrix. The scalar
// term a*I is applied implicitly (baddbmm beta) to avoid rounding it into low-precision products. The iteration is
// restarted from X <- Q X after ``restart`` steps to reset spurious negative eigenvalues of R.
Tensor ns_gram(Tensor x, int64_t steps, int64_t ck, int64_t restart) {
  Tensor R = at::bmm(x, x.transpose(-2, -1));
  Tensor Q;
  const int64_t n = R.size(-1);
  Tensor I = at::eye(n, R.options()).expand_as(R);
  for (int64_t i = 0; i < steps; ++i) {
    const Abc k = coeffs(ck, i);
    const bool fresh = i == 0 || (restart > 0 && i == restart);
    if (i > 0 && fresh) {
      x = at::bmm(Q, x);
      R = at::bmm(x, x.transpose(-2, -1));
    }
    Tensor Z = at::baddbmm(R, R, R, /*beta=*/k.b, /*alpha=*/k.c);       // b R + c R^2
    Q = fresh ? Z + k.a * I : at::baddbmm(Q, Q, Z, /*beta=*/k.a, /*alpha=*/1.0);   // Q (a I + Z)
    const bool last = i + 1 == steps || (restart > 0 && i + 1 == restart);
    if (!last) {
      Tensor RZ = at::baddbmm(R, R, Z, /*beta=*/k.a, /*alpha=*/1.0);    // R (a I + Z)
      R = at::baddbmm(RZ, Z, RZ, /*beta=*/k.a, /*alpha=*/1.0);          // (a I + Z) R (a I + Z)
    }
  }
  return at::bmm(Q, x);
}

Tensor newton_schulz(Tensor x, int64_t steps, int64_t ck = kMuonCoeffs, int64_t algo = kNsStandard) {
  const auto dtype = x.scalar_type();
  x = x.to(at::kFloat);
  x = (x / (at::linalg_vector_norm(x, 2, {-2, -1}, /*keepdim=*/true) + 1e-7)).to(dtype);
  const bool tr = x.size(-2) > x.size(-1);
  if (tr) x = x.transpose(-2, -1);
  const bool batched = x.dim() == 3;
  if (!batched) x = x.unsqueeze(0);
  x = x.contiguous();
  // auto: Gram only where it saves work (aspect >= 2) and not in bf16, whose 8-bit mantissa destabilizes it
  const bool gram = algo == kNsGram || (algo == kNsAuto && x.size(-1) >= 2 * x.size(-2) && dtype != at::kBFloat16);
  x = gram ? ns_gram(x, steps, ck, /*restart=*/2) : ns_standard(x, steps, ck);
  if (!batched) x = x.squeeze(0);
  return tr ? x.transpose(-2, -1) : x;
}

// Block-diagonal orthogonalization with [br, bc] blocks (bc == 0: square br x br); each block is scaled by
// sqrt(max(br, bc)), the full matrix by sqrt(max(m, n)).
Tensor muon(const Tensor& G, int64_t br, int64_t bc, int64_t steps, at::ScalarType dtype, int64_t ck, int64_t algo) {
  Tensor g = view2d(G).to(dtype);
  const int64_t m = g.size(0), n = g.size(1);
  if (bc <= 0) bc = br;
  if (br <= 0 || (m <= br && n <= bc))
    return (newton_schulz(g, steps, ck, algo) * std::sqrt(static_cast<double>(std::max(m, n)))).to(at::kFloat);
  Tensor X = at::constant_pad_nd(g, {0, (bc - n % bc) % bc, 0, (br - m % br) % br});
  const int64_t M = X.size(0), N = X.size(1);
  X = X.view({M / br, br, N / bc, bc}).permute({0, 2, 1, 3}).reshape({-1, br, bc});
  Tensor Y = newton_schulz(X, steps, ck, algo) * std::sqrt(static_cast<double>(std::max(br, bc)));
  Y = Y.view({M / br, N / bc, br, bc}).permute({0, 2, 1, 3}).reshape({M, N});
  return Y.slice(0, 0, m).slice(1, 0, n).to(at::kFloat);
}

Tensor factored(const Tensor& G, Tensor r, Tensor c, int64_t k, double beta2, double eps) {
  Tensor X = view2d(G).to(at::kFloat);
  Tensor pk = X.abs().amax();
  X = X / at::where(pk > 0, pk, at::ones_like(pk));
  Tensor X2 = X.square();
  r.mul_(beta2).add_(X2.mean(1), 1.0 - beta2);
  c.mul_(beta2).add_(X2.mean(0), 1.0 - beta2);
  const double bc = 1.0 - std::pow(beta2, static_cast<double>(k));
  Tensor rr = r / bc, cc = c / bc;
  Tensor V = (rr / rr.mean().clamp_min(eps)).unsqueeze(1) * cc.unsqueeze(0);
  return X / (V.sqrt() + eps);
}

Tensor sinkhorn(const Tensor& G, int64_t iters) {
  Tensor U = view2d(G).to(at::kFloat);
  for (int64_t i = 0; i < iters; ++i) {
    U = U / (U.square().mean(1, true).sqrt() + 1e-30);
    U = U / (U.square().mean(0, true).sqrt() + 1e-30);
  }
  return U;
}

}  // namespace

// ns_dtype: 0 fp32, 1 bf16, 2 fp16
Tensor bl_direction(Tensor G, int64_t kind, std::optional<Tensor> r, std::optional<Tensor> c, int64_t k, double beta2,
                    int64_t block, int64_t ns_steps, int64_t ns_dtype, int64_t sinkhorn_iters, double clip,
                    int64_t block_cols, int64_t ns_coeffs, int64_t ns_algo) {
  Tensor U;
  switch (kind) {
    case kRaw: U = view2d(G).to(at::kFloat); break;
    case kSign: U = view2d(G).to(at::kFloat).sign(); break;
    case kColnorm: {
      Tensor g = view2d(G).to(at::kFloat);
      U = g / (g.square().mean(0, true).sqrt() + 1e-30);
      break;
    }
    case kSinkhorn: U = sinkhorn(G, sinkhorn_iters); break;
    case kFactored:
      TORCH_CHECK(r.has_value() && c.has_value(), "factored direction needs r and c state");
      U = factored(G, *r, *c, k, beta2, 1e-12);
      break;
    case kMuon:
      U = muon(G, block, block_cols, ns_steps, ns_dtype == 1 ? at::kBFloat16 : ns_dtype == 2 ? at::kHalf : at::kFloat,
               ns_coeffs, ns_algo);
      break;
    default: TORCH_CHECK(false, "unknown direction ", kind);
  }
  Tensor rho = U.square().mean().sqrt();
  return (U / at::clamp_min(rho / clip, 1.0)).contiguous();
}
