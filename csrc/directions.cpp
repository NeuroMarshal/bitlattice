// Update directions computed from the gradient (ATen, device-agnostic). All return float32 [rows, cols] with the
// RMS clipped to ``clip``.
#include <ATen/ATen.h>

#include <cmath>
#include <optional>

using at::Tensor;

namespace {

enum Direction : int64_t { kRaw = 0, kSign = 1, kColnorm = 2, kSinkhorn = 3, kFactored = 4, kMuon = 5 };

Tensor view2d(const Tensor& g) { return g.dim() == 1 ? g.reshape({1, -1}) : g.reshape({g.size(0), -1}); }

Tensor newton_schulz(Tensor x, int64_t steps) {
  constexpr double a = 3.4445, b = -4.7750, c = 2.0315;
  x = x / (at::linalg_vector_norm(x, 2, {-2, -1}, /*keepdim=*/true) + 1e-7);
  const bool tr = x.size(-2) > x.size(-1);
  if (tr) x = x.transpose(-2, -1);
  const bool batched = x.dim() == 3;
  if (!batched) x = x.unsqueeze(0);
  x = x.contiguous();
  for (int64_t i = 0; i < steps; ++i) {
    Tensor s = at::bmm(x, x.transpose(-2, -1));
    Tensor B = at::baddbmm(s, s, s, /*beta=*/b, /*alpha=*/c);        // b*S + c*S^2
    x = at::baddbmm(x, B, x, /*beta=*/a, /*alpha=*/1.0);              // a*X + B X
  }
  if (!batched) x = x.squeeze(0);
  return tr ? x.transpose(-2, -1) : x;
}

Tensor muon(const Tensor& G, int64_t block, int64_t steps, at::ScalarType dtype) {
  Tensor g = view2d(G).to(dtype);
  const int64_t m = g.size(0), n = g.size(1);
  if (block <= 0 || (m <= block && n <= block))
    return (newton_schulz(g, steps) * std::sqrt(static_cast<double>(std::max(m, n)))).to(at::kFloat);
  Tensor X = at::constant_pad_nd(g, {0, (block - n % block) % block, 0, (block - m % block) % block});
  const int64_t M = X.size(0), N = X.size(1);
  X = X.view({M / block, block, N / block, block}).permute({0, 2, 1, 3}).reshape({-1, block, block});
  Tensor Y = newton_schulz(X, steps) * std::sqrt(static_cast<double>(block));
  Y = Y.view({M / block, N / block, block, block}).permute({0, 2, 1, 3}).reshape({M, N});
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

Tensor bl_direction(Tensor G, int64_t kind, std::optional<Tensor> r, std::optional<Tensor> c, int64_t k, double beta2,
                    int64_t block, int64_t ns_steps, int64_t ns_dtype, int64_t sinkhorn_iters, double clip) {
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
    case kMuon: U = muon(G, block, ns_steps, ns_dtype == 1 ? at::kBFloat16 : at::kFloat); break;
    default: TORCH_CHECK(false, "unknown direction ", kind);
  }
  Tensor rho = U.square().mean().sqrt();
  return (U / at::clamp_min(rho / clip, 1.0)).contiguous();
}
