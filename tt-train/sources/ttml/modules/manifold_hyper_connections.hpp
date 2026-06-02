// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "autograd/tensor.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "modules/rms_norm_module.hpp"

namespace ttml::modules {

// Manifold-Constrained Hyper-Connections (mHC), DeepSeek-V4 (Xie et al., 2026).
//
// mHC widens the inter-block residual stream by a factor n_hc and replaces the
// plain residual add with a learned, *stable* mixing. The residual state before
// layer l is X_l in R^{n_hc x d}; mapped here onto the N dimension of a rank-4
// tensor [B, n_hc, S, d]. With an inner layer F (attention / MoE / MLP):
//
//   X_hat = RMSNorm(vec(X_l))                            // dynamic mode only
//   A     = sigma( alpha_pre  * X_hat W_pre  + S_pre )   in (0, 1)^{1 x n_hc}
//   B     = Sinkhorn( alpha_res * X_hat W_res + S_res )  doubly-stochastic^{n_hc x n_hc}
//   C     = 2 sigma( alpha_post * X_hat W_post + S_post ) in (0, 2)^{n_hc x 1}
//   X_{l+1} = B X_l + C F(A X_l)
//
// The defining trick (vs. plain Hyper-Connections) is constraining B to the
// Birkhoff polytope of doubly-stochastic matrices via Sinkhorn-Knopp, which
// bounds ||B||_2 <= 1 (a non-expansive residual map) and keeps deep stacks
// numerically stable. A and C are bounded via sigmoid.
//
// Two parameterizations are supported:
//   * static  (dynamic_parameterization = false): A/B/C are learned but
//     input-independent (the raw_* parameters are constrained directly).
//   * dynamic (dynamic_parameterization = true): A/B/C are generated per token
//     from X_hat = RMSNorm(vec(X_l)) via the W_* projections, matching the
//     DeepSeek-V4 formulation.
//
// Both paths are fully differentiable, built from autograd ops (sigmoid, exp,
// sum, broadcast_to, reshape, permute, matmul_op) with no new kernels. When a
// mask is supplied it is forwarded to the wrapped inner layer (so mHC can wrap
// masked attention as well as MLP/MoE).
struct ManifoldHyperConnectionsConfig {
    uint32_t num_streams{4};      // n_hc, residual-stream expansion factor
    uint32_t hidden_dim{0};       // d, must be set to the layer hidden size
    uint32_t sinkhorn_iters{20};  // t_max
    float alpha_init{1e-2F};      // initial value of the dynamic gating factors
    bool dynamic_parameterization{false};
};

class ManifoldHyperConnections : public ModuleBase {
public:
    ManifoldHyperConnections(const ManifoldHyperConnectionsConfig& config, ModuleBasePtr inner_layer);

    // Takes and returns the *expanded* residual state X in R^{B x n_hc x S x d}.
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& expanded_residual) override;
    [[nodiscard]] autograd::TensorPtr operator()(
        const autograd::TensorPtr& expanded_residual, const std::optional<autograd::TensorPtr>& mask) override;

private:
    autograd::TensorPtr forward(const autograd::TensorPtr& x, const std::optional<autograd::TensorPtr>& mask);
    autograd::TensorPtr compute_static(const autograd::TensorPtr& x, const std::optional<autograd::TensorPtr>& mask);
    autograd::TensorPtr compute_dynamic(const autograd::TensorPtr& x, const std::optional<autograd::TensorPtr>& mask);
    autograd::TensorPtr run_inner(const autograd::TensorPtr& x, const std::optional<autograd::TensorPtr>& mask);

    ManifoldHyperConnectionsConfig m_config;
    ModuleBasePtr m_inner_layer;

    // --- static parameterization ---
    // Raw (pre-constraint) parameters. After the manifold constraints these
    // become A in (0,1)^{1 x n_hc}, B doubly-stochastic^{n_hc x n_hc}, and
    // C in (0,2)^{n_hc x 1}.
    autograd::TensorPtr m_raw_a;
    autograd::TensorPtr m_raw_b;
    autograd::TensorPtr m_raw_c;

    // --- dynamic parameterization ---
    std::shared_ptr<RMSNormLayer> m_vec_norm;
    std::shared_ptr<LinearLayer> m_w_pre;
    std::shared_ptr<LinearLayer> m_w_res;
    std::shared_ptr<LinearLayer> m_w_post;
    autograd::TensorPtr m_s_pre;
    autograd::TensorPtr m_s_res;
    autograd::TensorPtr m_s_post;
    autograd::TensorPtr m_alpha_pre;
    autograd::TensorPtr m_alpha_res;
    autograd::TensorPtr m_alpha_post;
};

}  // namespace ttml::modules
