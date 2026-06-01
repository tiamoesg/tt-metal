// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"
#include "modules/linear_module.hpp"

namespace ttml::modules {

// Manifold-Constrained Hyper-Connections (mHC), DeepSeek-V4 (Xie et al., 2026).
//
// mHC widens the inter-block residual stream by a factor n_hc and replaces the
// plain residual add with a learned, *stable* mixing. The residual state before
// layer l is X_l in R^{n_hc x d}; mapped here onto the N dimension of a rank-4
// tensor [B, n_hc, S, d]. With an inner layer F (attention / MoE / MLP):
//
//   X_hat   = RMSNorm(vec(X_l))                       // [B, S, n_hc*d]
//   A_l     = sigma( alpha_pre  * X_hat W_pre  + S_pre  )   in R^{1 x n_hc}
//   B_l     = Sinkhorn( alpha_res * X_hat W_res + S_res )   in doubly-stochastic R^{n_hc x n_hc}
//   C_l     = 2 sigma( alpha_post * X_hat W_post + S_post ) in R^{n_hc x 1}
//   X_{l+1} = B_l X_l + C_l F( A_l X_l )
//
// The defining trick (vs. plain Hyper-Connections) is constraining B_l to the
// Birkhoff polytope of doubly-stochastic matrices via Sinkhorn-Knopp, which
// bounds ||B_l||_2 <= 1 (a non-expansive residual map) and keeps deep stacks
// numerically stable. A and C are bounded via sigmoid.
//
// IMPLEMENTATION STATUS (staged):
//   * sinkhorn_knopp() and the sigmoid/2*sigmoid gates below are implemented and
//     correct at the ttnn::Tensor level (forward / inference path).
//   * Full end-to-end trainability of the A/B/C generators additionally requires
//     three autograd-level ops that tt-train does not expose yet:
//       - ops::sigmoid(TensorPtr)            (gate for A, C)
//       - ops::exp(TensorPtr)                (positivity step inside Sinkhorn)
//       - slice / concat along the N dimension (to mix the n_hc streams)
//     These are small, standard ops; see the .cpp for the exact insertion points.
//     Until they land, instantiate with `dynamic_parameterization = false` to use
//     learned-but-input-independent A/B/C (the static-bias-only variant).
struct ManifoldHyperConnectionsConfig {
    uint32_t num_streams{4};      // n_hc, residual-stream expansion factor
    uint32_t hidden_dim{0};       // d, must be set to the layer hidden size
    uint32_t sinkhorn_iters{20};  // t_max
    float alpha_init{1e-2F};      // initial value of the learnable gating factors
    bool dynamic_parameterization{true};
};

class ManifoldHyperConnections : public autograd::ModuleBase {
public:
    ManifoldHyperConnections(const ManifoldHyperConnectionsConfig& config, autograd::ModuleBasePtr inner_layer);

    // Takes and returns the *expanded* residual state X in R^{B x n_hc x S x d}.
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& expanded_residual) override;

private:
    ManifoldHyperConnectionsConfig m_config;
    autograd::ModuleBasePtr m_inner_layer;

    // Dynamic (input-dependent) parameter generators W_pre / W_res / W_post.
    std::shared_ptr<LinearLayer> m_w_pre;
    std::shared_ptr<LinearLayer> m_w_res;
    std::shared_ptr<LinearLayer> m_w_post;

    // Static (input-independent) biases S_pre / S_res / S_post and gating factors.
    autograd::TensorPtr m_s_pre;
    autograd::TensorPtr m_s_res;
    autograd::TensorPtr m_s_post;
    autograd::TensorPtr m_alpha_pre;
    autograd::TensorPtr m_alpha_res;
    autograd::TensorPtr m_alpha_post;
};

}  // namespace ttml::modules
