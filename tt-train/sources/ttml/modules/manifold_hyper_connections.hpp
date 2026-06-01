// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "autograd/module_base.hpp"
#include "autograd/tensor.hpp"

namespace ttml::modules {

// Manifold-Constrained Hyper-Connections (mHC), DeepSeek-V4 (Xie et al., 2026).
//
// mHC widens the inter-block residual stream by a factor n_hc and replaces the
// plain residual add with a learned, *stable* mixing. The residual state before
// layer l is X_l in R^{n_hc x d}; mapped here onto the N dimension of a rank-4
// tensor [B, n_hc, S, d]. With an inner layer F (attention / MoE / MLP):
//
//   A   = sigma(raw_A)            in (0, 1)^{1 x n_hc}
//   B   = Sinkhorn(raw_B)         doubly-stochastic in [0, 1]^{n_hc x n_hc}
//   C   = 2 sigma(raw_C)          in (0, 2)^{n_hc x 1}
//   X_{l+1} = B X_l + C F(A X_l)
//
// The defining trick (vs. plain Hyper-Connections) is constraining B to the
// Birkhoff polytope of doubly-stochastic matrices via Sinkhorn-Knopp, which
// bounds ||B||_2 <= 1 (a non-expansive residual map) and keeps deep stacks
// numerically stable. A and C are bounded via sigmoid.
//
// This module is fully differentiable: the gates (ops::sigmoid), the Sinkhorn
// projection (ops::exp / ops::sum / ops::broadcast_to / ops::div) and the stream
// mixing (ops::reshape + ops::matmul_op) all flow gradients into raw_A/B/C.
//
// STATUS: implements the *static* parameterization, i.e. A/B/C are learned but
// input-independent. The full DeepSeek-V4 *dynamic* form generates A/B/C
// per-token from RMSNorm(vec(X_l)) via additional projections; that needs a
// permute op to form vec(X_l) and per-token Sinkhorn, and is the next step
// (set dynamic_parameterization=true to opt in once implemented).
struct ManifoldHyperConnectionsConfig {
    uint32_t num_streams{4};      // n_hc, residual-stream expansion factor
    uint32_t hidden_dim{0};       // d, must be set to the layer hidden size
    uint32_t sinkhorn_iters{20};  // t_max
    bool dynamic_parameterization{false};
};

class ManifoldHyperConnections : public autograd::ModuleBase {
public:
    ManifoldHyperConnections(const ManifoldHyperConnectionsConfig& config, autograd::ModuleBasePtr inner_layer);

    // Takes and returns the *expanded* residual state X in R^{B x n_hc x S x d}.
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& expanded_residual) override;

private:
    ManifoldHyperConnectionsConfig m_config;
    autograd::ModuleBasePtr m_inner_layer;

    // Raw (pre-constraint) parameters. After the manifold constraints these
    // become A in (0,1)^{1 x n_hc}, B doubly-stochastic^{n_hc x n_hc}, and
    // C in (0,2)^{n_hc x 1}.
    autograd::TensorPtr m_raw_a;
    autograd::TensorPtr m_raw_b;
    autograd::TensorPtr m_raw_c;
};

}  // namespace ttml::modules
