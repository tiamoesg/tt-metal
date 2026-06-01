// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <core/ttnn_all_includes.hpp>

#include "optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

// Muon ("MomentUm Orthogonalized by Newton-schulz") optimizer.
//
// Reference: Jordan et al. (2024) and the DeepSeek-V4 variant (Algorithm 1),
// which updates the majority of weight matrices with Muon while keeping AdamW
// for the embedding, the prediction head, RMSNorm weights, and the mHC static
// biases/gating factors.
//
// Per logically-independent 2D weight matrix W in R^{n x m}:
//   M_t  = mu * M_{t-1} + G_t                         (momentum buffer)
//   O'_t = NewtonSchulz(mu * M_t + G_t)               (Nesterov + orthogonalize)
//   O_t  = O'_t * sqrt(max(n, m)) * gamma             (rescale update RMS)
//   W_t  = W_{t-1} * (1 - eta * lambda) - eta * O_t   (decoupled decay + update)
//
// Parameters that are not 2D matrices (e.g. 1D RMSNorm gains, biases, gating
// scalars) cannot be orthogonalized. By convention these should be optimized
// with AdamW instead; this optimizer applies a plain momentum (SGD-style)
// update to any non-matrix parameter it is handed, controlled by
// `fallback_on_non_matrix`. See `is_matrix_parameter` in the .cpp.
struct MuonConfig {
    float lr{2e-2F};
    float momentum{0.95F};
    float weight_decay{0.0F};
    bool nesterov{true};

    // Number of Newton-Schulz iterations. DeepSeek-V4 uses a hybrid schedule of
    // 10 iterations: the first (ns_steps - ns_stabilize_steps) use the
    // aggressive coefficients, the final ns_stabilize_steps pin the singular
    // values to 1.
    uint32_t ns_steps{10};
    uint32_t ns_stabilize_steps{2};

    // Update-RMS rescale factor (gamma in Algorithm 1). Chosen so Muon updates
    // can reuse AdamW-tuned learning rates.
    float update_rms_scale{0.2F};

    // If true, apply a plain momentum update to non-matrix parameters instead of
    // throwing. If false, non-matrix parameters must not be present.
    bool fallback_on_non_matrix{true};
};

class Muon : public OptimizerBase {
public:
    Muon(serialization::NamedParameters parameters, const MuonConfig& config);

    void zero_grad() override;

    void step() override;

    [[nodiscard]] serialization::StateDict get_state_dict() const override;
    void set_state_dict(const serialization::StateDict& dict) override;

    [[nodiscard]] size_t get_steps() const override;
    void set_steps(size_t steps) override;

    [[nodiscard]] float get_lr() const override {
        return m_config.lr;
    }

    void set_lr(float lr) override {
        m_config.lr = lr;
    }

private:
    size_t m_steps{0};
    MuonConfig m_config;
    // Per-parameter momentum buffer M_t.
    serialization::NamedParameters m_momentum;
};

}  // namespace ttml::optimizers
