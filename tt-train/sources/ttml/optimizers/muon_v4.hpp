// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "optimizers/optimizer_base.hpp"
#include "serialization/serializable.hpp"

namespace ttml::optimizers {

// DeepSeek-V4 variant of Muon (Algorithm 1 of the V4 report). Compared with the
// vanilla MuonComposite, this adds the four V4 nuances:
//   * Nesterov look-ahead before orthogonalization,
//   * decoupled weight decay,
//   * update-RMS rescale (so AdamW-tuned learning rates transfer),
//   * the hybrid Newton-Schulz coefficient schedule (aggressive then stabilizing).
//
// Per logically-independent 2D weight matrix W in R^{n x m}:
//   M_t  = mu * M_{t-1} + G_t
//   O'_t = HybridNewtonSchulz(mu * M_t + G_t)              [Nesterov + orthogonalize]
//   O_t  = O'_t * sqrt(max(n, m)) * update_rms_scale
//   W_t  = W_{t-1} * (1 - lr * weight_decay) - lr * O_t
//
// Non-matrix parameters (norm gains, biases, gating scalars) cannot be
// orthogonalized; following DeepSeek they should be optimized with AdamW. This
// optimizer applies a plain momentum update to any non-matrix parameter it is
// handed (controlled by `fallback_on_non_matrix`).
struct MuonV4Config {
    float lr{2e-2F};
    float momentum{0.95F};
    float weight_decay{0.0F};
    bool nesterov{true};
    int ns_steps{10};
    int ns_stabilize_steps{2};
    float update_rms_scale{0.2F};
    bool fallback_on_non_matrix{true};
};

class MuonV4 : public OptimizerBase {
public:
    [[nodiscard]] std::string get_name() const override {
        return "MuonV4";
    }

    MuonV4(serialization::NamedParameters parameters, const MuonV4Config& config);

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
    MuonV4Config m_config;
    serialization::NamedParameters m_momentum_buffer;
};

}  // namespace ttml::optimizers
