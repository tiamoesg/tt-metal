// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

#include "autograd/tensor.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

// Grouped Output Projection, DeepSeek-V4 (§2.3.1). In V4 the core-attention head
// count times head dim (c * n_h) is large, so projecting [o_{t,1}; ...; o_{t,n_h}]
// directly to a d-dimensional hidden state is expensive. Instead the n_h head
// outputs are split into g groups; each group (n_h/g heads, i.e. c*n_h/g dims) is
// projected to a smaller d_g intermediate, the g intermediates are concatenated,
// and a final projection maps g*d_g -> d:
//
//   o^{G'}_{t,i} = o^{G}_{t,i} W^{G}_i        (per-group, c*n_h/g -> d_g)
//   ^o_t = [o^{G'}_{t,1}; ...; o^{G'}_{t,g}] W^{O}     (g*d_g -> d)
//
// Each group has its own projection weights (grouped, like grouped convolutions).
struct GroupedOutputProjectionConfig {
    uint32_t num_heads{0};        // n_h
    uint32_t head_dim{0};         // c
    uint32_t num_groups{0};       // g (must divide n_h)
    uint32_t group_inter_dim{0};  // d_g
    uint32_t out_dim{0};          // d
};

class GroupedOutputProjection : public ModuleBase {
public:
    explicit GroupedOutputProjection(const GroupedOutputProjectionConfig& config);

    // attn_out: [B, n_h, S, c]  ->  [B, 1, S, d].
    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& attn_out) override;

private:
    GroupedOutputProjectionConfig m_config;
    std::vector<std::shared_ptr<LinearLayer>> m_group_proj;  // g x (c*n_h/g -> d_g)
    std::shared_ptr<LinearLayer> m_out_proj;                 // g*d_g -> d
};

}  // namespace ttml::modules
