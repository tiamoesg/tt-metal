// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_moe.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <core/ttnn_all_includes.hpp>
#include <stdexcept>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"
#include "ttnn/operations/reduction/topk/topk.hpp"

namespace ttml::modules {

DeepSeekMoE::DeepSeekMoE(const DeepSeekMoEConfig& config) : m_config(config) {
    if (config.dim == 0U || config.inter_dim == 0U || config.num_routed_experts == 0U || config.num_activated == 0U) {
        throw std::invalid_argument(
            "DeepSeekMoEConfig: dim, inter_dim, num_routed_experts, num_activated must be set.");
    }

    create_name("deepseek_moe");
    m_gate = std::make_shared<LinearLayer>(config.dim, config.num_routed_experts, /* has_bias */ false);
    register_module(m_gate, "gate");

    m_experts.reserve(config.num_routed_experts);
    for (uint32_t e = 0; e < config.num_routed_experts; ++e) {
        auto expert = std::make_shared<SwiGLUMLP>(SwiGLUMLPConfig{.dim = config.dim, .inter_dim = config.inter_dim});
        register_module(expert, fmt::format("expert_{}", e));
        m_experts.push_back(std::move(expert));
    }
    m_shared.reserve(config.num_shared_experts);
    for (uint32_t s = 0; s < config.num_shared_experts; ++s) {
        auto shared = std::make_shared<SwiGLUMLP>(SwiGLUMLPConfig{.dim = config.dim, .inter_dim = config.inter_dim});
        register_module(shared, fmt::format("shared_{}", s));
        m_shared.push_back(std::move(shared));
    }
}

autograd::TensorPtr DeepSeekMoE::operator()(const autograd::TensorPtr& x) {
    const auto shape = x->get_value().logical_shape().to_array_4D();
    const uint32_t batch = shape[0];
    const uint32_t seq = shape[2];
    const uint32_t dim = shape[3];
    const uint32_t experts = m_config.num_routed_experts;
    const auto full = ttnn::Shape({batch, 1U, seq, experts});
    auto* device = &autograd::ctx().get_device();

    // Gate scores per (token, expert): [B, 1, S, E].
    auto raw = (*m_gate)(x);
    autograd::TensorPtr scores;
    switch (m_config.score_func) {
        case MoEScoreFunc::Softmax: scores = ops::softmax(raw, /* dim */ -1); break;
        case MoEScoreFunc::Sigmoid: scores = ops::sigmoid(raw); break;
        case MoEScoreFunc::SqrtSoftplus:
        default: scores = ops::sqrt(ops::softplus(raw)); break;
    }

    // Top-k expert selection per token (stop-gradient): build a 0/1 keep mask.
    uint32_t k = std::min(m_config.num_activated, experts);
    if (k == 0U) {
        k = 1U;
    }
    auto scores_val = scores->get_value();
    auto topk_out = ttnn::topk(scores_val, k, /* dim */ -1, /* largest */ true, /* sorted */ true);
    const ttsl::SmallVector<uint32_t> start = {0U, 0U, 0U, k - 1U};
    const ttsl::SmallVector<uint32_t> end = {batch, 1U, seq, k};
    const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
    auto threshold = ttnn::repeat(ttnn::slice(topk_out[0], start, end, step), ttnn::Shape({1, 1, 1, experts}));
    auto keep = autograd::create_tensor(ttnn::ge(ttnn::subtract(scores_val, threshold), 0.0F));  // [B,1,S,E]

    // Routing weights: w = scores * keep, normalized (except softmax), scaled.
    auto weighted = ops::mul(scores, keep);
    autograd::TensorPtr weights;
    if (m_config.score_func == MoEScoreFunc::Softmax) {
        weights = ops::mul(weighted, m_config.route_scale);
    } else {
        auto denom = ops::sum(weighted, /* dim */ -1, /* keep_dim */ true);  // [B,1,S,1]
        weights = ops::mul(ops::div(weighted, ops::broadcast_to(denom, full)), m_config.route_scale);
    }

    // Shared expert(s) always run.
    autograd::TensorPtr y = (*m_shared[0])(x);
    for (size_t s = 1; s < m_shared.size(); ++s) {
        y = ops::add(y, (*m_shared[s])(x));
    }

    // Dense-masked routed experts: each expert scaled by its routing weight column
    // (extracted via a one-hot reduce to avoid non-tile-aligned slicing).
    for (uint32_t e = 0; e < experts; ++e) {
        std::vector<float> onehot(experts, 0.0F);
        onehot[e] = 1.0F;
        auto onehot_t = autograd::create_tensor(core::from_vector(onehot, ttnn::Shape({1, 1, 1, experts}), device));
        auto w_e = ops::sum(ops::mul(weights, onehot_t), /* dim */ -1, /* keep_dim */ true);  // [B,1,S,1]
        auto expert_out = (*m_experts[e])(x);                                                 // [B,1,S,d]
        y = ops::add(y, ops::mul(expert_out, ops::broadcast_to(w_e, ttnn::Shape({batch, 1U, seq, dim}))));
    }

    return y;
}

}  // namespace ttml::modules
