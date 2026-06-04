// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "models/deepseek_v4.hpp"

#include <fmt/format.h>

#include <stdexcept>

#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::models::deepseek_v4 {

DeepSeekV4Transformer::DeepSeekV4Transformer(const DeepSeekV4Config& config) : m_config(config) {
    if (config.vocab_size == 0U || config.dim == 0U || config.num_layers == 0U || config.ffn_inter_dim == 0U ||
        config.num_streams == 0U) {
        throw std::invalid_argument(
            "DeepSeekV4Config: vocab_size, dim, num_layers, ffn_inter_dim, num_streams must be set.");
    }

    create_name("deepseek_v4");

    m_embedding = std::make_shared<modules::Embedding>(config.vocab_size, config.dim);
    register_module(m_embedding, "embedding");

    m_blocks.reserve(config.num_layers);
    for (uint32_t i = 0; i < config.num_layers; ++i) {
        modules::DeepSeekV4BlockConfig block_cfg;
        block_cfg.dim = config.dim;
        block_cfg.num_streams = config.num_streams;
        block_cfg.ffn_inter_dim = config.ffn_inter_dim;
        block_cfg.sinkhorn_iters = config.sinkhorn_iters;
        block_cfg.alpha_init = config.alpha_init;
        block_cfg.attn = config.attn;
        if (!config.layer_sparse.empty()) {
            block_cfg.attn.use_sparse = config.layer_sparse[i];
        }
        auto block = std::make_shared<modules::DeepSeekV4Block>(block_cfg);
        register_module(block, fmt::format("block_{}", i));
        m_blocks.push_back(std::move(block));
    }

    m_final_norm = std::make_shared<modules::RMSNormLayer>(config.dim);
    m_head = std::make_shared<modules::LinearLayer>(config.dim, config.vocab_size, /* has_bias */ false);
    register_module(m_final_norm, "final_norm");
    register_module(m_head, "head");
}

autograd::TensorPtr DeepSeekV4Transformer::operator()(const autograd::TensorPtr& tokens) {
    auto emb = (*m_embedding)(tokens);  // [B, 1, S, d]
    const auto shape = emb->get_value().logical_shape().to_array_4D();
    const uint32_t batch = shape[0];
    const uint32_t seq = shape[2];
    const uint32_t dim = shape[3];
    const uint32_t n_hc = m_config.num_streams;

    // Expand the hidden state into n_hc residual-stream copies: [B, 1, S, d] -> [B, n_hc, S, d].
    auto x = ops::broadcast_to(emb, ttnn::Shape({batch, n_hc, seq, dim}));

    for (auto& block : m_blocks) {
        x = (*block)(x);
    }

    // Reduce the n_hc streams back to one (mean over streams): [B, n_hc, S, d] -> [B, 1, S, d].
    auto reduced = ops::mul(ops::sum(x, /* dim */ 1, /* keep_dim */ true), 1.0F / static_cast<float>(n_hc));

    auto normed = (*m_final_norm)(reduced);
    return (*m_head)(normed);  // [B, 1, S, vocab]
}

}  // namespace ttml::models::deepseek_v4
