// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/heavily_compressed_attention.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class HeavilyCompressedAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// End-to-end HCA: [B, 1, S, d] -> [B, 1, S, d], with gradients flowing back
// through the whole pipeline (compressor, query projections, norms, sink, and
// the grouped output projection).
TEST_F(HeavilyCompressedAttentionTest, ForwardShapeAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t batch = 1;
    const uint32_t seq = 128;
    const uint32_t dim = 64;

    ttml::modules::HeavilyCompressedAttentionConfig config;
    config.dim = dim;
    config.num_heads = 4;
    config.head_dim = 32;         // c (tile-aligned)
    config.query_comp_dim = 32;   // d_c
    config.compression_rate = 4;  // m' -> G = 32
    config.num_groups = 2;
    config.group_inter_dim = 16;
    auto attn = ttml::modules::HeavilyCompressedAttention(config);

    auto hidden = ttml::autograd::create_tensor(
        ttml::core::ones(ttnn::Shape({batch, 1, seq, dim}), device), /* requires_grad */ true);

    auto out = attn(hidden);
    const auto shape = out->get_value().logical_shape();
    EXPECT_EQ(shape[0], batch);
    EXPECT_EQ(shape[1], 1U);
    EXPECT_EQ(shape[2], seq);
    EXPECT_EQ(shape[3], dim);

    auto target = ttml::autograd::create_tensor(ttml::core::zeros(shape, device));
    auto loss = ttml::ops::mse_loss(out, target);
    loss->backward();

    auto params = attn.parameters();
    EXPECT_TRUE(params.at("heavily_compressed_attention/compressor/w_kv/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("heavily_compressed_attention/w_dq/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("heavily_compressed_attention/w_uq/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("heavily_compressed_attention/sink_logits")->is_grad_initialized());
    EXPECT_TRUE(params.at("heavily_compressed_attention/out_proj/out_proj/weight")->is_grad_initialized());
    EXPECT_TRUE(hidden->is_grad_initialized());
}
