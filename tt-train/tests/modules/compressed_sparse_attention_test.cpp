// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/compressed_sparse_attention.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/binary_ops.hpp"
#include "ops/losses.hpp"

class CompressedSparseAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// End-to-end CSA: [B,1,S,d] -> [B,1,S,d]. Dimensions honor the building blocks'
// constraints: the overlapping compressor needs m a multiple of the 32 tile
// (slice-based segment shift), and ttnn::topk needs the block count G a power of
// two. Here m=32, G=32, S=1024.
TEST_F(CompressedSparseAttentionTest, ForwardShapeAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t batch = 1;
    const uint32_t seq = 1024;
    const uint32_t dim = 64;

    ttml::modules::CompressedSparseAttentionConfig config;
    config.dim = dim;
    config.num_heads = 4;
    config.head_dim = 32;          // c
    config.query_comp_dim = 32;    // d_c
    config.compression_rate = 32;  // m -> G = 32
    config.index_head_dim = 32;    // c^I
    config.num_index_heads = 2;    // n^I_h
    config.top_k = 8;              // k (<= G)
    config.num_groups = 2;
    config.group_inter_dim = 16;
    auto attn = ttml::modules::CompressedSparseAttention(config);

    auto hidden = ttml::autograd::create_tensor(
        ttml::core::ones(ttnn::Shape({batch, 1, seq, dim}), device), /* requires_grad */ true);

    auto out = attn(hidden);
    const auto shape = out->get_value().logical_shape();
    EXPECT_EQ(shape[0], batch);
    EXPECT_EQ(shape[1], 1U);
    EXPECT_EQ(shape[2], seq);
    EXPECT_EQ(shape[3], dim);

    // The indexer scores are exposed for the auxiliary loss.
    auto scores = attn.last_index_scores();
    ASSERT_NE(scores, nullptr);
    EXPECT_EQ(scores->get_value().logical_shape()[3], seq / config.compression_rate);

    // Top-k selection is stop-gradient, so the indexer and index-key compressor
    // are trained only by the indexer's auxiliary loss -- drive backward through a
    // combined (main + aux-proxy) loss so every parameter receives a gradient.
    auto main_loss = ttml::ops::mse_loss(out, ttml::autograd::create_tensor(ttml::core::zeros(shape, device)));
    auto aux_loss = ttml::ops::mse_loss(
        scores, ttml::autograd::create_tensor(ttml::core::zeros(scores->get_value().logical_shape(), device)));
    auto loss = ttml::ops::add(main_loss, aux_loss);
    loss->backward();

    auto params = attn.parameters();
    EXPECT_TRUE(params.at("compressed_sparse_attention/kv_compressor/w_akv/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("compressed_sparse_attention/idx_key_compressor/w_akv/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("compressed_sparse_attention/indexer/w_iuq/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("compressed_sparse_attention/w_uq/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("compressed_sparse_attention/sink_logits")->is_grad_initialized());
    EXPECT_TRUE(params.at("compressed_sparse_attention/out_proj/out_proj/weight")->is_grad_initialized());
    EXPECT_TRUE(hidden->is_grad_initialized());
}
