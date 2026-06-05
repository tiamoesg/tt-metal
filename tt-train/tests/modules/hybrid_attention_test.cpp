// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/hybrid_attention.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class HybridAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// A sparse (CSA) hybrid layer: preserves [B,1,S,d], reports is_sparse(), and
// exposes the indexer scores for the auxiliary loss.
TEST_F(HybridAttentionTest, SparseLayerForward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t seq = 1024;
    const uint32_t dim = 64;

    ttml::modules::HybridAttentionConfig config;
    config.use_sparse = true;
    config.csa.dim = dim;
    config.csa.num_heads = 4;
    config.csa.head_dim = 32;
    config.csa.query_comp_dim = 32;
    config.csa.compression_rate = 32;  // m -> G = 32 (power of two)
    config.csa.index_head_dim = 32;
    config.csa.num_index_heads = 2;
    config.csa.top_k = 8;
    config.csa.num_groups = 2;
    config.csa.group_inter_dim = 16;
    auto attn = ttml::modules::HybridAttention(config);

    EXPECT_TRUE(attn.is_sparse());

    auto hidden = ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({1, 1, seq, dim}), device));
    auto out = attn(hidden);
    EXPECT_EQ(out->get_value().logical_shape()[3], dim);
    EXPECT_EQ(out->get_value().logical_shape()[2], seq);
    EXPECT_NE(attn.last_index_scores(), nullptr);
}

// A dense (HCA) hybrid layer: preserves shape and exposes no index scores.
TEST_F(HybridAttentionTest, DenseLayerForward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t seq = 128;
    const uint32_t dim = 64;

    ttml::modules::HybridAttentionConfig config;
    config.use_sparse = false;
    config.hca.dim = dim;
    config.hca.num_heads = 4;
    config.hca.head_dim = 32;
    config.hca.query_comp_dim = 32;
    config.hca.compression_rate = 4;  // m' -> G = 32
    config.hca.num_groups = 2;
    config.hca.group_inter_dim = 16;
    auto attn = ttml::modules::HybridAttention(config);

    EXPECT_FALSE(attn.is_sparse());

    auto hidden = ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({1, 1, seq, dim}), device), true);
    auto out = attn(hidden);
    EXPECT_EQ(out->get_value().logical_shape()[3], dim);
    EXPECT_EQ(attn.last_index_scores(), nullptr);

    auto target = ttml::autograd::create_tensor(ttml::core::zeros(out->get_value().logical_shape(), device));
    auto loss = ttml::ops::mse_loss(out, target);
    loss->backward();
    EXPECT_TRUE(hidden->is_grad_initialized());
}
