// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/deepseek_v4_block.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class DeepSeekV4BlockTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

namespace {
ttml::modules::DeepSeekV4BlockConfig make_block_config(uint32_t dim, uint32_t n_hc, bool sparse) {
    ttml::modules::DeepSeekV4BlockConfig cfg;
    cfg.dim = dim;
    cfg.num_streams = n_hc;
    cfg.ffn_inter_dim = 128;
    cfg.attn.use_sparse = sparse;
    if (sparse) {
        cfg.attn.csa.dim = dim;
        cfg.attn.csa.num_heads = 4;
        cfg.attn.csa.head_dim = 32;
        cfg.attn.csa.query_comp_dim = 32;
        cfg.attn.csa.compression_rate = 32;  // m -> G power of two
        cfg.attn.csa.index_head_dim = 32;
        cfg.attn.csa.num_index_heads = 2;
        cfg.attn.csa.top_k = 8;
        cfg.attn.csa.num_groups = 2;
        cfg.attn.csa.group_inter_dim = 16;
    } else {
        cfg.attn.hca.dim = dim;
        cfg.attn.hca.num_heads = 4;
        cfg.attn.hca.head_dim = 32;
        cfg.attn.hca.query_comp_dim = 32;
        cfg.attn.hca.compression_rate = 4;
        cfg.attn.hca.num_groups = 2;
        cfg.attn.hca.group_inter_dim = 16;
    }
    return cfg;
}
}  // namespace

// A dense (HCA) V4 block preserves the expanded residual shape [B, n_hc, S, d] and
// backpropagates through the whole mHC-wrapped attention + FFN.
TEST_F(DeepSeekV4BlockTest, HcaBlockForwardAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t dim = 64;
    const uint32_t n_hc = 4;
    const uint32_t seq = 128;

    auto block = ttml::modules::DeepSeekV4Block(make_block_config(dim, n_hc, /* sparse */ false));
    auto x = ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({1, n_hc, seq, dim}), device), true);

    auto out = block(x);
    EXPECT_EQ(out->get_value().logical_shape(), x->get_value().logical_shape());

    auto target = ttml::autograd::create_tensor(ttml::core::zeros(out->get_value().logical_shape(), device));
    auto loss = ttml::ops::mse_loss(out, target);
    loss->backward();
    EXPECT_TRUE(x->is_grad_initialized());

    // The mHC dynamic generators and the FFN must receive gradients.
    auto params = block.parameters();
    EXPECT_TRUE(params.at("deepseek_v4_block/attn_mhc/w_res/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("deepseek_v4_block/ffn_mhc/inner_layer/w1/weight")->is_grad_initialized());
}

// A sparse (CSA) V4 block preserves shape end-to-end.
TEST_F(DeepSeekV4BlockTest, CsaBlockForwardShape) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t dim = 64;
    const uint32_t n_hc = 4;
    const uint32_t seq = 1024;

    auto block = ttml::modules::DeepSeekV4Block(make_block_config(dim, n_hc, /* sparse */ true));
    auto x = ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({1, n_hc, seq, dim}), device));
    auto out = block(x);
    EXPECT_EQ(out->get_value().logical_shape(), x->get_value().logical_shape());
}
