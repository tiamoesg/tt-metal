// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/deepseek_moe.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class DeepSeekMoETest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// MoE preserves [B,1,S,d] and feeds gradients to the gate, the routed experts, and
// the shared expert (gate gradient flows through the differentiable routing weights).
TEST_F(DeepSeekMoETest, ForwardShapeAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t dim = 64;
    const uint32_t seq = 32;

    ttml::modules::DeepSeekMoEConfig config;
    config.dim = dim;
    config.inter_dim = 128;
    config.num_routed_experts = 8;  // power of two for ttnn::topk
    config.num_activated = 2;
    config.num_shared_experts = 1;
    config.score_func = ttml::modules::MoEScoreFunc::SqrtSoftplus;
    config.route_scale = 2.5F;
    auto moe = ttml::modules::DeepSeekMoE(config);

    auto x = ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({1, 1, seq, dim}), device), true);
    auto out = moe(x);
    EXPECT_EQ(out->get_value().logical_shape()[3], dim);
    EXPECT_EQ(out->get_value().logical_shape()[2], seq);

    auto target = ttml::autograd::create_tensor(ttml::core::zeros(out->get_value().logical_shape(), device));
    auto loss = ttml::ops::mse_loss(out, target);
    loss->backward();

    auto params = moe.parameters();
    EXPECT_TRUE(params.at("deepseek_moe/gate/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("deepseek_moe/expert_0/w1/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("deepseek_moe/shared_0/w1/weight")->is_grad_initialized());
    EXPECT_TRUE(x->is_grad_initialized());
}
