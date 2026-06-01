// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/manifold_hyper_connections.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <memory>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"

class ManifoldHyperConnectionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// Forward must preserve the expanded residual shape [B, n_hc, S, d], and the
// backward must populate gradients for the raw mHC parameters.
TEST_F(ManifoldHyperConnectionsTest, ForwardShapeAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t batch = 2;
    const uint32_t n_hc = 4;
    const uint32_t seq = 32;
    const uint32_t dim = 64;

    ttml::modules::ManifoldHyperConnectionsConfig config;
    config.num_streams = n_hc;
    config.hidden_dim = dim;
    config.sinkhorn_iters = 10;

    auto inner = std::make_shared<ttml::modules::LinearLayer>(dim, dim);
    auto mhc = ttml::modules::ManifoldHyperConnections(config, inner);

    auto residual = ttml::autograd::create_tensor(
        ttml::core::ones(ttnn::Shape({batch, n_hc, seq, dim}), device), /* requires_grad */ true);

    auto out = mhc(residual);
    EXPECT_EQ(out->get_value().logical_shape(), residual->get_value().logical_shape());

    auto loss = ttml::ops::mse_loss(out, residual);
    loss->backward();

    auto params = mhc.parameters();
    EXPECT_TRUE(params.at("manifold_hyper_connections/raw_a")->is_grad_initialized());
    EXPECT_TRUE(params.at("manifold_hyper_connections/raw_b")->is_grad_initialized());
    EXPECT_TRUE(params.at("manifold_hyper_connections/raw_c")->is_grad_initialized());
}
