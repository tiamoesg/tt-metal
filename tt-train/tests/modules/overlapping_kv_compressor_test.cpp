// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/overlapping_kv_compressor.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <core/ttnn_all_includes.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class OverlappingKVCompressorTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// [B,1,n,d] -> [B,1,n/m,c]; all four projections and both positional biases get
// gradients, and the output is finite (the -inf padding of entry 0's previous
// segment must not produce NaNs).
TEST_F(OverlappingKVCompressorTest, ForwardShapeAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t batch = 1;
    const uint32_t seq = 64;
    const uint32_t dim = 64;
    const uint32_t compressed_dim = 32;
    const uint32_t rate = 32;  // n/m = 2 compressed entries

    ttml::modules::OverlappingKVCompressorConfig config;
    config.dim = dim;
    config.compressed_dim = compressed_dim;
    config.compression_rate = rate;
    auto compressor = ttml::modules::OverlappingKVCompressor(config);

    auto hidden = ttml::autograd::create_tensor(
        ttml::core::ones(ttnn::Shape({batch, 1, seq, dim}), device), /* requires_grad */ true);

    auto out = compressor(hidden);
    const auto shape = out->get_value().logical_shape();
    EXPECT_EQ(shape[0], batch);
    EXPECT_EQ(shape[1], 1U);
    EXPECT_EQ(shape[2], seq / rate);
    EXPECT_EQ(shape[3], compressed_dim);

    for (float v : ttml::core::to_vector(out->get_value())) {
        EXPECT_TRUE(std::isfinite(v));
    }

    auto target = ttml::autograd::create_tensor(ttml::core::zeros(shape, device));
    auto loss = ttml::ops::mse_loss(out, target);
    loss->backward();

    auto params = compressor.parameters();
    EXPECT_TRUE(params.at("overlapping_kv_compressor/w_akv/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("overlapping_kv_compressor/w_bkv/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("overlapping_kv_compressor/w_az/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("overlapping_kv_compressor/w_bz/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("overlapping_kv_compressor/bias_a")->is_grad_initialized());
    EXPECT_TRUE(params.at("overlapping_kv_compressor/bias_b")->is_grad_initialized());
}
