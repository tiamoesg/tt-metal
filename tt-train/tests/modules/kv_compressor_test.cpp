// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "modules/kv_compressor.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "ops/losses.hpp"

class KVCompressorTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// The HCA compressor maps [B, 1, n, d] -> [B, 1, n/m', c] and its parameters
// (W^KV, W^Z, positional bias) must receive gradients.
TEST_F(KVCompressorTest, ForwardShapeAndBackward) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t batch = 2;
    const uint32_t seq = 32;
    const uint32_t dim = 16;
    const uint32_t compressed_dim = 8;
    const uint32_t rate = 4;  // n/m' = 8 compressed entries

    ttml::modules::KVCompressorConfig config;
    config.dim = dim;
    config.compressed_dim = compressed_dim;
    config.compression_rate = rate;
    auto compressor = ttml::modules::KVCompressor(config);

    auto hidden = ttml::autograd::create_tensor(
        ttml::core::ones(ttnn::Shape({batch, 1, seq, dim}), device), /* requires_grad */ true);

    auto out = compressor(hidden);
    const auto out_shape = out->get_value().logical_shape();
    EXPECT_EQ(out_shape[0], batch);
    EXPECT_EQ(out_shape[1], 1U);
    EXPECT_EQ(out_shape[2], seq / rate);
    EXPECT_EQ(out_shape[3], compressed_dim);

    auto target = ttml::autograd::create_tensor(ttml::core::zeros(out_shape, device));
    auto loss = ttml::ops::mse_loss(out, target);
    loss->backward();

    auto params = compressor.parameters();
    EXPECT_TRUE(params.at("kv_compressor/w_kv/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("kv_compressor/w_z/weight")->is_grad_initialized());
    EXPECT_TRUE(params.at("kv_compressor/pos_bias")->is_grad_initialized());
}

// With zero positional bias and constant input, each compressed entry is the
// uniform (softmax-of-equal-logits) average over its segment. Two segments fed
// identical data must produce identical compressed entries.
TEST_F(KVCompressorTest, UniformPoolingWithZeroBias) {
    auto* device = &ttml::autograd::ctx().get_device();
    const uint32_t batch = 1;
    const uint32_t seq = 8;
    const uint32_t dim = 4;
    const uint32_t compressed_dim = 4;
    const uint32_t rate = 4;  // 2 compressed entries

    ttml::modules::KVCompressorConfig config;
    config.dim = dim;
    config.compressed_dim = compressed_dim;
    config.compression_rate = rate;
    auto compressor = ttml::modules::KVCompressor(config);

    auto hidden = ttml::autograd::create_tensor(ttml::core::ones(ttnn::Shape({batch, 1, seq, dim}), device));
    auto out = compressor(hidden);
    auto values = ttml::core::to_vector(out->get_value());  // 2 entries x compressed_dim

    // Both compressed entries see identical (constant) input, so they must match.
    for (uint32_t j = 0; j < compressed_dim; ++j) {
        EXPECT_NEAR(values[j], values[compressed_dim + j], 1e-2F);
    }
}
