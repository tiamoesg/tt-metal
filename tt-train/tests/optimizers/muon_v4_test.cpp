// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/muon_v4.hpp"

#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>
#include <vector>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"

class MuonV4Test : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }
    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// Muon-V4 should drive a simple regression loss down. The 64->16 weight is a
// genuine 2D matrix, exercising the hybrid Newton-Schulz path; the 1D bias
// exercises the non-matrix momentum fallback.
TEST_F(MuonV4Test, Convergence) {
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 32;
    const size_t num_features = 64;
    const size_t num_out = 16;

    std::vector<float> features(batch_size * num_features);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            features[i * num_features + j] = static_cast<float>(i) * 0.1F;
        }
    }
    std::vector<float> targets(batch_size * num_out);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_out; ++j) {
            targets[i * num_out + j] = static_cast<float>(i) * 0.1F;
        }
    }

    auto data = ttml::autograd::create_tensor(
        ttml::core::from_vector(features, ttnn::Shape({batch_size, 1, 1, num_features}), device));
    auto target = ttml::autograd::create_tensor(
        ttml::core::from_vector(targets, ttnn::Shape({batch_size, 1, 1, num_out}), device));

    auto model = ttml::modules::LinearLayer(num_features, num_out);
    ttml::optimizers::MuonV4Config config;
    config.lr = 1e-2F;
    auto optimizer = ttml::optimizers::MuonV4(model.parameters(), config);

    std::vector<float> losses;
    const size_t steps = 100;
    losses.reserve(steps);
    for (size_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto prediction = model(data);
        auto loss = ttml::ops::mse_loss(prediction, target);
        losses.push_back(ttml::core::to_vector(loss->get_value())[0]);
        loss->backward();
        optimizer.step();
        ttml::autograd::ctx().reset_graph();
    }

    EXPECT_LT(losses.back(), losses.front());
    EXPECT_LT(losses.back(), 1e-2F);
}

// With the fallback disabled, a 1D parameter must fail fast rather than silently
// skipping orthogonalization.
TEST_F(MuonV4Test, NonMatrixWithoutFallbackThrows) {
    auto* device = &ttml::autograd::ctx().get_device();
    ttml::serialization::NamedParameters params;
    params.emplace("bias", ttml::autograd::create_tensor(ttml::core::zeros(ttnn::Shape({1, 1, 1, 8}), device)));

    ttml::optimizers::MuonV4Config config;
    config.fallback_on_non_matrix = false;
    auto optimizer = ttml::optimizers::MuonV4(params, config);

    params.at("bias")->set_grad(ttml::core::ones(ttnn::Shape({1, 1, 1, 8}), device));
    EXPECT_THROW(optimizer.step(), std::runtime_error);
}
