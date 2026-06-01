// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "optimizers/muon.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "core/tt_tensor_utils.hpp"
#include "modules/linear_module.hpp"
#include "ops/losses.hpp"

class MuonFullTest : public ::testing::Test {
protected:
    void SetUp() override {
        ttml::autograd::ctx().open_device();
    }

    void TearDown() override {
        ttml::autograd::ctx().reset_graph();
        ttml::autograd::ctx().close_device();
    }
};

// Muon should drive a simple regression loss down. The 64->16 weight is stored
// as [1, 1, 16, 64], a genuine 2D matrix, so it exercises the Newton-Schulz
// path; the 1D bias exercises the non-matrix momentum fallback.
TEST_F(MuonFullTest, MuonConvergence) {
    using namespace ttml::ops;
    auto* device = &ttml::autograd::ctx().get_device();
    const size_t batch_size = 32;
    const size_t num_features = 64;
    const size_t num_out = 16;

    std::vector<float> features;
    features.reserve(batch_size * num_features);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            features.push_back(static_cast<float>(i) * 0.1F);
        }
    }

    std::vector<float> targets;
    targets.reserve(batch_size * num_out);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < num_out; ++j) {
            targets.push_back(static_cast<float>(i) * 0.1F);
        }
    }

    auto data_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(features, ttnn::Shape({batch_size, 1, 1, num_features}), device));
    auto targets_tensor = ttml::autograd::create_tensor(
        ttml::core::from_vector(targets, ttnn::Shape({batch_size, 1, 1, num_out}), device));

    auto model = ttml::modules::LinearLayer(num_features, num_out);
    auto muon_config = ttml::optimizers::MuonConfig();
    muon_config.lr = 1e-2F;
    muon_config.weight_decay = 0.F;
    auto optimizer = ttml::optimizers::Muon(model.parameters(), muon_config);

    const size_t steps = 100;
    std::vector<float> losses;
    losses.reserve(steps);
    for (size_t step = 0; step < steps; ++step) {
        optimizer.zero_grad();
        auto prediction = model(data_tensor);
        auto loss = ttml::ops::mse_loss(prediction, targets_tensor);
        losses.emplace_back(ttml::core::to_vector(loss->get_value())[0]);
        loss->backward();
        optimizer.step();
        ttml::autograd::ctx().reset_graph();
    }

    EXPECT_LT(losses.back(), losses.front());
    EXPECT_LT(losses.back(), 1e-2F);
}

// With fallback disabled, handing Muon a 1D parameter must fail fast rather than
// silently skipping orthogonalization.
TEST_F(MuonFullTest, NonMatrixWithoutFallbackThrows) {
    auto* device = &ttml::autograd::ctx().get_device();
    ttml::serialization::NamedParameters params;
    params.emplace("bias", ttml::autograd::create_tensor(ttml::core::zeros(ttnn::Shape({1, 1, 1, 8}), device)));

    auto config = ttml::optimizers::MuonConfig();
    config.fallback_on_non_matrix = false;
    auto optimizer = ttml::optimizers::Muon(params, config);

    // Seed a gradient so step() actually processes the parameter.
    params.at("bias")->set_grad(ttml::core::ones(ttnn::Shape({1, 1, 1, 8}), device));
    EXPECT_THROW(optimizer.step(), std::runtime_error);
}
