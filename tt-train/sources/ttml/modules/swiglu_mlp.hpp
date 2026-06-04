// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "autograd/tensor.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"

namespace ttml::modules {

// SwiGLU feed-forward network: w2( silu(w1(x)) * w3(x) ), the dense FFN used by
// DeepSeek (the per-expert and shared-expert MLP). Bias-free, matching the V4
// reference Expert.
struct SwiGLUMLPConfig {
    uint32_t dim{0};        // hidden size
    uint32_t inter_dim{0};  // intermediate (expansion) size
};

class SwiGLUMLP : public ModuleBase {
public:
    explicit SwiGLUMLP(const SwiGLUMLPConfig& config);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& x) override;

private:
    std::shared_ptr<LinearLayer> m_w1;  // dim -> inter
    std::shared_ptr<LinearLayer> m_w2;  // inter -> dim
    std::shared_ptr<LinearLayer> m_w3;  // dim -> inter
};

}  // namespace ttml::modules
