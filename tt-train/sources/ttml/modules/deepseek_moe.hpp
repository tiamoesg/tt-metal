// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "autograd/tensor.hpp"
#include "modules/linear_module.hpp"
#include "modules/module_base.hpp"
#include "modules/swiglu_mlp.hpp"

namespace ttml::modules {

// DeepSeekMoE feed-forward (model.py Gate + Expert + MoE), the routed alternative
// to a dense SwiGLU FFN. A gate scores the experts per token, the top-k routed
// experts run (weighted by their normalized scores) and a shared expert always
// runs:
//
//   scores = score_func( x W_gate )                       [.., E]
//   keep   = top-k(scores) over experts                    (0/1, stop-grad)
//   w      = scores * keep ; (non-softmax) w /= sum(w) ; w *= route_scale
//   y      = shared(x) + sum_e w_e * expert_e(x)
//
// Execution uses dense masking (every routed expert runs on every token, scaled
// by its routing weight, which is 0 for non-selected tokens) -- the strategy
// ttml's Python DeepSeek MoE uses; correct and fully differentiable, O(E) compute.
// `score_func = SqrtSoftplus` is the DeepSeek-V4 default. ttnn::topk requires the
// expert count to be a power of two (pad otherwise).
enum class MoEScoreFunc : uint8_t { SqrtSoftplus = 0, Softmax = 1, Sigmoid = 2 };

struct DeepSeekMoEConfig {
    uint32_t dim{0};
    uint32_t inter_dim{0};           // per-expert SwiGLU intermediate size
    uint32_t num_routed_experts{0};  // E (power of two for ttnn::topk)
    uint32_t num_activated{0};       // top-k experts per token
    uint32_t num_shared_experts{1};  // always-on shared experts
    MoEScoreFunc score_func{MoEScoreFunc::SqrtSoftplus};
    float route_scale{1.0F};
};

class DeepSeekMoE : public ModuleBase {
public:
    explicit DeepSeekMoE(const DeepSeekMoEConfig& config);

    [[nodiscard]] autograd::TensorPtr operator()(const autograd::TensorPtr& x) override;

private:
    DeepSeekMoEConfig m_config;
    std::shared_ptr<LinearLayer> m_gate;  // dim -> E
    std::vector<std::shared_ptr<SwiGLUMLP>> m_experts;
    std::vector<std::shared_ptr<SwiGLUMLP>> m_shared;
};

}  // namespace ttml::modules
