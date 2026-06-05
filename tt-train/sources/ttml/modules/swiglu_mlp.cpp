// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "swiglu_mlp.hpp"

#include <stdexcept>

#include "ops/binary_ops.hpp"
#include "ops/unary_ops.hpp"

namespace ttml::modules {

SwiGLUMLP::SwiGLUMLP(const SwiGLUMLPConfig& config) {
    if (config.dim == 0U || config.inter_dim == 0U) {
        throw std::invalid_argument("SwiGLUMLPConfig: dim and inter_dim must be set.");
    }
    create_name("swiglu_mlp");
    m_w1 = std::make_shared<LinearLayer>(config.dim, config.inter_dim, /* has_bias */ false);
    m_w2 = std::make_shared<LinearLayer>(config.inter_dim, config.dim, /* has_bias */ false);
    m_w3 = std::make_shared<LinearLayer>(config.dim, config.inter_dim, /* has_bias */ false);
    register_module(m_w1, "w1");
    register_module(m_w2, "w2");
    register_module(m_w3, "w3");
}

autograd::TensorPtr SwiGLUMLP::operator()(const autograd::TensorPtr& x) {
    // w2( silu(w1(x)) * w3(x) )
    return (*m_w2)(ops::mul(ops::silu((*m_w1)(x)), (*m_w3)(x)));
}

}  // namespace ttml::modules
