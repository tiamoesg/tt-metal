// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/reshape_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr reshape(const autograd::TensorPtr& tensor, const ttnn::Shape& new_shape) {
    auto original_shape = tensor->get_value().logical_shape();
    auto out = autograd::create_tensor(ttnn::reshape(tensor->get_value(), new_shape));
    autograd::GradFunction grad = [tensor, out, original_shape]() {
        tensor->add_grad(ttnn::reshape(out->get_grad(), original_shape));
    };
    auto links = autograd::get_links(tensor);
    out->set_node(autograd::ctx().add_backward_node(std::move(grad), links));
    return out;
}

}  // namespace ttml::ops
