// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"

namespace ttml::ops {

autograd::TensorPtr slice(
    const autograd::TensorPtr& tensor, const std::array<uint32_t, 4>& start, const std::array<uint32_t, 4>& end) {
    const ttsl::SmallVector<uint32_t> start_v(start.begin(), start.end());
    const ttsl::SmallVector<uint32_t> end_v(end.begin(), end.end());
    const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};

    auto out = autograd::create_tensor(ttnn::slice(tensor->get_value(), start_v, end_v, step));
    const auto input_shape = tensor->get_value().logical_shape().to_array_4D();

    autograd::GradFunction grad = [tensor, out, start, input_shape]() {
        // Pad the upstream gradient back to the input shape: before = start[d],
        // after = input_shape[d] - (start[d] + grad_extent[d]).
        const auto g = out->get_grad().logical_shape().to_array_4D();
        const ttsl::SmallVector<ttnn::operations::data_movement::PadSpecDim> padding = {
            {start[0], input_shape[0] - (start[0] + g[0])},
            {start[1], input_shape[1] - (start[1] + g[1])},
            {start[2], input_shape[2] - (start[2] + g[2])},
            {start[3], input_shape[3] - (start[3] + g[3])},
        };
        tensor->add_grad(
            ttnn::pad(out->get_grad(), padding, /* value */ 0.0F, /* use_multicore */ false, std::nullopt));
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

}  // namespace ttml::ops
