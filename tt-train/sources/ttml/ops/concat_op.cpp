// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_op.hpp"

#include <core/ttnn_all_includes.hpp>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/graph_utils.hpp"

namespace ttml::ops {

autograd::TensorPtr concat(const std::vector<autograd::TensorPtr>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("ops::concat requires at least one tensor.");
    }
    const int axis = (dim < 0) ? dim + 4 : dim;

    std::vector<ttnn::Tensor> raw;
    std::vector<uint32_t> sizes;
    raw.reserve(tensors.size());
    sizes.reserve(tensors.size());
    for (const auto& t : tensors) {
        raw.push_back(t->get_value());
        sizes.push_back(t->get_value().logical_shape().to_array_4D()[axis]);
    }

    auto out = autograd::create_tensor(ttnn::concat(raw, axis));

    autograd::GradFunction grad = [tensors, out, sizes, axis]() {
        const auto g = out->get_grad();
        const auto gshape = g.logical_shape().to_array_4D();
        uint32_t offset = 0;
        for (size_t i = 0; i < tensors.size(); ++i) {
            ttsl::SmallVector<uint32_t> start = {0U, 0U, 0U, 0U};
            ttsl::SmallVector<uint32_t> end = {gshape[0], gshape[1], gshape[2], gshape[3]};
            const ttsl::SmallVector<uint32_t> step = {1U, 1U, 1U, 1U};
            start[axis] = offset;
            end[axis] = offset + sizes[i];
            tensors[i]->add_grad(ttnn::slice(g, start, end, step));
            offset += sizes[i];
        }
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensors));
    return out;
}

}  // namespace ttml::ops
