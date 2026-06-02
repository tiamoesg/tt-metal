// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/permute_op.hpp"

#include <core/ttnn_all_includes.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"

namespace ttml::ops {

autograd::TensorPtr permute(const autograd::TensorPtr& tensor, const ttnn::SmallVector<int64_t>& dims) {
    auto out = autograd::create_tensor(ttnn::permute(tensor->get_value(), dims));

    // Inverse permutation: if forward moves axis dims[i] to position i, the
    // backward moves it back.
    ttnn::SmallVector<int64_t> inverse_dims(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        inverse_dims[static_cast<size_t>(dims[i])] = static_cast<int64_t>(i);
    }

    autograd::GradFunction grad = [tensor, out, inverse_dims]() {
        tensor->add_grad(ttnn::permute(out->get_grad(), inverse_dims));
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

}  // namespace ttml::ops
