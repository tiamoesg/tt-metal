// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ops/unary_ops.hpp"

#include <array>
#include <optional>
#include <stdexcept>

#include "autograd/auto_context.hpp"
#include "autograd/graph.hpp"
#include "autograd/graph_utils.hpp"
#include "autograd/tensor.hpp"
#include "core/compute_kernel_config.hpp"
#include "core/tt_tensor_utils.hpp"
#include "metal/operations.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/experimental/unary_backward/gelu_backward/gelu_backward.hpp"
#include "ttnn/operations/moreh/moreh_mean/moreh_mean.hpp"
#include "ttnn/operations/moreh/moreh_mean_backward/moreh_mean_backward.hpp"
#include "ttnn/operations/moreh/moreh_softmax/moreh_softmax.hpp"
#include "ttnn/operations/moreh/moreh_softmax_backward/moreh_softmax_backward.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn_fixed/trivial_ttnn_ops.hpp"

namespace ttml::ops {

autograd::TensorPtr relu(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::relu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        tt::tt_metal::MemoryConfig mem_config;
        auto res = ttnn::relu_bw(out->get_grad(), tensor->get_value(), mem_config);
        tensor->add_grad(res[0]);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));

    return out;
}

autograd::TensorPtr gelu(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::gelu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        static const std::string approx_mode = "none";
        auto dL_dt = ttnn::experimental::gelu_bw(out->get_grad(), tensor->get_value(), approx_mode);
        tensor->add_grad(dL_dt);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr silu(const autograd::TensorPtr& tensor, bool use_composite_bw) {
    auto out = autograd::create_tensor(ttnn::silu(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out, use_composite_bw]() {
        auto res = use_composite_bw ? ttnn::silu_bw(out->get_grad(), tensor->get_value())
                                    : std::vector<std::optional<ttnn::Tensor>>(
                                          {ttml::metal::silu_bw(tensor->get_value(), out->get_grad())});
        assert(res.size() == 1U && "Silu backward should return only one gradient");
        tensor->add_grad(res.front().value());
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));

    return out;
}

autograd::TensorPtr log_softmax(const autograd::TensorPtr& tensor, int dim) {
    auto log_softmax = ttnn_fixed::log_softmax(tensor->get_value(), dim);
    auto out = autograd::create_tensor(log_softmax);
    autograd::GradFunction grad = [tensor, out, dim]() {
        auto softmax = ttnn::exp(out->get_value());
        auto sum_grad_over_dim = ttnn_fixed::sum_over_dim(out->get_grad(), dim);
        auto grad = ttnn::subtract(out->get_grad(), ttnn::multiply(softmax, sum_grad_over_dim));
        tensor->add_grad(grad);
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr log_softmax_moreh(const autograd::TensorPtr& tensor, int dim) {
    auto log_softmax = ttnn::moreh_softmax(
        tensor->get_value(),
        /* axis */ dim,
        /* output */ std::nullopt,
        ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOp::LOGSOFTMAX,
        ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy::NONE,
        /* output_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::softmax());
    auto out = autograd::create_tensor(log_softmax);

    autograd::GradFunction grad = [tensor, out, dim]() {
        auto grad = ttnn::moreh_softmax_backward(
            out->get_value(),
            out->get_grad(),
            /* axis */ dim,
            /* output */ std::nullopt,
            ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOp::LOGSOFTMAX,
            ttnn::operations::moreh::moreh_softmax_backward::MorehSoftmaxBackwardOpParallelizationStrategy::NONE,
            /* output_mem_config */ std::nullopt,
            /* compute_kernel_config */ core::ComputeKernelConfig::precise());
        tensor->add_grad(grad);
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr mean(const autograd::TensorPtr& tensor) {
    auto shape = ttnn::Shape({1, 1, 1, 1});
    auto out =
        autograd::create_tensor(core::empty(shape, &autograd::ctx().get_device(), tensor->get_value().memory_config()));
    ttnn::moreh_mean(
        tensor->get_value(),
        std::nullopt,
        true,
        std::nullopt,
        out->get_value(),
        std::nullopt,
        /* device_compute_kernel_config */ core::ComputeKernelConfig::precise());
    autograd::GradFunction grad = [tensor, out]() {
        auto resulting_shape = tensor->get_value().logical_shape();
        auto res = ttnn::moreh_mean_backward(
            out->get_grad(),
            std::nullopt,
            false,
            resulting_shape,
            std::nullopt,
            std::nullopt,
            core::ComputeKernelConfig::precise());
        tensor->add_grad(res);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr broadcast_batch(const autograd::TensorPtr& tensor, uint32_t new_batch_dim) {
    if (new_batch_dim == 1 || tensor->get_value().logical_shape()[0] == new_batch_dim) {
        return tensor;
    }
    auto out = ttml::autograd::create_tensor();
    auto repeats = ttnn::Shape({new_batch_dim, 1, 1, 1});
    // currently assuming tensor came with shape: {1,X,Y,Z} and we want to get {B,X,Y,Z}
    out->set_value(ttnn::repeat(tensor->get_value(), repeats));

    autograd::GradFunction grad = [tensor, out]() {
        auto res = ttnn_fixed::sum_over_batch(out->get_grad());
        tensor->add_grad(res);
    };

    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr exp(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::exp(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        auto res = ttnn::exp_bw(out->get_grad(), tensor->get_value());
        tensor->add_grad(res[0].value());
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr clip(const autograd::TensorPtr& tensor, float lo, float hi) {
    auto out = autograd::create_tensor();
    out->set_value(ttnn::clip(tensor->get_value(), lo, hi));
    autograd::GradFunction grad = [tensor, out, lo, hi]() {
        auto res = ttnn::clip_bw(out->get_grad(), tensor->get_value(), lo, hi);
        tensor->add_grad(res[0]);
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr sigmoid(const autograd::TensorPtr& tensor) {
    auto out = autograd::create_tensor(ttnn::sigmoid(tensor->get_value()));
    autograd::GradFunction grad = [tensor, out]() {
        // d/dx sigmoid(x) = s * (1 - s), where s = sigmoid(x) is the output.
        auto s = out->get_value();
        auto one_minus_s = ttnn::add(ttnn::multiply(s, -1.0F), 1.0F);
        auto local_grad = ttnn::multiply(s, one_minus_s);
        tensor->add_grad(ttnn::multiply(out->get_grad(), local_grad));
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr sum(const autograd::TensorPtr& tensor, int dim, bool keep_dim) {
    if (!keep_dim) {
        throw std::runtime_error("ops::sum(tensor, dim, keep_dim) currently supports keep_dim=true only.");
    }
    auto out = autograd::create_tensor(ttnn::sum(
        tensor->get_value(),
        /* dim_arg */ ttnn::SmallVector<int>{dim},
        /* keep_dim */ true,
        /* output_mem_config */ std::nullopt,
        /* compute_kernel_config */ core::ComputeKernelConfig::precise()));
    autograd::GradFunction grad = [tensor, out]() {
        // Backward of a keep_dim sum replicates the reduced (size-1) dimension
        // back to the input shape.
        const auto in = tensor->get_value().logical_shape().to_array_4D();
        const auto g = out->get_grad().logical_shape().to_array_4D();
        auto repeats = ttnn::Shape({in[0] / g[0], in[1] / g[1], in[2] / g[2], in[3] / g[3]});
        tensor->add_grad(ttnn::repeat(out->get_grad(), repeats));
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

autograd::TensorPtr broadcast_to(const autograd::TensorPtr& tensor, const ttnn::Shape& target_shape) {
    auto input_shape = tensor->get_value().logical_shape();
    if (input_shape == target_shape) {
        return tensor;
    }
    const auto in = input_shape.to_array_4D();
    const auto out_arr = target_shape.to_array_4D();
    auto repeats = ttnn::Shape({out_arr[0] / in[0], out_arr[1] / in[1], out_arr[2] / in[2], out_arr[3] / in[3]});

    auto out = autograd::create_tensor(ttnn::repeat(tensor->get_value(), repeats));
    autograd::GradFunction grad = [tensor, out]() {
        auto input_shape = tensor->get_value().logical_shape();
        auto grad_shape = out->get_grad().logical_shape();
        ttnn::SmallVector<int64_t> broadcast_dims;
        for (size_t i = 0; i < input_shape.size(); ++i) {
            if (input_shape[i] != grad_shape[i]) {
                broadcast_dims.push_back(static_cast<int64_t>(i));
            }
        }
        if (broadcast_dims.empty()) {
            tensor->add_grad(out->get_grad());
            return;
        }
        tensor->add_grad(ttnn::moreh_sum(
            out->get_grad(),
            broadcast_dims,
            /* keep_dim */ true,
            /* output_tensor */ std::nullopt,
            /* memory_config_arg */ std::nullopt,
            core::ComputeKernelConfig::precise()));
    };
    out->set_node(autograd::add_backward_node(std::move(grad), out, tensor));
    return out;
}

}  // namespace ttml::ops
