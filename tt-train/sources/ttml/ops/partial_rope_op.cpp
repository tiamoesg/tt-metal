// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "partial_rope_op.hpp"

#include <fmt/format.h>

#include <stdexcept>

#include "ops/concat_op.hpp"
#include "ops/slice_op.hpp"

namespace ttml::ops {

autograd::TensorPtr partial_rope(
    const autograd::TensorPtr& input, const RotaryEmbeddingParams& rope_params, uint32_t token_position) {
    const auto shape = input->get_value().logical_shape().to_array_4D();
    const uint32_t full = shape[3];
    const uint32_t rope_dim = rope_params.head_dim;

    if (rope_dim > full) {
        throw std::invalid_argument(
            fmt::format("partial_rope: rope dim {} exceeds the input's last dimension {}.", rope_dim, full));
    }
    if (rope_dim == full) {
        return rope(input, rope_params, token_position);  // full RoPE
    }

    const uint32_t nope = full - rope_dim;
    auto nope_part = ops::slice(input, {0U, 0U, 0U, 0U}, {shape[0], shape[1], shape[2], nope});
    auto rope_part = ops::slice(input, {0U, 0U, 0U, nope}, {shape[0], shape[1], shape[2], full});
    auto rotated = rope(rope_part, rope_params, token_position);
    return ops::concat({nope_part, rotated}, /* dim */ 3);
}

}  // namespace ttml::ops
