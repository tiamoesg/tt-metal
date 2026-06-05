// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attention_masks.hpp"

#include <vector>

#include "core/tt_tensor_utils.hpp"

namespace ttml::ops {

tt::tt_metal::Tensor compressed_causal_keep(
    uint32_t seq, uint32_t groups, uint32_t rate, ttnn::distributed::MeshDevice* device) {
    std::vector<float> data(static_cast<size_t>(seq) * groups, 0.0F);
    for (uint32_t t = 0; t < seq; ++t) {
        // Reference (model.py): a query at position t sees compressed block s iff
        // s < (t + 1) / rate -- i.e. all blocks that have fully completed by t.
        const uint32_t allowed = (t + 1U) / rate;
        for (uint32_t s = 0; s < groups; ++s) {
            data[static_cast<size_t>(t) * groups + s] = (s < allowed) ? 1.0F : 0.0F;
        }
    }
    return core::from_vector(data, ttnn::Shape({1, 1, seq, groups}), device);
}

tt::tt_metal::Tensor sliding_window_keep(uint32_t seq, uint32_t window, ttnn::distributed::MeshDevice* device) {
    std::vector<float> data(static_cast<size_t>(seq) * seq, 0.0F);
    for (uint32_t t = 0; t < seq; ++t) {
        const uint32_t lo = (t + 1U > window) ? (t + 1U - window) : 0U;  // first visible token
        for (uint32_t j = lo; j <= t; ++j) {
            data[static_cast<size_t>(t) * seq + j] = 1.0F;
        }
    }
    return core::from_vector(data, ttnn::Shape({1, 1, seq, seq}), device);
}

}  // namespace ttml::ops
