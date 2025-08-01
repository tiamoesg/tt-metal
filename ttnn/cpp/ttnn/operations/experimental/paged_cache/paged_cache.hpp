// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental::paged_cache {

struct PagedUpdateCacheOperation {
    static ttnn::Tensor invoke(
        const Tensor& cache_tensor,
        const Tensor& input_tensor,
        const std::vector<uint32_t>& update_idxs,
        const std::optional<const Tensor>& update_idxs_tensor,
        std::optional<bool> share_cache,
        const std::optional<const Tensor>& page_table,
        uint32_t batch_offset,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
        std::optional<const std::set<ttnn::MeshCoordinate>> mesh_coords);
};

struct PagedFusedUpdateCacheOperation {
    static std::tuple<ttnn::Tensor, ttnn::Tensor> invoke(
        const Tensor& cache_tensor1,
        const Tensor& input_tensor1,
        const Tensor& cache_tensor2,
        const Tensor& input_tensor2,
        const std::vector<uint32_t>& update_idxs,
        const std::optional<const Tensor>& update_idxs_tensor,
        std::optional<bool> share_cache,
        const std::optional<const Tensor>& page_table,
        uint32_t batch_offset,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
        std::optional<const std::set<ttnn::MeshCoordinate>> mesh_coords);
};

struct PagedFillCacheOperation {
    static ttnn::Tensor invoke(
        const Tensor& cache_tensor,
        const Tensor& input_tensor,
        const Tensor& page_table,
        const std::optional<const Tensor>& batch_idx_tensor,
        uint32_t batch_idx_fallback,
        std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
        std::optional<const std::set<ttnn::MeshCoordinate>> mesh_coords);
};

}  // namespace operations::experimental::paged_cache

namespace experimental {

constexpr auto paged_update_cache = ttnn::register_operation<
    "ttnn::experimental::paged_update_cache",
    ttnn::operations::experimental::paged_cache::PagedUpdateCacheOperation>();

constexpr auto paged_fused_update_cache = ttnn::register_operation<
    "ttnn::experimental::paged_fused_update_cache",
    ttnn::operations::experimental::paged_cache::PagedFusedUpdateCacheOperation>();

constexpr auto paged_fill_cache = ttnn::register_operation<
    "ttnn::experimental::paged_fill_cache",
    ttnn::operations::experimental::paged_cache::PagedFillCacheOperation>();

}  // namespace experimental

}  // namespace ttnn
