// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_scatter_minimal_async_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/reduce_scatter_minimal_async.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_reduce_scatter_minimal_async(pybind11::module& module, const ccl_operation_t& operation, const char* doc) {
    // namespace py = pybind11;

    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const std::optional<std::vector<ttnn::Tensor>>& persistent_output_buffers,
               const int32_t dim,
               const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
               const std::optional<GlobalSemaphore>& barrier_semaphore,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id,
               std::optional<uint32_t> cluster_axis,
               std::optional<uint32_t> chunks_per_sync,
               std::optional<uint32_t> num_workers_per_link,
               std::optional<uint32_t> num_buffers_per_channel) -> ttnn::Tensor {
                return self(
                    input_tensor,
                    persistent_output_buffers,
                    dim,
                    multi_device_global_semaphore,
                    barrier_semaphore,
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id,
                    cluster_axis,
                    chunks_per_sync,
                    num_workers_per_link,
                    num_buffers_per_channel);
            },
            py::arg("input_tensor"),
            py::arg("persistent_output_buffers") = std::nullopt,
            py::arg("dim"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("barrier_semaphore") = std::nullopt,
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt,
            py::arg("cluster_axis") = std::nullopt,
            py::arg("chunks_per_sync") = std::nullopt,
            py::arg("num_workers_per_link") = std::nullopt,
            py::arg("num_buffers_per_channel") = std::nullopt});
}

}  // namespace detail

void py_bind_reduce_scatter_minimal_async(pybind11::module& module) {
    detail::bind_reduce_scatter_minimal_async(
        module,
        ttnn::experimental::reduce_scatter_minimal_async,
        R"doc(

        Performs an reduce-scatter operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            input_tensor (ttnn.Tensor): multi-device tensor.
            dim (int): Dimension to scatter.
            mesh_device (MeshDevice): Device mesh to perform the line-all-gather operation on.

        Mesh Tensor Programming Guide : https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Programming_Mesh_of_Devices/Programming_Mesh_of_Devices_with_TT-NN.md

        Keyword Args:
            num_links (int, optional): Number of links to use for the all-gather operation. Defaults to `1`.
            memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `input tensor memory config`.
            topology (ttnn.Topology, optional): The topology configuration to run the operation in. Valid options are Ring and Linear. Defaults to `ttnn.Topology.Ring`.

        Returns:
            ttnn.Tensor: the output tensor.

        Example:

        )doc");
}

}  // namespace ttnn::operations::experimental::ccl
