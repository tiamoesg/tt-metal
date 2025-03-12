# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Optional, List
import ttnn
import os
import multiprocessing as mp
import threading

# Enforce multiprocessing "spawn" mode to prevent fork-related issues
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Ignore if already set

# Global lock to ensure safe parallel device access
_device_lock = threading.Lock()

def is_forked_process():
    """Detect if this process was forked from a parent process."""
    return mp.parent_process() is not None and mp.get_start_method() == "fork"

# Wrap open_device to ensure thread/process safety
def open_device(device_id=0):
    """
    Open the device safely, allowing true parallel execution.

    - Uses a lock to prevent race conditions.
    - Supports multiprocessing without fork-related issues.
    - Ensures independent initialization per process.
    """
    if is_forked_process():
        raise RuntimeError(
            "ttnn.open_device() cannot be called in a forked process. "
            "Use multiprocessing.set_start_method('spawn') instead."
        )
    
    with _device_lock:  # Ensure thread/process safety
        return ttnn._ttnn.device.open_device(device_id)

def get_device_core_grid(device):
    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)

# TODO: Device = ttnn._ttnn.Device
Device = ttnn._ttnn.device.Device
Device.core_grid = property(get_device_core_grid)
DispatchCoreType = ttnn._ttnn.device.DispatchCoreType
DispatchCoreAxis = ttnn._ttnn.device.DispatchCoreAxis
DispatchCoreConfig = ttnn._ttnn.device.DispatchCoreConfig
Arch = ttnn._ttnn.device.Arch
DEFAULT_L1_SMALL_SIZE = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE
DEFAULT_TRACE_REGION_SIZE = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE

init_device_compute_kernel_config = ttnn._ttnn.operations.core.init_device_compute_kernel_config

def close_device(device: "ttnn.device.Device"):
    """
    Close the device and remove it from the device cache.
    """
    synchronize_device(device)
    ttnn._ttnn.device.close_device(device)

enable_program_cache = ttnn._ttnn.device.enable_program_cache
disable_and_clear_program_cache = ttnn._ttnn.device.disable_and_clear_program_cache
synchronize_device = ttnn._ttnn.device.synchronize_device
synchronize_mesh_device = ttnn._ttnn.device.synchronize_mesh_device
GetDefaultDevice = ttnn._ttnn.device.GetDefaultDevice
SetDefaultDevice = ttnn._ttnn.device.SetDefaultDevice
GetPCIeDeviceID = ttnn._ttnn.device.GetPCIeDeviceID
GetNumPCIeDevices = ttnn._ttnn.device.GetNumPCIeDevices

def CreateDevice(device_id: int, num_command_queues: int = 1, l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
                 trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE, dispatch_core_config: DispatchCoreConfig = ttnn._ttnn.device.DispatchCoreConfig()):
    return ttnn._ttnn.device.CreateDevice(device_id, num_command_queues, l1_small_size, trace_region_size, dispatch_core_config)

def CreateDevices(device_ids: List[int], num_command_queues: int = 1, l1_small_size: int = ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
                  trace_region_size: int = ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE, dispatch_core_config: DispatchCoreConfig = ttnn._ttnn.device.DispatchCoreConfig()):
    return ttnn._ttnn.device.CreateDevices(device_ids, num_command_queues, l1_small_size, trace_region_size, dispatch_core_config)

CloseDevice = ttnn._ttnn.device.CloseDevice
CloseDevices = ttnn._ttnn.device.CloseDevices

def DumpDeviceProfiler(device):
    ttnn._ttnn.device.DumpDeviceProfiler(device)

GetNumAvailableDevices = ttnn._ttnn.device.GetNumAvailableDevices
EnablePersistentKernelCache = ttnn._ttnn.device.EnablePersistentKernelCache
DisablePersistentKernelCache = ttnn._ttnn.device.DisablePersistentKernelCache
EnableMemoryReports = ttnn._ttnn.device.EnableMemoryReports
DisableMemoryReports = ttnn._ttnn.device.DisableMemoryReports
DeallocateBuffers = ttnn._ttnn.device.deallocate_buffers

@contextlib.contextmanager
def manage_device(device_id: int) -> "ttnn.device.Device":
    """
    Context manager for opening and closing a device.
    """
    device = open_device(device_id=device_id)
    try:
        yield device
    finally:
        close_device(device)

def dump_device_memory_state(device, prefix=""):
    ttnn._ttnn.device.DumpDeviceMemoryState(device, prefix)

def get_memory_view(device, buffer_type):
    return ttnn._ttnn.device.GetMemoryView(device, buffer_type)

def is_wormhole_b0(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.WORMHOLE_B0
    ARCH_NAME = ttnn.get_arch_name()
    return "wormhole_b0" in ARCH_NAME

def is_grayskull(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.GRAYSKULL
    ARCH_NAME = ttnn.get_arch_name()
    return "grayskull" in ARCH_NAME

def is_blackhole(device=None):
    if device is not None:
        return device.arch() == ttnn._ttnn.device.Arch.BLACKHOLE
    ARCH_NAME = ttnn.get_arch_name()
    return "blackhole" in ARCH_NAME

SetDefaultDevice = ttnn._ttnn.device.SetDefaultDevice
GetDefaultDevice = ttnn._ttnn.device.GetDefaultDevice
format_input_tensor = ttnn._ttnn.device.format_input_tensor
format_output_tensor = ttnn._ttnn.device.format_output_tensor
pad_to_tile_shape = ttnn._ttnn.device.pad_to_tile_shape

SubDevice = ttnn._ttnn.device.SubDevice
SubDeviceId = ttnn._ttnn.device.SubDeviceId
SubDeviceManagerId = ttnn._ttnn.device.SubDeviceManagerId

DefaultQueueId = ttnn._ttnn.device.DefaultQueueId

__all__ = []
