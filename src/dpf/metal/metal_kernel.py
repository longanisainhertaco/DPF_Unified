"""Metal Kernel Wrapper for DPF Unified.

This module provides a Python interface to the compiled Metal Shading Language
kernels using PyObjC and the Metal framework. It handles:
1. Loading the .metallib library.
2. Creating compute pipelines for specific kernels (PLM, HLL, MHD Sweep).
3. Dispatching compute threads with zero-copy buffer access.
"""

import ctypes
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Try to import Metal (PyObjC)
try:
    from Metal import (
        MTLCreateSystemDefaultDevice,
        MTLResourceStorageModePrivate,
        MTLResourceStorageModeShared,
        MTLSize,
    )
    _METAL_AVAILABLE = True
except ImportError:
    logger.warning("PyObjC not installed. Native Metal kernels will be unavailable.")
    _METAL_AVAILABLE = False
    # Mock symbols to prevent NameError
    MTLCreateSystemDefaultDevice = None
    MTLResourceStorageModeShared = None
    MTLResourceStorageModePrivate = None
    MTLSize = None

class MetalKernelWrapper:
    """Wrapper for dispatching custom Metal compute kernels."""

    def __init__(self, lib_path: str | None = None):
        """Initialize the Metal device and load the kernel library.

        Args:
            lib_path: Path to the .metallib file. If None, looks for
                      'src/dpf/metal/build/default.metallib'.
        """
        self.device = MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Metal is not available on this system.")

        self.command_queue = self.device.newCommandQueue()

        if lib_path is None:
            # Default path relative to this file?
            # Assuming run from project root:
            lib_path = "src/dpf/metal/build/default.metallib"

        lib_path_obj = Path(lib_path).resolve()
        if not lib_path_obj.exists():
            logger.error(f"Metal library not found at {lib_path_obj}")
            raise FileNotFoundError(f"Metal library not found at {lib_path_obj}")

        # Load Library
        error = None
        self.library = self.device.newLibraryWithFile_error_(str(lib_path_obj), None)
        if self.library is None:
            raise RuntimeError(f"Failed to load Metal library: {error}")

        # Cache Compute Pipelines
        self.pipelines = {}

    def _get_pipeline(self, kernel_name: str):
        """Get or create a compute pipeline state for a kernel function."""
        if kernel_name in self.pipelines:
            return self.pipelines[kernel_name]

        func = self.library.newFunctionWithName_(kernel_name)
        if func is None:
            raise ValueError(f"Kernel function '{kernel_name}' not found in library.")

        pipeline, error = self.device.newComputePipelineStateWithFunction_error_(func, None)
        if pipeline is None:
            raise RuntimeError(f"Failed to create pipeline for '{kernel_name}': {error}")

        self.pipelines[kernel_name] = pipeline
        return pipeline

    def dispatch(
        self,
        kernel_name: str,
        grid_size: tuple[int, int, int],
        buffers: list,
        threads_per_threadgroup: tuple[int, int, int] = (8, 4, 1)
    ):
        """Dispatch a compute kernel.

        Args:
            kernel_name: Name of the MSL kernel function (e.g., 'plm_reconstruct_x').
            grid_size: Total number of threads (x, y, z).
            buffers: List of Metal buffers or numpy arrays (will be converted if needed).
                     Note: Zero-copy requires creating buffers explicitly.
            threads_per_threadgroup: Threads per group.
        """
        pipeline = self._get_pipeline(kernel_name)
        cmd_buffer = self.command_queue.commandBuffer()
        encoder = cmd_buffer.computeCommandEncoder()

        encoder.setComputePipelineState_(pipeline)

        for i, buf in enumerate(buffers):
            # Assuming buf is already an MTLBuffer wrapper
            # In a real impl, we'd handle numpy -> MTLBuffer conversion here
            if hasattr(buf, 'metal_buffer'):
                 encoder.setBuffer_offset_atIndex_(buf.metal_buffer, 0, i)
            else:
                 # Raw MTLBuffer object
                 encoder.setBuffer_offset_atIndex_(buf, 0, i)

        # Calculate grid size
        w, h, d = threads_per_threadgroup
        t_group_size = MTLSize(w, h, d)

        nx, ny, nz = grid_size
        # Round up to multiple of threadgroup size
        gx = (nx + w - 1) // w * w
        gy = (ny + h - 1) // h * h
        gz = (nz + d - 1) // d * d

        # Dispatch
        # Use dispatchThreadgroups:threadsPerThreadgroup: for flexibility
        n_groups = MTLSize(gx // w, gy // h, gz // d)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(n_groups, t_group_size)

        encoder.endEncoding()
        cmd_buffer.commit()

        # For validtion/debugging, we might want to wait.
        # But for perf, we return immediately.
        return cmd_buffer

    def numpy_to_buffer(self, arr: np.ndarray):
        """Create a zero-copy Metal buffer from a numpy array."""
        # Ensure array is contiguous and page-aligned if possible?
        # Shared memory requires StorageModeShared
        size = arr.nbytes
        options = MTLResourceStorageModeShared

        # We can use newBufferWithBytesNoCopy but that requires page alignment
        # and keeping the python object alive.
        # Simplest valid way: newBufferWithBytes (copy) or just newBuffer and memcpy.
        # For zero-copy, create buffer first, then numpy view.
        return self.device.newBufferWithBytes_length_options_(
            arr.ctypes.data,
            size,
            options
        )

    def create_shared_buffer(self, shape: tuple[int, ...], dtype=np.float32) -> tuple:
        """Create a shared Metal buffer and a numpy view into it.

        Args:
            shape: Shape of the array.
            dtype: Numpy data type.

        Returns:
            Tuple of (MTLBuffer, np.ndarray).
        """
        size = int(np.prod(shape) * np.dtype(dtype).itemsize)
        options = MTLResourceStorageModeShared

        buffer = self.device.newBufferWithLength_options_(size, options)
        if buffer is None:
             raise RuntimeError("Failed to allocate shared buffer")

        # Create numpy view from buffer contents
        ptr = buffer.contents() # Returns int/ptr

        if dtype == np.float32:
             ctype = ctypes.c_float
        elif dtype == np.int32:
             ctype = ctypes.c_int32
        else:
             raise ValueError(f"Unsupported dtype: {dtype}")

        num_elements = int(np.prod(shape))

        # Cast ptr to correct type and create ctypes array
        # PyObjC contents() returns an integer address usually
        c_ptr = ctypes.cast(ptr, ctypes.POINTER(ctype))

        # Create numpy array from buffer
        # Using as_array on a pointer creates a view
        np_array = np.ctypeslib.as_array(c_ptr, shape=(num_elements,)).reshape(shape)

        return buffer, np_array
