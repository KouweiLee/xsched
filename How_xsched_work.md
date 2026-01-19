# How xsched Works: Interception and Scheduling Mechanism

This document explains the technical implementation of `xsched` and how it intercepts CUDA calls to achieve transparent scheduling and preemption without modifying application source code.

## 1. Core Interception: `LD_PRELOAD`

The foundation of `xsched` is the Linux `LD_PRELOAD` environment variable.

### How it works:
When a program starts, the **Dynamic Linker** (e.g., `ld-linux.so`) resolves external symbols (like `cuLaunchKernel`) by searching through linked libraries. By setting `LD_PRELOAD=/path/to/libxsched_cuda_shim.so`, we tell the linker to load our shim library **before** any other libraries, including the official NVIDIA driver (`libcuda.so`).

### Symbol Redirection:
- **Normal Flow**: Application -> `libcuda.so` (Official)
- **xsched Flow**: Application -> `libxsched_cuda_shim.so` (xsched) -> `libcuda.so` (Official)

If `xsched` exports a symbol with the exact same name and signature as a CUDA driver API, the linker will bind the application's call to `xsched`'s implementation instead of the original one.

## 2. Implementation in xsched

### Interception Layer (`intercept.cpp`)
In `platforms/cuda/shim/src/intercept.cpp`, `xsched` uses macros to define functions that match the CUDA Driver API. For example:

```cpp
DEFINE_EXPORT_C_REDIRECT_CALL(XLaunchKernel, CUresult, cuLaunchKernel, ...);
```

This expands to a standard C function named `cuLaunchKernel`. When the application calls `cuLaunchKernel(...)`, it is actually executing code inside `xsched`.

### Scheduling Logic (`XLaunchKernel`)
Once intercepted, the call enters `xsched`'s internal logic (the `X...` functions):
1. **Task Wrapping**: The kernel parameters (grid, block, params, etc.) are wrapped into a command object (e.g., `CudaCommand`).
2. **Queueing**: This object is submitted to a **Virtual Queue** (XQueue) managed by `xsched`.
3. **Scheduling**: `xsched`'s scheduler decides when to actually run this task based on priority, resource availability, and preemption policies.

### Calling the Real Driver (Avoiding Recursion)
To actually run the task on the GPU, `xsched` must eventually call the real NVIDIA driver. To avoid calling itself (which would cause infinite recursion), it uses `dlsym` with the `RTLD_NEXT` flag or by explicitly opening the system `libcuda.so`.

This logic is typically handled in `driver.h` and `symbol.h` via `GetSymbol`:

```cpp
// Pseudocode for getting the real address
void* real_func = dlsym(RTLD_NEXT, "cuLaunchKernel");
```

## 3. Transparency and Benefits

- **Zero Code Modification**: Applications run unmodified. Just prefix the execution command with `LD_PRELOAD`.
- **Granular Control**: By sitting between the application and the driver, `xsched` can pause, resume, or prioritize GPU tasks at the driver API level.
- **PTDS Support**: `xsched` handles both standard and Per-Thread Default Stream (PTDS) versions of APIs (e.g., `cuLaunchKernel_ptsz`) to ensure coverage across different compilation modes.

---
*Created on 2026-01-18 for xsched internal documentation.*
