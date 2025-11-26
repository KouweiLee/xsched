# GPU Stream Concurrency Check

This directory contains tools to check if your GPU supports concurrent execution of multiple CUDA streams.

## What is Stream Concurrency?

Modern GPUs (Compute Capability >= 2.0) support **Hyper-Q** technology, which allows:
- Multiple CUDA kernels to execute simultaneously on the same GPU
- Different streams to run kernels concurrently
- Better GPU utilization through parallel execution

## Tools

### 1. Python Version (Recommended)

**File**: `check_stream_concurrency.py`

**Requirements**:
```bash
pip install torch
# Optional: pip install nvidia-ml-py
```

**Usage**:
```bash
python3 check_stream_concurrency.py
```

**Features**:
- Checks GPU compute capability
- Verifies Hyper-Q support
- Tests actual concurrent execution with multiple streams
- Works with PyTorch

### 2. CUDA C++ Version

**File**: `check_stream_concurrency.cu`

**Requirements**:
- CUDA toolkit
- nvcc compiler

**Build and Run**:
```bash
make
./check_stream_concurrency [num_streams] [num_iterations]
```

**Example**:
```bash
make
./check_stream_concurrency 4 100
```

**Features**:
- Direct CUDA API access
- More detailed GPU information
- Lower-level concurrency testing

## Understanding the Results

### Compute Capability

- **>= 2.0 (Fermi+)**: Supports concurrent kernels via Hyper-Q
- **< 2.0 (Pre-Fermi)**: Limited or no concurrent kernel support

### Concurrent Kernels Property

The `concurrentKernels` property in CUDA device properties indicates:
- **true**: GPU can execute multiple kernels simultaneously
- **false**: Kernels are serialized (one at a time)

### Why This Matters for XSched

- XSched creates an XQueue for each CUDA stream
- If GPU supports concurrent execution, multiple XQueues can run simultaneously
- This enables better GPU utilization and scheduling flexibility
- YOLO/PyTorch creating many streams is beneficial when GPU supports concurrency

## Example Output

```
======================================================================
GPU Stream Concurrency Check
======================================================================

GPU 0: NVIDIA GeForce RTX 3090
  Compute Capability: 8.6
  Concurrent Kernels: Yes
  ✓ This GPU supports concurrent execution of multiple streams

Testing concurrent execution on GPU 0...
  ✓ Concurrent execution test PASSED
    Completed 150 iterations in 1.00 seconds
    Throughput: 150.00 iterations/second
```

## Notes

- All modern GPUs (since Fermi, 2010) support concurrent kernels
- The number of concurrent kernels depends on GPU architecture
- XSched benefits from concurrent execution support
- Multiple streams from YOLO/PyTorch can run concurrently if GPU supports it

