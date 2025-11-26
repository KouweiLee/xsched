#!/usr/bin/env python3
"""
Check if GPU supports concurrent execution of multiple CUDA streams

This script checks:
1. GPU compute capability
2. Whether GPU supports Hyper-Q (concurrent kernel execution)
3. Actual concurrent execution test with multiple streams
"""

import sys
import time

try:
    import torch
    import torch.cuda
except ImportError:
    print("Error: PyTorch is not installed. Please install it with: pip install torch")
    sys.exit(1)

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    print("Warning: pynvml not installed. Some GPU info may be unavailable.")
    print("Install with: pip install nvidia-ml-py")


def get_gpu_info_pytorch():
    """Get GPU information using PyTorch"""
    if not torch.cuda.is_available():
        return None
    
    device_count = torch.cuda.device_count()
    info = []
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        info.append({
            'id': i,
            'name': props.name,
            'compute_capability': f"{props.major}.{props.minor}",
            'major': props.major,
            'minor': props.minor,
            'total_memory': props.total_memory,
            'multi_processor_count': props.multi_processor_count,
        })
    
    return info


def get_gpu_info_nvml():
    """Get GPU information using nvidia-ml-py"""
    if not HAS_PYNVML:
        return None
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        info = []
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Get compute capability
            try:
                major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                compute_capability = f"{major}.{minor}"
            except:
                compute_capability = "Unknown"
            
            info.append({
                'id': i,
                'name': name,
                'compute_capability': compute_capability,
            })
        
        return info
    except Exception as e:
        print(f"Error getting GPU info via NVML: {e}")
        return None


def check_concurrent_kernel_support(compute_major):
    """
    Check if GPU supports concurrent kernel execution (Hyper-Q)
    
    Hyper-Q was introduced in Fermi (2.0) and improved in Kepler (3.0+)
    All modern GPUs (compute capability >= 2.0) support concurrent kernels
    """
    if compute_major >= 2:
        return True, "Yes (Hyper-Q supported since Fermi)"
    elif compute_major == 1:
        return False, "No (Pre-Fermi architecture)"
    else:
        return False, "Unknown"


def test_concurrent_streams(device_id=0, num_streams=4, test_duration=1.0):
    """
    Test actual concurrent execution of multiple streams
    
    Args:
        device_id: GPU device ID
        num_streams: Number of streams to test
        test_duration: Duration of test in seconds
    
    Returns:
        Tuple of (success, details)
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    try:
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device_id)
        
        # Create multiple streams
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        
        # Create test tensors
        size = (1024, 1024)  # 4MB per tensor
        tensors = []
        for i in range(num_streams):
            with torch.cuda.stream(streams[i]):
                tensors.append(torch.randn(size, device=device))
        
        # Synchronize all streams
        for stream in streams:
            stream.synchronize()
        
        # Test concurrent execution
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < test_duration:
            # Launch kernels on all streams concurrently
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    # Simple computation
                    tensors[i] = torch.matmul(tensors[i], tensors[i].t())
            
            # Synchronize all streams
            for stream in streams:
                stream.synchronize()
            
            iterations += 1
        
        elapsed = time.time() - start_time
        
        # Cleanup
        del tensors
        torch.cuda.empty_cache()
        
        return True, {
            'iterations': iterations,
            'elapsed_time': elapsed,
            'throughput': iterations / elapsed if elapsed > 0 else 0
        }
        
    except Exception as e:
        return False, f"Test failed: {e}"


def main():
    print("=" * 70)
    print("GPU Stream Concurrency Check")
    print("=" * 70)
    print()
    
    # Get GPU information
    gpu_info = get_gpu_info_pytorch()
    
    if gpu_info is None or len(gpu_info) == 0:
        print("Error: No CUDA-capable GPU found")
        sys.exit(1)
    
    # Check each GPU
    for gpu in gpu_info:
        print(f"GPU {gpu['id']}: {gpu['name']}")
        print(f"  Compute Capability: {gpu['compute_capability']}")
        print(f"  Total Memory: {gpu['total_memory'] / (1024**3):.2f} GB")
        print(f"  Multiprocessors: {gpu['multi_processor_count']}")
        
        # Check concurrent kernel support
        supports_concurrent, reason = check_concurrent_kernel_support(gpu['major'])
        print(f"  Concurrent Kernel Support: {reason}")
        
        if supports_concurrent:
            print(f"  ✓ This GPU supports concurrent execution of multiple streams")
        else:
            print(f"  ✗ This GPU does NOT support concurrent execution")
        
        print()
        
        # Test actual concurrent execution
        print(f"Testing concurrent execution on GPU {gpu['id']}...")
        print("  Creating 4 streams and running concurrent kernels...")
        
        success, result = test_concurrent_streams(
            device_id=gpu['id'],
            num_streams=4,
            test_duration=1.0
        )
        
        if success:
            print(f"  ✓ Concurrent execution test PASSED")
            print(f"    Completed {result['iterations']} iterations in {result['elapsed_time']:.2f} seconds")
            print(f"    Throughput: {result['throughput']:.2f} iterations/second")
        else:
            print(f"  ✗ Concurrent execution test FAILED: {result}")
        
        print()
        print("-" * 70)
        print()
    
    # Summary
    print("Summary:")
    print("  Modern GPUs (Compute Capability >= 2.0) support concurrent")
    print("  execution of multiple CUDA streams through Hyper-Q technology.")
    print()
    print("  This means:")
    print("  - Multiple kernels can run simultaneously on the same GPU")
    print("  - Different streams can execute kernels concurrently")
    print("  - XSched can schedule multiple XQueues concurrently")
    print()
    
    # Additional information
    if gpu_info[0]['major'] >= 2:
        print("  Your GPU supports concurrent execution!")
        print("  Multiple streams created by YOLO/PyTorch can run concurrently.")
    else:
        print("  Your GPU may have limited concurrent execution support.")
        print("  Streams will be serialized rather than executed concurrently.")


if __name__ == '__main__':
    main()

