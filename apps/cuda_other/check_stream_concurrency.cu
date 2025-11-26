#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Simple kernel for testing
__global__ void busy_kernel(float *data, int n, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        // Do some computation
        for (int i = 0; i < iterations; i++) {
            val = val * 1.0001f + 0.0001f;
        }
        data[idx] = val;
    }
}

void check_cuda_error(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

void print_gpu_info(int device)
{
    cudaDeviceProp prop;
    check_cuda_error(cudaGetDeviceProperties(&prop, device), "Get device properties");
    
    printf("GPU %d: %s\n", device, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels: %s\n", 
           prop.concurrentKernels ? "Yes" : "No");
    printf("  Max Threads Per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("\n");
}

bool test_concurrent_streams(int num_streams, int num_iterations)
{
    const int N = 1024 * 1024;  // 1M elements
    const size_t size = N * sizeof(float);
    
    // Allocate host memory
    float *h_data = (float*)malloc(size * num_streams);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return false;
    }
    
    // Initialize host data
    for (int i = 0; i < N * num_streams; i++) {
        h_data[i] = 1.0f;
    }
    
    // Allocate device memory for each stream
    float **d_data = (float**)malloc(num_streams * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(num_streams * sizeof(cudaStream_t));
    
    // Create streams and allocate device memory
    for (int i = 0; i < num_streams; i++) {
        check_cuda_error(cudaStreamCreate(&streams[i]), "Create stream");
        check_cuda_error(cudaMalloc(&d_data[i], size), "Allocate device memory");
    }
    
    // Copy data to device (async)
    for (int i = 0; i < num_streams; i++) {
        check_cuda_error(
            cudaMemcpyAsync(d_data[i], &h_data[i * N], size, 
                           cudaMemcpyHostToDevice, streams[i]),
            "Copy to device"
        );
    }
    
    // Launch kernels concurrently on all streams
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    int kernel_iterations = 100;
    
    clock_t start = clock();
    
    for (int iter = 0; iter < num_iterations; iter++) {
        // Launch kernels on all streams
        for (int i = 0; i < num_streams; i++) {
            busy_kernel<<<gridSize, blockSize, 0, streams[i]>>>(
                d_data[i], N, kernel_iterations
            );
        }
    }
    
    // Synchronize all streams
    for (int i = 0; i < num_streams; i++) {
        check_cuda_error(cudaStreamSynchronize(streams[i]), "Synchronize stream");
    }
    
    clock_t end = clock();
    double elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Check for kernel launch errors
    for (int i = 0; i < num_streams; i++) {
        check_cuda_error(cudaGetLastError(), "Kernel launch");
    }
    
    // Copy results back
    for (int i = 0; i < num_streams; i++) {
        check_cuda_error(
            cudaMemcpyAsync(&h_data[i * N], d_data[i], size,
                           cudaMemcpyDeviceToHost, streams[i]),
            "Copy from device"
        );
    }
    
    // Final synchronization
    for (int i = 0; i < num_streams; i++) {
        check_cuda_error(cudaStreamSynchronize(streams[i]), "Final sync");
    }
    
    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
    }
    free(d_data);
    free(streams);
    free(h_data);
    
    printf("  Concurrent execution test:\n");
    printf("    Streams: %d\n", num_streams);
    printf("    Iterations: %d\n", num_iterations);
    printf("    Elapsed time: %.3f seconds\n", elapsed);
    printf("    Throughput: %.2f iterations/second\n", num_iterations / elapsed);
    
    return true;
}

int main(int argc, char **argv)
{
    int num_streams = 4;
    int num_iterations = 100;
    
    if (argc > 1) {
        num_streams = atoi(argv[1]);
    }
    if (argc > 2) {
        num_iterations = atoi(argv[2]);
    }
    
    printf("======================================================================\n");
    printf("GPU Stream Concurrency Check\n");
    printf("======================================================================\n");
    printf("\n");
    
    int device_count;
    check_cuda_error(cudaGetDeviceCount(&device_count), "Get device count");
    
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable devices found\n");
        return 1;
    }
    
    for (int device = 0; device < device_count; device++) {
        check_cuda_error(cudaSetDevice(device), "Set device");
        print_gpu_info(device);
        
        printf("Testing concurrent execution...\n");
        if (test_concurrent_streams(num_streams, num_iterations)) {
            printf("  ✓ Test completed successfully\n");
        } else {
            printf("  ✗ Test failed\n");
        }
        printf("\n");
        
        if (device < device_count - 1) {
            printf("----------------------------------------------------------------------\n");
        }
    }
    
    printf("Summary:\n");
    printf("  - GPUs with Compute Capability >= 2.0 support concurrent kernels\n");
    printf("  - This is called Hyper-Q technology (introduced in Fermi)\n");
    printf("  - Multiple CUDA streams can execute kernels simultaneously\n");
    printf("  - XSched can schedule multiple XQueues concurrently\n");
    printf("\n");
    
    return 0;
}

