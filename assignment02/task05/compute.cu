#include "compute.h"
#include <chrono>
#include <iostream>

#define tW 24

__global__ void gpuMulKernel(double* result, double* A, double* B, 
                       int rows, int cols, int intermediate) {
    int tY = threadIdx.y, tX = threadIdx.x;
    int col = blockDim.x * blockIdx.x + tX;
    int row = blockDim.y * blockIdx.y + tY;

    if (col < cols && row < rows) {
        result[row * cols + col] = 0;
    }

    __shared__ double Mds[tW][tW];
    __shared__ double Nds[tW][tW];

    int phases = ceil((double) intermediate / tW);
    double partialSum = 0;

    for (int ph = 0; ph < phases; ph++) {
        if (row < rows && (ph * tW + tX) < intermediate) {
            Mds[tY][tX] = A[row * intermediate + ph * tW + tX];
        }
        else {
            Mds[tY][tX] = 0.0;
        }

        if ((ph * tW + tY) < intermediate && col < cols) {
            Nds[tY][tX] = B[ph * tW * cols + tY * cols + col];
        }
        else {
            Nds[tY][tX] = 0.0;
        }

        __syncthreads();

        if (col < cols && row < rows) {
            for (int i = 0; i < tW; i++) {
                partialSum += Mds[tY][i] * Nds[i][tX];
            }
        }
        __syncthreads();
    }

    if (row < rows && col < cols)
        result [row * cols + col] += partialSum;
}

void gpuMul(Matrix &result, Matrix &A, Matrix &B, 
        chrono::microseconds &duration) {
    if (A.cols != B.rows) {
        throw runtime_error("matrix size mismatch");
    }

    // Allocate device memory
    int sizeA = sizeof(double) * A.rows * A.cols;
    int sizeB = sizeof(double) * B.rows * B.cols;
    int sizeResult = sizeof(double) * result.rows * result.cols;

    double* A_d;
    double* B_d;
    double* result_d;
    cudaMalloc((void **) &A_d, sizeA);
    cudaMalloc((void **) &B_d, sizeB);
    cudaMalloc((void **) &result_d, sizeResult);

    /* One of the optimal threads per block for A2000 GPU = 128 threads.
     * it assigns 1536/128 = 12 blocks per SM.
     *
     * A2000 GPU has limits 16 blocks, 1536 threads and 48 warps per SM.
     * 65,336 registers/1536 threads = 42.67 registers per thread in SM. 
     * 
     * Which is why the current configuration can achieve 100% occupancy if the
     * number of registers allocated is below or equal to 42 registers per 
     * thread.
     * 
     * For tiling. Shared memory of an SM in A2000 is 100KB. Shared memory per
     * block is 48KB. We can stay within those bounds easily. To maintain
     * high occupancy (100% duh) along with a good shared memory usage, 
     * 16 - 24 tileWidth seems to work best
     */

    dim3 blockDim(24, 24, 1);
    dim3 gridDim(
        ceil((double) result.cols / blockDim.x), 
        ceil((double) result.rows / blockDim.y),
        1
    );

    auto startTimePoint = chrono::steady_clock::now();
    
    // Copy data to the device
    cudaMemcpy((void*) A_d, (void*) A.data, sizeA, 
        cudaMemcpyHostToDevice
    );
    cudaMemcpy((void*) B_d, (void*) B.data, sizeB, 
        cudaMemcpyHostToDevice
    );

    // Kernel Run. Compute.
    gpuMulKernel<<<gridDim, blockDim>>>(result_d, A_d, B_d, 
        result.rows, result.cols, A.cols);
    
    cudaDeviceSynchronize();

    // Copy result to the host
    cudaMemcpy((void*) result.data, (void*) result_d, sizeResult, 
        cudaMemcpyDeviceToHost);
 
    auto endTimePoint = chrono::steady_clock::now();
    
    duration = chrono::duration_cast<chrono::microseconds>(
            endTimePoint - startTimePoint
    );

    // Free the memory
    cudaFree((void*) result_d);
    cudaFree((void*) A_d);
    cudaFree((void*) B_d);
}