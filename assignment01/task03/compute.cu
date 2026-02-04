#include "compute.cuh"
#include <vector>

__global__ void computeSum(int *result, int *A_d, int rows, int columns) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (column < columns && row < rows) {
        int index = row * columns + column;
        result[index] += A_d[index];
    }
}

void gpuSum(int *&result_d, int *A_h, int rows, int columns) {
    int *A_d;
    size_t size = rows * columns * sizeof(int);
    
    cudaMalloc(&A_d, size);
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);

    int threadsX = 16;
    int threadsY = 16;
    dim3 blockSize(threadsX, threadsY);
    dim3 gridSize(ceil((float)columns / threadsX), ceil((float)rows / threadsY)); 

    computeSum<<<gridSize, blockSize>>>(result_d, A_d, rows, columns);
    
    cudaDeviceSynchronize();
    
    cudaFree(A_d);
}