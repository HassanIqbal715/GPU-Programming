#include "matrix.h"
#include <chrono>
using namespace std;

__global__ void gpuMulKernel(double* result, double* A, double* B, 
                       int rows, int cols, int intermediate);

void gpuMul(Matrix &result, Matrix &A, Matrix &B, 
    chrono::microseconds &duration);