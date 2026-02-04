__global__ void computeSum(int *result, int *A_d, int rows, int columns);
void gpuSum(int *&result_d, int *A, int rows, int columns);