#include "compute.h"

void cpuMul(double *&result, int *A, int*B, int rows, int middle, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            for (int k = 0; k < middle; k++) {
                result[i * columns + j] += (A[i * middle + k] * B[k * columns + j]);
            }
        }
    }
}