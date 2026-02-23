#include "compute.h"
#include <iostream>

void cpuMul(Matrix &result, Matrix &A, Matrix &B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("matrix size mismatch");
    }

    for (int i = 0; i < A.rows; i++) {
        for (int k = 0; k < A.cols; k++) {
            double temp = A.data[i * A.cols + k]; 
            for (int j = 0; j < B.cols; j++) {
                result.data[i * B.cols + j] += temp * B.data[k * B.cols + j];
            }
        }
    }
}