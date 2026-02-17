#include "compute.h"
#include <iostream>

void cpuMul(Matrix &result, Matrix &A, Matrix &B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("matrix size mismatch");
    }

    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            for (int k = 0; k < A.cols; k++) {
                result.data[i * B.cols + j] += (A.data[i * A.cols + k] * 
                    B.data[k * B.cols + j]);
            }
        }
    }
}