#include "compute.h"

void cpuSum(int *&result, int *A, int rows, int columns) {
    int index = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            index = i * rows + j;
            result[index] += (A[index]);
        }
    }
}