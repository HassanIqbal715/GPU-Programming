#pragma once

struct Matrix {
    int rows;
    int cols;
    double *data;

    Matrix();
    Matrix(const Matrix& other);
    ~Matrix();
    Matrix& operator=(const Matrix& other);
    void free();
};
