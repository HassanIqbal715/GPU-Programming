#include "matrix.h"

Matrix::Matrix() {
    rows = 0;
    cols = 0;
    data = nullptr;
}

Matrix::Matrix(const Matrix& other) {
    rows = other.rows;
    cols = other.cols;

    if (other.data != nullptr) {
        data = new double[rows * cols];
        for (int i = 0; i < rows * cols; ++i) 
            data[i] = other.data[i];
    }
    else {
        data = nullptr;
    }
}

Matrix::~Matrix() {
    if (data)
        free();
}

Matrix &Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        free();
        rows = other.rows;
        cols = other.cols;
        if (other.data) {
            data = new double[rows * cols];
            for (int i = 0; i < rows * cols; ++i) 
                data[i] = other.data[i];
        } 
        else {
            data = nullptr;
        }
    }
    return *this;
}

void Matrix::free() {
    delete[] data;
    data = nullptr;
}