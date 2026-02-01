#include"utils.h"
#include<string>
#include<iostream>

int** createMatrix(int size) {
    int** matrix = new int*[size];
    for (int i = 0; i < size; i++) {
        matrix[i] = new int[size];
    }

    int index = 0;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = 0;
        }
    }

    return matrix;
}

int** createMatrix(int* inputData, int size) {
    int** matrix = new int*[size];
    for (int i = 0; i < size; i++) {
        matrix[i] = new int[size];
    }

    int index = 0;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = inputData[index++];
        }
    }

    return matrix;
}

int* extractInputData(string inputData, int size) {
    int* array = new int[size];
    int temp = 0;
    int index = 0;

    for (int i = 0; i < inputData.length() + 1; i++) {
        if (inputData[i] == ' ' || inputData[i] == '\0' || inputData[i] == '\n') {
            array[index++] = temp;

            temp = 0;
            continue;
        }
        temp *= 10;
        temp += (static_cast<int>(inputData[i]) - '0');
    }

    return array;
}