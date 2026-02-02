#include"utils.h"
#include<string>
#include<iostream>

// Create an empty matrix
int** allocateMatrix(int rows, int columns) {
    int** matrix = new int*[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new int[columns];
    }

    return matrix;
}

// Create matrix initialized to 0
int** createMatrix(int rows, int columns) {
    int** matrix = allocateMatrix(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = 0;
        }
    }

    return matrix;
}

// Create a matrix and fill it with the data provided
int** createMatrix(vector<int> inputData, int rows, int columns) {
    int** matrix = allocateMatrix(rows, columns);

    int index = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i][j] = inputData[index++];
        }
    }

    return matrix;
}

// Convert a inputData provided in a string to integer data
// Returns array of data
vector<int> extractInputData(string inputData, int rows, int columns) {
    vector<int> array;
    int temp = 0;

    for (unsigned long i = 0; i < inputData.length() + 1; i++) {
        if (inputData[i] == ' ' || inputData[i] == '\0' || inputData[i] == '\n' || inputData[i] == '\r') {
            array.push_back(temp);

            temp = 0;
            continue;
        }
        temp *= 10;
        temp += (static_cast<int>(inputData[i]) - '0');
    }

    return array;
}

// Create an array with random values
vector<int> randomizeArray(int rows, int columns) {
    vector<int> array;

    // fill array with numbers between 1 and 100 inclusive
    for (int i = 0; i < rows*columns; i++) {
        array.push_back(rand() % 100 + 1);
    }

    return array;
}