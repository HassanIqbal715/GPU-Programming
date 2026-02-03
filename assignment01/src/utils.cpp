#include "../include/utils.h"
#include <iostream>
#include <string>

// Create an empty matrix
int *allocateMatrix(int rows, int columns) {
    int *matrix = new int [rows * columns];
    return matrix;
}

// Create matrix initialized to 0
int *createMatrix(int rows, int columns) {
    int *matrix = allocateMatrix(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i * rows + j] = 0;
        }
    }

    return matrix;
}

// Create a matrix and fill it with the data provided
int *createMatrix(vector<int> inputData, int rows, int columns) {
    int *matrix = allocateMatrix(rows, columns);

    int index = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i * rows + j] = inputData[index++];
        }
    }

    return matrix;
}

// Convert a inputData provided in a string to integer data
// Returns array of data
int* extractInputData(string inputData, int rows, int columns) {
    int* array = new int[rows * columns];
    int index = 0;
    int temp = 0;

    for (unsigned long i = 0; i < inputData.length() + 1; i++) {
        if (inputData[i] == ' ' || inputData[i] == '\0' ||
            inputData[i] == '\n' || inputData[i] == '\r') {
            array[index++] = temp;

            temp = 0;
            continue;
        }
        temp *= 10;
        temp += (static_cast<int>(inputData[i]) - '0');
    }

    return array;
}

// Write matrix to a file
void writeMatrix(int *&matrix, int rows, int columns, File *&targetFile) {    
    // Prepare text for writing and write it to the outputFile.
    string buffer;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            buffer.append(to_string(matrix[i * rows + j]));
            buffer.append(" ");
        }
        targetFile->writeFile(buffer, (i == 0)); // (i == 0) to clear on 1st run
        buffer.clear(); // empty buffer string/text
    }
}