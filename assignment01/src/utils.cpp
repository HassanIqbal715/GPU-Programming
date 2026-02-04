#include "../include/utils.h"
#include <iostream>
#include <string>

// Create matrix initialized to 0
int *createMatrix(int rows, int columns) {
    int *matrix = new int [rows * columns];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i * rows + j] = 0;
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

    for (unsigned int i = 0; i < inputData.length() + 1; i++) {
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
void writeMatrix(int *&matrix, int n, int rows, int columns, File *&targetFile) {
    // Prepare text for writing and write it to the outputFile.
    string buffer;

    targetFile->writeFile(to_string(n));
    targetFile->writeFile(to_string(rows) + " " + to_string(columns));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            buffer.append(to_string(matrix[i * rows + j]));
            buffer.append(" ");
        }
        targetFile->writeFile(buffer, false); // (i == 0) to clear on 1st run
        buffer.clear(); // empty buffer string/text
    }
}