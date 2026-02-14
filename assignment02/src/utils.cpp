#include "utils.h"
#include <iostream>
#include <string>

// Create matrix initialized to 0
double *createMatrix(int rows, int columns) {
    double *matrix = new double [rows * columns];

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i * columns + j] = 0;
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
void writeMatrix(double *&matrix, int n, int rows, int columns, File *&targetFile) {
    // Prepare text for writing and write it to the outputFile.
    string buffer;

    targetFile->writeFile(to_string(n));
    targetFile->writeFile(to_string(rows) + " " + to_string(columns));
    for (int i = 0; i < rows * columns; i++) {
        buffer.append(to_string(matrix[i]));
        buffer.append(" ");
    }
    targetFile->writeFile(buffer, false);
    buffer.clear(); // empty buffer string/text
}

void handleArguments(string& inputPath, string& outputPath, int argc, char* argv[]) {
    cout << "==================================\n";
    if (argc <= 1) {
        // No arguments
        cerr << "Warning: I/O files not provided. Using the default paths\n";
        cout << "Input file path: " << inputPath << endl;
        cout << "Output file path: " << outputPath << endl;
    } 
    else if (argc == 2) {
        // One argument provided. Assume it's the inputPath
        inputPath = argv[1];
        cout << "Input path set: " << inputPath << endl;
        cout << "Output file path: " << outputPath << endl;
    } 
    else {
        // Two arguments provided. Assume it's first the inputPath and then the
        // outputPath.
        inputPath = argv[1];
        outputPath = argv[2];
        cout << "Input path set: " << inputPath << endl;
        cout << "Output path set: " << outputPath << endl;
    }
    cout << "==================================\n";
}