#include "utils.h"
#include <iostream>
#include <string>

// Create matrix initialized to 0
void createMatrix(Matrix &mat, int rows, int columns) {
    mat.free();
    mat.data = new double [rows * columns];
    mat.rows = rows;
    mat.cols = columns;

    for (int i = 0; i < rows * columns; i++) {
        mat.data[i] = 0.0;
    }
}

// Convert a inputData provided in a string to integer data
// Returns array of data
double* extractInputData(string inputData, int rows, int columns) {
    double* array = new double[rows * columns];
    int index = 0;
    int temp = 0;

    for (long unsigned int i = 0; i < inputData.length() + 1; i++) {
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
void writeMatrix(Matrix &mat, string &outputPath) {
    // Prepare text for writing and write it to the outputFile.
    string buffer;
    File *outputFile = new File(outputPath, FileMode::WRITE);

    outputFile->writeFile(to_string(mat.rows) + " " + to_string(mat.cols));
    for (int i = 0; i < mat.rows * mat.cols; i++) {
        buffer.append(to_string(mat.data[i]));
        buffer.append(" ");
    }

    outputFile->writeFile(buffer, false);
    buffer.clear(); // empty buffer string/text

    delete outputFile;
}

void handleArguments(string& inputPath, string& outputPath, int argc, char* argv[]) {
    if (argc <= 1) {
        // No arguments
        cout << "Warning: I/O files not provided. Using the default paths\n\n";
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
}

vector<Matrix> loadData(string& inputPath) {
    File *inputFile = new File(inputPath, FileMode::READ);

    vector<string> inputDataString = inputFile->readFile();

    if (inputDataString.size() == 0) {
        cerr << "Error! Empty input file provided!\n";
        return {};
    }

    vector<Matrix> inputDataMatrices;

    double *sizeArray;
    double *tempArray;
    
    // extracting data for matrix 1
    sizeArray = extractInputData(inputDataString[0], 1, 2);
    tempArray = extractInputData(inputDataString[1], (int) sizeArray[0], (int) sizeArray[1]);
    Matrix mat1;
    mat1.rows = (int) sizeArray[0];
    mat1.cols = (int) sizeArray[1];
    mat1.data = tempArray;
    inputDataMatrices.push_back(mat1);
    
    delete[] sizeArray;

    // extracting data for matrix 2
    sizeArray = extractInputData(inputDataString[2], 1, 2);
    tempArray = extractInputData(inputDataString[3], sizeArray[0], sizeArray[1]);
    Matrix mat2;
    mat2.rows = (int) sizeArray[0];
    mat2.cols = (int) sizeArray[1];
    mat2.data = tempArray;
    inputDataMatrices.push_back(mat2);
    
    delete[] sizeArray;

    // Immediately clear the read data because it is of no use anymore.
    delete inputFile;
    
    return inputDataMatrices;
}
