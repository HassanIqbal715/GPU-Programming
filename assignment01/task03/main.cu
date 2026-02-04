#include "../include/file.h"
#include "../include/utils.h"
#include "compute.cuh"
#include <iostream>
#include <chrono>
#include <vector>

int main(int argc, char* argv[]) {
    string inputPath = "../data/input.txt";
    string outputPath = "../data/output.txt";


    cout << "==================================\n";
    if (argc <= 1) {
        // No arguments
        cerr << "Warning: I/O not provided. Using the default paths\n";
        cout << "Input file path: ../data/input.txt\n";
        cout << "Output file path: ../data/output.txt\n";
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

    cout << "Input file path: " << inputPath;
    cout << "Output file path: " << outputPath;

    // Read and extract data from inputFile
    File *inputFile = new File(inputPath, FileMode::READ);

    vector<string> inputDataString = inputFile->readFile();

    // Number of matrices
    int* numberOfMatrices = extractInputData(inputDataString[0], 1, 1);
    int n = numberOfMatrices[0];

    // Rows and Columns
    int* sizeArray = extractInputData(inputDataString[1], 1, 2);
    int rows = sizeArray[0];
    int columns = sizeArray[1];

    // Matrices data
    vector<int*> inputDataMatrices;
    int* tempArray;
    for (int i = 2; i < n + 2; i++) {
        tempArray = extractInputData(inputDataString[i], rows, columns);
        inputDataMatrices.push_back(tempArray);
    }

    // Immediately clear the read data because it is of no use anymore.
    delete inputFile;

    File *outputFile = new File(outputPath, FileMode::WRITE);

    // Empty matrix
    int *C = createMatrix(rows, columns);

    int *C_d;
    size_t total_size = rows * columns * sizeof(int);
    
    chrono::steady_clock::time_point startTimePoint = chrono::steady_clock::now();
    cudaMalloc(&C_d, total_size);
    cudaMemset(C_d, 0, total_size);

    // Compute the sum of all the matrices
    for (int i = 0; i < n; i++) {
        gpuSum(C_d, inputDataMatrices[i], rows, columns);
    }

    // Copy back the result
    cudaMemcpy(C, C_d, total_size, cudaMemcpyDeviceToHost);
    cudaFree(C_d);

    chrono::steady_clock::time_point endTimePoint = chrono::steady_clock::now();

    chrono::microseconds duration = chrono::duration_cast<chrono::microseconds>(endTimePoint - startTimePoint);

    cout << static_cast<double>(duration.count()) / 1000000;

    // Write the result to the output file
    writeMatrix(C, n, rows, columns, outputFile);
    cout << "Successfully written to: " << outputPath << endl;
    
    // Clear allocated memory
    for (int i = 0; i < n; i++) {
        delete[] inputDataMatrices[i];
    }

    delete[] C;
    delete outputFile;

    cudaDeviceSynchronize();
    return 0;
}