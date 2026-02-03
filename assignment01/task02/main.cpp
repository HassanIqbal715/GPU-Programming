#include "../include/file.h"
#include "../include/utils.h"
#include "compute.h"
#include <ctime>
#include <iostream>
#include <vector>
using namespace std;

/*
 * First argument provided is for the input file path.
 * Second argument provided is for the output file path.
 */

int main(int argc, char *argv[]) {
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

    vector<int *> inputDataMatrices;
    int *sizeArray;
    int n = 0;
    int rows = 0, columns = 0;

    // Read and extract data from inputFile
    File *inputFile = new File(inputPath, FileMode::READ);

    vector<string> inputDataString = inputFile->readFile();

    int *numberOfMatrices = extractInputData(inputDataString[0], 1, 1);
    n = numberOfMatrices[0];
    sizeArray = extractInputData(inputDataString[1], 1, 2);

    rows = sizeArray[0];
    columns = sizeArray[1];

    int *tempArray;
    for (int i = 2; i < n + 2; i++) {
        tempArray = extractInputData(inputDataString[i], rows, columns);
        inputDataMatrices.push_back(tempArray);
    }

    // Immediately clear the read data because it is of no use anymore.
    delete inputFile;

    File *outputFile = new File(outputPath, FileMode::WRITE);

    int *C = createMatrix(rows, columns);

    // Compute the sum and store it in an empty matrix C
    for (int i = 0; i < n; i++) {
        cpuSum(C, inputDataMatrices[i], rows, columns);
    }

    // Write the result to the output file
    writeMatrix(C, rows, columns, outputFile);
    cout << "Successfully written output to: " << outputPath << endl;

    // Clear allocated memory
    for (int i = 0; i < n; i++) {
        delete[] inputDataMatrices[i];
    }

    delete[] C;
    delete outputFile;
}