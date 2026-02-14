#include "file.h"
#include "utils.h"
#include "compute.h"
#include <iostream>
#include <chrono>
#include <vector>
using namespace std;

/*
 * First argument provided is for the input file path.
 * Second argument provided is for the output file path.
 */

int main(int argc, char *argv[]) {
    string inputPath = "../../data/input.txt";
    string outputPath = "../../data/output.txt";

    handleArguments(inputPath, outputPath, argc, argv);

    // Read and extract data from inputFile
    File *inputFile = new File(inputPath, FileMode::READ);

    vector<string> inputDataString = inputFile->readFile();

    if (inputDataString.size() == 0) {
        cerr << "Error! Empty input file provided!\n";
        return 1;
    }

    int *numberOfMatrices = extractInputData(inputDataString[0], 1, 1);
    int n = numberOfMatrices[0];

    vector<int *> inputDataMatrices;

    int *sizeArray;
    int resultRows = 0, resultColumns = 0, intermediateRows = 0;

    int *tempArray;

    for (int i = 1; i < (n * 2) + 1; i++) {    
        sizeArray = extractInputData(inputDataString[i++], 1, 2);
        // Reading the first matrix's size
        if (i - 1 == 1) {
            resultRows = sizeArray[0];
            intermediateRows = sizeArray[1];
        }
        // Reading the last matrix's size
        if (i == n * 2)
            resultColumns = sizeArray[1];

        tempArray = extractInputData(inputDataString[i], sizeArray[0], sizeArray[1]);
        inputDataMatrices.push_back(tempArray);
        delete[] sizeArray;
    }

    // Immediately clear the read data because it is of no use anymore.
    delete inputFile;

    File *outputFile = new File(outputPath, FileMode::WRITE);

    // Result matrix
    double *C = createMatrix(resultRows, resultColumns);

    auto startTimePoint = chrono::steady_clock::now();
    
    // Compute the sum and store it in result matrix
    cpuMul(C, inputDataMatrices[0], inputDataMatrices[1], resultRows, 
        intermediateRows, resultColumns);

    auto endTimePoint = chrono::steady_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(
        endTimePoint - startTimePoint
    );

    cout << "Execution Time: ";
    cout << static_cast<double>(duration.count()) / 1000000;
    cout << "seconds" << endl;

    // Write the result to the output file
    writeMatrix(C, n, resultRows, resultColumns, outputFile);
    cout << "Successfully written output to: " << outputPath << endl;

    // Clear allocated memory
    for (int i = 0; i < n; i++) {
        delete[] inputDataMatrices[i];
    }

    delete[] C;
    delete outputFile;
}