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

    cout << "========================\n";
    handleArguments(inputPath, outputPath, argc, argv);
    cout << "========================\n";

    // Read and extract data from inputFile
    vector<Matrix> matrices = loadData(inputPath);

    if (matrices.size() < 2) {
        cerr << "Error: File not compatible or empty\n";
        return 1;
    }

    // Result matrix
    Matrix result;
    createMatrix(result, matrices[0].rows, matrices[1].cols);

    auto startTimePoint = chrono::steady_clock::now();
    
    // Compute the sum and store it in result matrix
    try {
        cpuMul(result, matrices[0], matrices[1]);
    }
    catch(runtime_error err) {
        cerr << "Error: " << err.what() << "\n";
        return 1;
    }

    auto endTimePoint = chrono::steady_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(
        endTimePoint - startTimePoint
    );

    cout << "Execution Time: ";
    cout << static_cast<double>(duration.count()) / 1000000;
    cout << "seconds" << endl;

    // Write the result to the output file
    writeMatrix(result, outputPath);
    cout << "\nSuccessfully written output to: " << outputPath << endl;
}