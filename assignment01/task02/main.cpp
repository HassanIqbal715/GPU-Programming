#include<iostream>
#include<ctime>
#include<vector>
#include"file.h"
#include"utils.h"
using namespace std;

/*
* First argument provided is for the input file path.
* Second argument provided is for the output file path.
*/

int main(int argc, char* argv[]) {
    string inputPath = "./data/input.txt";
    string outputPath = "./data/output.txt";

    cout << "==================================\n";
    if (argc <= 1) {
        // No arguments
        cerr << "Warning: I/O not provided. Using the default paths\n";
        cout << "Input file path: ./data/input.txt\n";
        cout << "Output file path: ./data/output.txt\n";
    }
    else if (argc == 2) {
        // One argument provided. Assume it's the inputPath
        inputPath = argv[1];
        cout << "Input path set: " << inputPath << endl;
        cout << "Output file path: " << outputPath << endl;
    }
    else {
        // Two arguments provided. Assume it's first the inputPath and then the outputPath.
        inputPath = argv[1];
        outputPath = argv[2];
        cout << "Input path set: " << inputPath << endl;
        cout << "Output path set: " << outputPath << endl;
    }
    cout << "==================================\n";

    // Check the user's preference.
    int option = 0;
    cout << "1. Use the " << inputPath << " file as input.\n";
    cout << "2. Use a randomly generated array with size N.\n";
    cout << "Enter anything else to quit.\n";
    cout << "Choose: ";
    cin >> option;

    vector<int> inputDataArray1 = {};
    vector<int> inputDataArray2 = {};
    vector<int> sizeArray = {};
    int rows = 0, columns = 0;

    if (option == 1) {
        // Read and extract data from inputFile
        File* inputFile = new File(inputPath, FileMode::READ);

        vector<string> inputDataString = inputFile->readFile();

        sizeArray = extractInputData(inputDataString[0], 1, 2);
        inputDataArray1 = extractInputData(inputDataString[1], rows, columns);
        inputDataArray2 = extractInputData(inputDataString[2], rows, columns);

        rows = sizeArray[0];
        columns = sizeArray[1];

        // Immediately clear the read data because it is of no use anymore.
        delete inputFile;
    }
    else if (option == 2) {
        // Create random matrices
        srand(time(0));
        cout << "Enter number of rows: ";
        cin >> rows;
        cout << "Enter number of columns: ";
        cin >> columns;
        inputDataArray1 = randomizeArray(rows, columns);
        inputDataArray2 = randomizeArray(rows, columns);
    }
    else {
        cout << "Okay! Bye bye!\n";
        return 0;
    }

    File* outputFile = new File(outputPath, FileMode::WRITE);

    int** A = createMatrix(inputDataArray1, rows, columns);
    int** B = createMatrix(inputDataArray2, rows, columns);
    int** C = createMatrix(rows, columns);

    // Compute the sum and store it in an empty matrix C
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    // Prepare text for writing and write it to the outputFile.
    string buffer;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            buffer.append(to_string(C[i][j]));
            buffer.append(" ");
            cout << C[i][j] << " ";
        }
        outputFile->writeFile(buffer, (i == 0)); // (i == 0) to clear on 1st run
        cout << '\n';
        buffer.clear(); // empty buffer string/text
    }

    // Clear allocated memory
    for (int i = 0; i < rows; i++) {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete outputFile;
}