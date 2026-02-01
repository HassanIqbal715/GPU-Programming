#include<iostream>
#include<ctime>
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

    int* inputDataArray1 = NULL;
    int* inputDataArray2 = NULL;
    int n = 0;

    if (option == 1) {
        // Read and extract data from inputFile
        File* inputFile = new File(inputPath, FileMode::READ);

        string* inputDataString = inputFile->readFile();

        // Index(0,0) has the size n for the matrices
        n = static_cast<int>(inputDataString[0][0]) - '0';

        inputDataArray1 = extractInputData(inputDataString[1], n*n);
        inputDataArray2 = extractInputData(inputDataString[2], n*n);

        // Immediately clear the read data because it is of no use anymore.
        delete[] inputDataString;
        delete inputFile;
    }
    else if (option == 2) {
        // Create random matrices
        srand(time(0));
        cout << "Enter size n: ";
        cin >> n;
        inputDataArray1 = randomizeArray(n*n);
        inputDataArray2 = randomizeArray(n*n);
        cout << "Created\n";
    }
    else {
        cout << "Okay! Bye bye!\n";
        return 0;
    }

    File* outputFile = new File(outputPath, FileMode::WRITE);

    int** A = createMatrix(inputDataArray1, n);
    int** B = createMatrix(inputDataArray2, n);
    int** C = createMatrix(n);

    // Compute the sum and store it in an empty matrix C
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    // Prepare text for writing and write it to the outputFile.
    string buffer;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buffer.append(to_string(C[i][j]));
            buffer.append(" ");
            cout << C[i][j] << " ";
        }
        outputFile->writeFile(buffer, (i == 0)); // (i == 0) to clear on 1st run
        cout << '\n';
        buffer.clear(); // empty buffer string/text
    }

    // Clear allocated memory
    for (int i = 0; i < n; i++) {
        delete[] A[i];
        delete[] B[i];
        delete[] C[i];
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] inputDataArray1;
    delete[] inputDataArray2;
    delete outputFile;
}