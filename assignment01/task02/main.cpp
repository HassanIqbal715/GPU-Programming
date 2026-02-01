#include<iostream>
#include"file.h"
#include"utils.h"
using namespace std;

int main(int argc, char* argv[]) {

    string inputPath = "./data/input.txt";
    string outputPath = "./data/output.txt";

    cout << "==================================\n";
    if (argc <= 1) {
        cerr << "Warning: I/O not provided. Using the default paths\n";
        cout << "Input file path: ./data/input.txt\n";
        cout << "Output file path: ./data/output.txt\n";
    }
    else if (argc == 2) {
        inputPath = argv[1];
        cout << "Input path set: " << inputPath << endl;
        cout << "Output file path: " << outputPath << endl;
    }
    else {
        inputPath = argv[1];
        outputPath = argv[2];
        cout << "Input path set: " << inputPath << endl;
        cout << "Output path set: " << outputPath << endl;
    }
    cout << "==================================\n";

    File* inputFile = new File(inputPath, FileMode::READ);
    File* outputFile = new File(outputPath, FileMode::WRITE);

    string* inputDataString = inputFile->readFile();

    int n = static_cast<int>(inputDataString[0][0]) - '0';

    int* inputDataArray1 = extractInputData(inputDataString[1], n*n);
    int* inputDataArray2 = extractInputData(inputDataString[2], n*n);

    int** A = createMatrix(inputDataArray1, n);
    int** B = createMatrix(inputDataArray2, n);
    int** C = createMatrix(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    string buffer;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            buffer.append(to_string(C[i][j]));
            buffer.append(" ");
            cout << C[i][j] << " ";
        }
        outputFile->writeFile(buffer, (i == 0));
        cout << '\n';
        buffer.clear();
    }

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
    delete[] inputDataString;
    delete outputFile;
    delete inputFile;
    return 0;
}