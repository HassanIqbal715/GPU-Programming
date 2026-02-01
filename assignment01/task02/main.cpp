#include<iostream>
#include<ctime>
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
        File* inputFile = new File(inputPath, FileMode::READ);

        string* inputDataString = inputFile->readFile();

        n = static_cast<int>(inputDataString[0][0]) - '0';

        inputDataArray1 = extractInputData(inputDataString[1], n*n);
        inputDataArray2 = extractInputData(inputDataString[2], n*n);

        delete[] inputDataString;
        delete inputFile;
    }
    else if (option == 2) {
        srand(time(0));
        cout << "Enter size N: ";
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
    delete outputFile;
    return 0;
}