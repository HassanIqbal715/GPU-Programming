#pragma once
#include "file.h"
#include "matrix.h"
#include <string>
#include <vector>
using namespace std;

void createMatrix(Matrix &mat, int rows, int columns);
double* extractInputData(string inputData, int rows, int columns);
void writeMatrix(Matrix &mat, string &outputPath);
void handleArguments(string& inputPath, string& outputPath, int argc, char* argv[]);
vector<Matrix> loadData(string& inputPath);