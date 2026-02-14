#pragma once
#include "file.h"
#include <string>
#include <vector>
using namespace std;

double *createMatrix(int rows, int columns);
int* extractInputData(string inputData, int rows, int columns);
void writeMatrix(double *&matrix, int n, int rows, int columns, File *&targetFile);
void handleArguments(string& inputPath, string& outputPath, int argc, char* argv[]);