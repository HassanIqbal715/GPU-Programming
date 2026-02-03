#pragma once
#include "file.h"
#include <string>
#include <vector>
using namespace std;

int *allocateMatrix(int rows, int columns);
int *createMatrix(int rows, int columns);
int *createMatrix(vector<int> inputData, int rows, int columns);
int* extractInputData(string inputData, int rows, int columns);
vector<int> randomizeArray(int rows, int columns);
void writeMatrix(int *&matrix, int rows, int columns, File *&targetFile);