#pragma once
#include "file.h"
#include <string>
#include <vector>
using namespace std;

int *createMatrix(int rows, int columns);
int* extractInputData(string inputData, int rows, int columns);
void writeMatrix(int *&matrix, int n, int rows, int columns, File *&targetFile);