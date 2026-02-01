#pragma once
#include<string>
using namespace std;

int** createMatrix(int size);
int** createMatrix(int* inputData, int size);
int* extractInputData(string inputData, int size);