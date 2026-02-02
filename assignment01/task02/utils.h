#pragma once
#include<string>
#include<vector>
using namespace std;

int** allocateMatrix(int rows, int columns);
int** createMatrix(int rows, int columns);
int** createMatrix(vector<int> inputData, int rows, int columns);
vector<int> extractInputData(string inputData, int rows, int columns);
vector<int> randomizeArray(int rows, int columns);