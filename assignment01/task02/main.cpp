#include<iostream>
using namespace std;

int main() {
    string input = "3 123456789 123456789";
    string output = "";

    int A[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int B[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    int C[3][3];

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }

    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cout << C[i][j] << " ";
        }
        cout << '\n';
    }

    return 0;
}