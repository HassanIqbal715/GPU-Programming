#include <iostream>
using namespace std;

int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);

    cudaDeviceProp devProp;
    for (int i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
        cout << "Name: " << devProp.name << "\n";
        cout << "Max Threads Per Block: " << devProp.maxThreadsPerBlock << "\n";
        cout << "Number of SMs: " << devProp.multiProcessorCount << "\n";
        cout << "Blocks per SM: " << devProp.maxBlocksPerMultiProcessor << "\n";
        cout << "Threads per SM: " << devProp.maxThreadsPerMultiProcessor << "\n";
        cout << "Warps per SM: " << devProp.maxThreadsPerMultiProcessor / devProp.warpSize << "\n";
        cout << "Registers per SM: " << devProp.regsPerMultiprocessor << "\n";
        cout << "Shared Memory per SM: " << devProp.sharedMemPerMultiprocessor << "\n";
        cout << "Shared Memory per block: " << devProp.sharedMemPerBlock << "\n";
        cout << "Clock rate: " << (double) devProp.clockRate / 1000000.0 << "GHz\n";
        cout << "Registers per block: " << devProp.regsPerBlock << "\n";
        cout << "Warp size: " << devProp.warpSize << "\n";
        cout << "Max thread dim(x, y, z): " << "(" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << ")\n"; 
        cout << "Max grid size(x, y, z): " << "(" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", " << devProp.maxGridSize << ")\n";
    }
}