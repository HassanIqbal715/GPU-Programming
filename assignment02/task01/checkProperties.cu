#include <iostream>
using namespace std;

// I got this function from a stackoverflow post.
// Had no idea that all of this is required to find the cores per SM.
int getSPcores(cudaDeviceProp devProp) {  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
      else printf("Unknown device type\n");
      break;
     case 9: // Hopper
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     case 10: // Blackwell
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     case 12: // Blackwell
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);

    cudaDeviceProp devProp;
    for (int i = 0; i < devCount; i++) {
        cudaGetDeviceProperties(&devProp, i);
        cout << "Name: " << devProp.name << "\n";
        cout << "\nBlock info:\n";
        cout << "Max Threads Per Block: " << devProp.maxThreadsPerBlock << "\n";
        cout << "Shared Memory per block: " << (double) devProp.sharedMemPerBlock / 1024.0 << "KB\n";
        cout << "Registers per block: " << devProp.regsPerBlock << "\n";
        cout << "Max thread dim(x, y, z): " << "(" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << ")\n"; 
        cout << "Max grid size(x, y, z): " << "(" << devProp.maxGridSize[0] << ", " << devProp.maxGridSize[1] << ", " << devProp.maxGridSize[2] << ")\n";
        cout << "\nMultiprocessors (SM) info:\n";
        cout << "Number of SMs: " << devProp.multiProcessorCount << "\n";
        cout << "Blocks per SM: " << devProp.maxBlocksPerMultiProcessor << "\n";
        cout << "Threads per SM: " << devProp.maxThreadsPerMultiProcessor << "\n";
        cout << "Warps per SM: " << devProp.maxThreadsPerMultiProcessor / devProp.warpSize << "\n";
        cout << "Registers per SM: " << devProp.regsPerMultiprocessor << "\n";
        cout << "Shared Memory per SM: " << (double) devProp.sharedMemPerMultiprocessor / 1024.0 << "KB\n";
        cout << "\nMisc:\n";
        cout << "Clock rate: " << (double) devProp.clockRate / 1000000.0 << "GHz\n";
        cout << "Warp size: " << devProp.warpSize << "\n";
        cout << "Memory Bandwidth: " << ((double) devProp.memoryClockRate / 1000000) * (devProp.memoryBusWidth / 8) * 2 << "GB/s\n";
        cout << "Peak performance: " << getSPcores(devProp) * ((double) devProp.clockRate / 1000000) * 2 << "GFLOPS\n";
    }
}