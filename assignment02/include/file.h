#pragma once
#include <fstream>
#include <string>
#include <vector>
using namespace std;

enum FileMode {READ, WRITE, READ_WRITE};

class File {
private:
    string path;
    fstream file;
    FileMode openMode;

public:
    File(string path);
    File(string path, FileMode openMode);
    void createFile();
    bool openFile();
    void writeFile(string line, bool pointerReset = false);
    vector<string> readFile();
    ~File();
};