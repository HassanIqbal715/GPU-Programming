#pragma once
#include<fstream>
#include<string>
using namespace std;

class File {
private:
    string path;
    fstream file;

public:
    File(string path);
    void createFile();
    bool openFile();
    void writeFile(string line);
    string* readFile();
    ~File();
};