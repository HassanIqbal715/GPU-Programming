#include"file.h"
#include<iostream>
#include<filesystem>

File::File(string path) {
    this->path = path;
    this->openMode = FileMode::READ_WRITE;
}

File::File(string path, FileMode openMode) {
    this->path = path;
    this->openMode = openMode;
}

void File::createFile() {
    char buffer[500];
    int bufferIndex = 0;
    int latestSlashIndex = -1;

    // finding and putting directory and file names in the buffer
    for (unsigned long i = 0; i < path.length(); i++) {
        if (path[i] == '/' || path[i] == '\\')
            latestSlashIndex = bufferIndex;

        buffer[bufferIndex] = path[i];
        bufferIndex++;
    }

    // Create directories if a "slash" was found
    if (latestSlashIndex != -1) {
        buffer[latestSlashIndex] = '\0';
        filesystem::create_directories(buffer);
    }

    filesystem::path filePath = path;

    // Create the file
    ofstream f(filePath);
    if (f.is_open()) f.close();
}

// Opens file for the stored open mode. Returns true on open and false on error
bool File::openFile() {
    if (file.is_open()) {
        return true;
    }

    std::filesystem::path filePath = path;

    if (!std::filesystem::exists(filePath))
        createFile();

    if (openMode == FileMode::READ_WRITE)
        file.open(filePath, ios::in | ios::out);
    else if (openMode == FileMode::READ)
        file.open(filePath, ios::in);
    else if (openMode == FileMode::WRITE)
        file.open(filePath, ios::out);
    else 
        file.open(filePath, ios::in | ios::out);

    return file.is_open();
}

// Write the line provided. Reset the writing pointer to the beginning if pointerReset is true
void File::writeFile(string line, bool pointerReset = false) {
    if (!openFile() || openMode == FileMode::READ) return;

    file.clear(); // reset the flags
    if (pointerReset) {
        file.seekg(0, ios_base::beg); // start writing from the 1st character
    }
    
    file << line << endl;
}

// Read the file line by line. Return a string pointer with all the data.
string* File::readFile() {
    if (!openFile() || openMode == FileMode::WRITE) return nullptr;

    file.seekp(0, ios_base::beg);

    string line;
    string* data = new string[500];
    int index = 0;
    while(getline(file, line)) {
        data[index++] = line;
    }

    data[index] = '\0';
    return data;
}

File::~File() {
    if (file.is_open())
        file.close();
}