#include"file.h"
#include<filesystem>

File::File(string path) {
    this->path = path;
}

void File::createFile() {
    char buffer[500];
    int bufferIndex = 0;
    int latestSlashIndex = -1;

    // finding and putting directory and file names in the buffer
    for (int i = 0; i < path.length(); i++) {
        if (path[i] == '/' || path[i] == '\\')
            latestSlashIndex = bufferIndex;

        buffer[bufferIndex] = path[i];
        bufferIndex++;
    }

    if (latestSlashIndex != -1) {
        buffer[latestSlashIndex] = '\0';
        filesystem::create_directories(buffer);
    }

    filesystem::path filePath = path;

    ofstream f(filePath);
    if (f.is_open()) f.close();
}