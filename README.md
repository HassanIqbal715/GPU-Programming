# GPU-Programming
A repository to store assignments and practices for the GPU Programming course (CS327) being offered at my university.

## Build instructions
`assignment02` and onwards contain a CMakeList.txt file that can be used to compile easily.

A possible compilation method is:
1. Create a build folder inside the respective task like `assignment02/task03/build`.
2. Run the cmake command with the `taskxx/build/` path as the build folder argument
   and `taskxx/` path as the source argument.
3. cd into `build/` and run using `./bin/main.x` while being in the build folder.
   Where x is some file type depending on your OS. It works for linux
   (Probably works for windows too, Idk).
