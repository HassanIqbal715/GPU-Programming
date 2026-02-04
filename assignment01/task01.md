# Environment Configuration
Since I don't own a NVIDIA GPU, I used google colab for this assignment. It is my GPU source as well as my compilation/running environment for all CUDA code.

## GPU Instance
16GB Tesla T4 GPU by Google Colab.

## NVIDIA CUDA
- Version V12.5.82.
- NVCC compiler that Colab provides with T4.
- Architecture sm_75 cuda. Compatible with NVIDIA RTX 20xx series and Tesla T4.

## Running Cuda Program
I have to follow numerous steps to use my personal IDE with google colab. Though this is temporary and just for this assignment. I hope I can come up with a better workflow/pipeline.

0. Prequesite/setup step is to integrate colab kernel and connect it in VSCode.
1. Sync local files with google drive using `rclone`. It's a single command of `rclone sync source destination`.
2. Mount google drive in colab `drive.mount()`. It asks for authentication every single time by the way :(.
3. Use bash commands to cd into the source files in a python notebook which is connected to colab.
4. Use bash for nvcc and compile the code. This command uses an extra `-arch=sm_75` flag due to incompatibility of T4 with the latest cuda supported architecture.
5. Use bash to run the code `./main.o`. Easy.

## Extra
I feel like I should mention this. GPU time (copying + computation) takes way more time than CPU computation. Copying is very slow when I tried to check time on only the computation. I wonder if we could keep the data on the GPU memory for longer periods until we don't need to process the data anymore.