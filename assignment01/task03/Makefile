# Makefile
NVCC = nvcc
TARGET = cuda_app
SRC = program.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)