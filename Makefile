# Compiler and flags
NVCC = nvcc
INCLUDE_DIR = .\include
SRC_DIR_CPU = src\cpu
SRC_DIR_GPU = src\gpu
SRC_MAIN_CPU = $(SRC_DIR_CPU)\main.cpp
SRC_MAIN_GPU = $(SRC_DIR_GPU)\main.cu
OUT_FILE = main.exe
EXP_FILE = main.exp
LIB_FILE = main.lib

# Default target
all: cpu-run

# Build CPU version
cpu:
	$(NVCC) -I$(INCLUDE_DIR) $(SRC_MAIN_CPU) -o $(OUT_FILE)

# Build GPU version
gpu:
	$(NVCC) --ptxas-options --suppress-stack-size-warning -diag-suppress 20014 -diag-suppress 20011 -G -I$(INCLUDE_DIR) $(SRC_MAIN_GPU) -o $(OUT_FILE)

# Run CPU version
cpu-run: cpu
	.\$(OUT_FILE)

# Run GPU version
gpu-run: gpu
	.\$(OUT_FILE)

# Sanitize GPU version
gpu-sanitize: gpu
	compute-sanitizer --leak-check full --track-stream-ordered-races all .\$(OUT_FILE)

# Clean build artifacts
clean:
	del $(OUT_FILE) $(EXP_FILE) $(LIB_FILE)
