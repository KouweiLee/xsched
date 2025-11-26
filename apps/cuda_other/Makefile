# Makefile for CUDA stream concurrency check

NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_70 -std=c++11

TARGET = check_stream_concurrency
SOURCE = check_stream_concurrency.cu

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

help:
	@echo "Usage:"
	@echo "  make          - Build the program"
	@echo "  make run      - Build and run the program"
	@echo "  make clean    - Remove built files"
	@echo ""
	@echo "Program usage:"
	@echo "  ./$(TARGET) [num_streams] [num_iterations]"
	@echo "    num_streams: Number of streams to test (default: 4)"
	@echo "    num_iterations: Number of iterations (default: 100)"

