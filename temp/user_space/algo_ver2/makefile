CXX := g++
CXXFLAGS := -g -Wall -std=gnu++23 -mavx512f -O3 -march=native -ffast-math
LDFLAGS := -ltbb
SRC_DIR := /run/media/trung/hdddrive/CODE/CTDLGTVSOOP/src/AI-C++-NUMPY/algo_ver2
INCLUDE_DIR := $(SRC_DIR)/include/SIMD
BIN := test.run

SRCS := $(SRC_DIR)/test.cxx
all: $(BIN)

$(BIN): $(SRCS)
	$(CXX) $(CXXFLAGS) -v $(SRCS) -o $(BIN) $(LDFLAGS)

run: all
	./$(BIN)

clean:
	rm -f $(BIN)
