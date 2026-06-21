# Makefile for the PS9 self-contained C11 real-time core.
# Builds src/ -> bin/wfs_rt. Links ONLY -lm (no BLAS/LAPACK/FFTW/GSL).
# AVX2/FMA kernels are #ifdef __AVX2__ guarded inside the sources and are
# enabled by -march=native on AVX2 hosts; the code degrades to portable
# scalar C elsewhere. OpenMP is optional (#ifdef _OPENMP).
#
# Targets:
#   make        / make all   -> build bin/wfs_rt
#   make test                -> build and run the built-in C self-test
#   make clean               -> remove objects and binaries
#   make debug               -> build with -O0 -g -fsanitize=address,undefined

CC      ?= gcc
STD      = -std=c11
WARN     = -Wall -Wextra -Wshadow -Wpointer-arith -Wcast-qual
OPT      = -O3 -march=native -funroll-loops
OPENMP   = -fopenmp
CFLAGS  ?= $(STD) $(WARN) $(OPT) $(OPENMP)
LDLIBS   = -lm

SRC_DIR  = src
BIN_DIR  = bin

# All translation units that have a .c file (headers excluded).
SRCS = \
	$(SRC_DIR)/bmp.c \
	$(SRC_DIR)/matio.c \
	$(SRC_DIR)/linalg.c \
	$(SRC_DIR)/centroid.c \
	$(SRC_DIR)/slopes.c \
	$(SRC_DIR)/reconstruct.c \
	$(SRC_DIR)/dmcmd.c \
	$(SRC_DIR)/pipeline.c \
	$(SRC_DIR)/main.c

OBJS = $(SRCS:.c=.o)

TARGET = $(BIN_DIR)/wfs_rt

.PHONY: all clean test debug

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDLIBS)

# Generic compile rule. aoconfig.h is header-only; every TU may include any header.
$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Built-in self-test: runs wfs_rt --selftest (AOMX + BMP roundtrips, GEMV sanity).
test: $(TARGET)
	$(TARGET) --selftest

debug: CFLAGS = $(STD) $(WARN) -O0 -g -fsanitize=address,undefined $(OPENMP)
debug: clean all

clean:
	rm -f $(OBJS) $(TARGET)
	rm -rf $(BIN_DIR)
