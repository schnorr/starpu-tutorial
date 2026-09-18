# Instructor-only demo, run on the GPU machine (see ../../handout.org).
# The tutorial Docker/Podman image installs CUDA/cuBLAS via apt (Ubuntu's
# nvidia-cuda-toolkit), which does NOT ship pkg-config .pc files for
# cuda/cudart/cublas — pkg-config would just fail and silently produce no
# flags at all (not even for starpu-1.4, since a single failing package
# in the same pkg-config call blanks out the whole result). Headers land
# in /usr/include and libraries in /usr/lib/x86_64-linux-gnu, both already
# on gcc's default search path, so no extra -I/-L is needed — just link
# the libraries directly. If you're building this on a different (non
# apt/Ubuntu) CUDA install where cuda.pc/cudart.pc/cublas.pc do exist,
# switch back to pkg-config for those.
STARPU_VERSION=1.4

CC = gcc
CFLAGS += -O3 -Wall -Wextra $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION)) -lcudart -lcublas

all: soma_matrix_cuda

soma_matrix_cuda: soma_matrix_cuda.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

clean:
	rm -f soma_matrix_cuda
