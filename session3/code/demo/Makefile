# Instructor-only demo, run on the GPU machine (see handouts/session3.org).
# Adjust CUDA_PKG below to match the CUDA toolkit version installed on that
# machine's pkg-config search path (e.g. cuda-11.0, cudart-11.0, cublas-11.0,
# or the unversioned cudart/cublas names on newer toolkits).
STARPU_VERSION=1.4
CUDA_PKG=cuda cudart cublas

CC = gcc
CFLAGS += -O3 -Wall -Wextra $(shell pkg-config --cflags starpu-$(STARPU_VERSION) $(CUDA_PKG))
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION) $(CUDA_PKG))

all: soma_matrix_cuda

soma_matrix_cuda: soma_matrix_cuda.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

clean:
	rm -f soma_matrix_cuda
