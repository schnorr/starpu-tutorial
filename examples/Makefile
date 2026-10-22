# Build the chapter examples.  Override STARPU to match your installation,
# e.g. make STARPU=starpu-1.3
STARPU ?= starpu-1.4
CFLAGS += -O2 -Wall $(shell pkg-config --cflags $(STARPU))
LDLIBS += $(shell pkg-config --libs $(STARPU))
# Link cuBLAS when StarPU was built with CUDA (detected from starpu_config.h,
# or forced with make CUDA=1)
STARPU_INCLUDE := $(patsubst -I%,%,$(shell pkg-config --cflags-only-I $(STARPU)))
CUDA ?= $(shell grep -qs 'define STARPU_USE_CUDA 1' \
          $(addsuffix /starpu_config.h,$(STARPU_INCLUDE)) && echo 1)
ifeq ($(CUDA),1)
LDLIBS += -lcublas -lcudart
endif

PROGS = hello_world vector_scal gemm_seq gemm_starpu
# exercise skeleton of the hands-on session, when present
PROGS += $(basename $(wildcard gemm_starpu_todo.c))

all: $(PROGS)

gemm_seq: CFLAGS = -O2 -Wall
gemm_seq: LDLIBS =

clean:
	rm -f $(PROGS)
