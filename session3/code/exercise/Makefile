STARPU_VERSION=1.4
CC = gcc
CFLAGS += -O3 -Wall -Wextra $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION))

all: multi_matrix

multi_matrix: multi_matrix.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

clean:
	rm -f multi_matrix
