STARPU_VERSION=1.4
CC = gcc
CFLAGS += -O3 -Wall -Wextra $(shell pkg-config --cflags starpu-$(STARPU_VERSION))
LDLIBS += $(shell pkg-config --libs starpu-$(STARPU_VERSION))

all: multi_matrix_two_impl

multi_matrix_two_impl: multi_matrix_two_impl.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

clean:
	rm -f multi_matrix_two_impl *.o
	rm -f paje* prof_* *.parquet *.feather *.csv.gz
	rm -f dag.dot dag.csv.gz atree.dot atree.csv entities.csv types.csv
	rm -f data.rec tasks.rec sched_tasks.rec comms.rec trace.rec
	rm -f activity.data distrib.data trace.html starpu.log
	rm -f starvz_plot.pdf
