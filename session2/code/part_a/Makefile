# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2022-2025  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

# Trimmed for Session 2, Part A: only the three programs used in this
# session are built here (upstream also builds a "variables" example, not
# needed for this session — see the full starpu-docker/tutorial/C/Makefile
# for that). The CUDA/OpenCL implementations are still built when the
# installed StarPU has that support enabled: vector_scal_task_insert.c
# conditionally references them via #ifdef STARPU_USE_CUDA/STARPU_USE_OPENCL
# (a property of the StarPU install, not of this Makefile), so the object/
# library must be available whenever the headers enable it, even though
# this session's exercises are run CPU-only.
PROGS=vector_scal0 vector_scal_task_insert vector_scal_task_insert_filter

include ./starpu.mk

vector_scal_task_insert: vector_scal_task_insert.o vector_scal_cpu.o
vector_scal_task_insert_filter: vector_scal_task_insert_filter.o vector_scal_cpu.o

ifeq ($(USE_CUDA),1)
vector_scal_task_insert: vector_scal_cuda.o
vector_scal_task_insert_filter: vector_scal_cuda.o
vector_scal_task_insert: LDLIBS+=-L$(CUDA_PATH)/lib64 -lcudart -lstdc++
vector_scal_task_insert_filter: LDLIBS+=-L$(CUDA_PATH)/lib64 -lcudart -lstdc++
endif

ifeq ($(USE_OPENCL),1)
vector_scal_task_insert: vector_scal_opencl.o
vector_scal_task_insert_filter: vector_scal_opencl.o
vector_scal_task_insert: LDLIBS+=-lOpenCL
vector_scal_task_insert_filter: LDLIBS+=-lOpenCL
endif
