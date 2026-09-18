/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2024  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>

extern struct starpu_opencl_program programs;

void vector_scal_opencl(void *buffers[], void *_args)
{
	/* length of the vector */
	unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
	/* OpenCL copy of the vector pointer */
	cl_mem val = (cl_mem) STARPU_VECTOR_GET_DEV_HANDLE(buffers[0]);

	/* extract the value arguments */
	float factor;
	starpu_codelet_unpack_args(_args, &factor);

	/* Load the OpenCL kernel */
	int id, devid, err;
	cl_kernel kernel;
	cl_command_queue queue;

	id = starpu_worker_get_id();
	devid = starpu_worker_get_devid(id);

	err = starpu_opencl_load_kernel(&kernel, &queue, &programs,
					"vector_mult_opencl", devid);   /* Name of the codelet defined above */
	if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);

	err = clSetKernelArg(kernel, 0, sizeof(val), &val);
	err |= clSetKernelArg(kernel, 1, sizeof(n), &n);
	err |= clSetKernelArg(kernel, 2, sizeof(factor), &factor);
	if (err) STARPU_OPENCL_REPORT_ERROR(err);

	{
		size_t local=64;
		size_t global=n;
		err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
		if (err != CL_SUCCESS) STARPU_OPENCL_REPORT_ERROR(err);
	}

	clFinish(queue);

	starpu_opencl_release_kernel(kernel);
}
