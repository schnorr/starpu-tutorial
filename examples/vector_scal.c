#include <stdio.h>
#include <starpu.h>

#define NX 1024

void scal_cpu(void *buffers[], void *cl_arg)
{
    float factor;
    starpu_codelet_unpack_args(cl_arg, &factor);

    unsigned n = STARPU_VECTOR_GET_NX(buffers[0]);
    float *v = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    for (unsigned i = 0; i < n; i++)
        v[i] *= factor;
}

struct starpu_codelet scal_cl =
{
    .cpu_funcs = { scal_cpu },
    .nbuffers = 1,
    .modes = { STARPU_RW },
    .name = "scal",
};

int main(void)
{
    float vector[NX];
    for (unsigned i = 0; i < NX; i++)
        vector[i] = 1.0f;

    int ret = starpu_init(NULL);
    if (ret == -ENODEV) return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    starpu_data_handle_t vector_h;
    starpu_vector_data_register(&vector_h, STARPU_MAIN_RAM,
                                (uintptr_t)vector, NX, sizeof(vector[0]));

    /* Both tasks access vector_h in RW mode: StarPU infers that the
       second one must wait for the first (read-after-write). */
    float two = 2.0f, three = 3.0f;
    ret = starpu_task_insert(&scal_cl, STARPU_RW, vector_h,
                             STARPU_VALUE, &two, sizeof(two), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
    ret = starpu_task_insert(&scal_cl, STARPU_RW, vector_h,
                             STARPU_VALUE, &three, sizeof(three), 0);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");

    /* Unregistering waits for the tasks and brings the data back to RAM. */
    starpu_data_unregister(vector_h);
    starpu_shutdown();

    printf("vector[0] = %.1f (expected 6.0)\n", vector[0]);
    return vector[0] == 6.0f ? 0 : 1;
}
