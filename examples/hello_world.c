#include <stdio.h>
#include <unistd.h>
#include <starpu.h>

void hello_cpu(void *buffers[], void *cl_arg)
{
    (void)buffers; (void)cl_arg;
    printf("Hello world from worker %d\n", starpu_worker_get_id());
    sleep(1); /* keep the worker busy so that the tasks run in parallel */
}

struct starpu_codelet hello_cl =
{
    .cpu_funcs = { hello_cpu },
    .nbuffers = 0,
    .name = "hello",
};

int main(void)
{
    int ret = starpu_init(NULL);
    if (ret == -ENODEV) return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    starpu_topology_print(stdout);
    for (int i = 0; i < 4; i++) {
        ret = starpu_task_insert(&hello_cl, 0);
        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
    }
    starpu_task_wait_for_all();

    starpu_shutdown();
    return 0;
}
