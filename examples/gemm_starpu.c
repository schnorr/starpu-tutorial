/* Tiled matrix multiplication C = C + A * B with StarPU.
   Matrices are N x N, column-major, partitioned into NB x NB blocks.
   Each block product is a task with a CPU and (optionally) a CUDA
   implementation; the runtime decides where each task runs. */
#include <stdio.h>
#include <stdlib.h>
#include <starpu.h>
#ifdef STARPU_USE_CUDA
#include <cublas_v2.h>
#include <starpu_cublas_v2.h>
#endif

void gemm_cpu(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    const float *A = (float *)STARPU_MATRIX_GET_PTR(buffers[0]);
    const float *B = (float *)STARPU_MATRIX_GET_PTR(buffers[1]);
    float *C       = (float *)STARPU_MATRIX_GET_PTR(buffers[2]);
    unsigned m = STARPU_MATRIX_GET_NX(buffers[2]);  /* rows of C    */
    unsigned n = STARPU_MATRIX_GET_NY(buffers[2]);  /* columns of C */
    unsigned k = STARPU_MATRIX_GET_NY(buffers[0]);  /* columns of A */
    unsigned lda = STARPU_MATRIX_GET_LD(buffers[0]);
    unsigned ldb = STARPU_MATRIX_GET_LD(buffers[1]);
    unsigned ldc = STARPU_MATRIX_GET_LD(buffers[2]);

    for (unsigned j = 0; j < n; j++)
        for (unsigned l = 0; l < k; l++) {
            float b = B[l + j * ldb];
            for (unsigned i = 0; i < m; i++)
                C[i + j * ldc] += A[i + l * lda] * b;
        }
}

#ifdef STARPU_USE_CUDA
void gemm_cuda(void *buffers[], void *cl_arg)
{
    (void)cl_arg;
    const float one = 1.0f;
    cublasStatus_t st = cublasSgemm(
        starpu_cublas_get_local_handle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        STARPU_MATRIX_GET_NX(buffers[2]),
        STARPU_MATRIX_GET_NY(buffers[2]),
        STARPU_MATRIX_GET_NY(buffers[0]),
        &one,
        (float *)STARPU_MATRIX_GET_PTR(buffers[0]),
        STARPU_MATRIX_GET_LD(buffers[0]),
        (float *)STARPU_MATRIX_GET_PTR(buffers[1]),
        STARPU_MATRIX_GET_LD(buffers[1]),
        &one,
        (float *)STARPU_MATRIX_GET_PTR(buffers[2]),
        STARPU_MATRIX_GET_LD(buffers[2]));
    if (st != CUBLAS_STATUS_SUCCESS)
        STARPU_CUBLAS_REPORT_ERROR(st);
}
#endif

/* History-based performance model: StarPU records the duration of each
   implementation per block size and uses it in model-based schedulers. */
static struct starpu_perfmodel gemm_model =
{
    .type = STARPU_HISTORY_BASED,
    .symbol = "gemm",
};

static struct starpu_codelet gemm_cl =
{
    .cpu_funcs = { gemm_cpu },
#ifdef STARPU_USE_CUDA
    .cuda_funcs = { gemm_cuda },
    .cuda_flags = { STARPU_CUDA_ASYNC },
#endif
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_RW },
    .model = &gemm_model,
    .name = "gemm",
};

int main(int argc, char *argv[])
{
    unsigned N  = argc > 1 ? atoi(argv[1]) : 2048;
    unsigned NB = argc > 2 ? atoi(argv[2]) : 8;
    if (N % NB) { fprintf(stderr, "N must be a multiple of NB\n"); return 1; }

    int ret = starpu_init(NULL);
    if (ret == -ENODEV) return 77;
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
#ifdef STARPU_USE_CUDA
    starpu_cublas_init();
#endif

    /* starpu_malloc returns pinned memory when GPUs are present,
       which makes CPU<->GPU transfers asynchronous and faster. */
    float *A, *B, *C;
    size_t size = (size_t)N * N * sizeof(float);
    starpu_malloc((void **)&A, size);
    starpu_malloc((void **)&B, size);
    starpu_malloc((void **)&C, size);
    for (size_t x = 0; x < (size_t)N * N; x++) {
        A[x] = 1.0f; B[x] = 2.0f; C[x] = 0.0f;
    }

    /* Register each matrix once, then partition it into NB x NB blocks. */
    starpu_data_handle_t A_h, B_h, C_h;
    starpu_matrix_data_register(&A_h, STARPU_MAIN_RAM, (uintptr_t)A, N, N, N, sizeof(float));
    starpu_matrix_data_register(&B_h, STARPU_MAIN_RAM, (uintptr_t)B, N, N, N, sizeof(float));
    starpu_matrix_data_register(&C_h, STARPU_MAIN_RAM, (uintptr_t)C, N, N, N, sizeof(float));

    struct starpu_data_filter rows =
    {
        .filter_func = starpu_matrix_filter_block,
        .nchildren = NB,
    };
    struct starpu_data_filter cols =
    {
        .filter_func = starpu_matrix_filter_vertical_block,
        .nchildren = NB,
    };
    starpu_data_map_filters(A_h, 2, &rows, &cols);
    starpu_data_map_filters(B_h, 2, &rows, &cols);
    starpu_data_map_filters(C_h, 2, &rows, &cols);

    double start = starpu_timing_now();
    for (unsigned i = 0; i < NB; i++)
        for (unsigned j = 0; j < NB; j++)
            for (unsigned k = 0; k < NB; k++) {
                ret = starpu_task_insert(&gemm_cl,
                    STARPU_R,  starpu_data_get_sub_data(A_h, 2, i, k),
                    STARPU_R,  starpu_data_get_sub_data(B_h, 2, k, j),
                    STARPU_RW, starpu_data_get_sub_data(C_h, 2, i, j),
                    0);
                STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
            }
    starpu_task_wait_for_all();
    double end = starpu_timing_now();

    /* Gather the blocks back into the original matrices in main memory. */
    starpu_data_unpartition(A_h, STARPU_MAIN_RAM);
    starpu_data_unpartition(B_h, STARPU_MAIN_RAM);
    starpu_data_unpartition(C_h, STARPU_MAIN_RAM);
    starpu_data_unregister(A_h);
    starpu_data_unregister(B_h);
    starpu_data_unregister(C_h);

    double secs = (end - start) / 1e6;
    printf("N=%u NB=%u time=%.3f s GFlop/s=%.2f C[0]=%.1f (expected %.1f)\n",
           N, NB, secs, 2.0 * N * N * N / secs / 1e9, C[0], 2.0f * N);
    int ok = C[0] == 2.0f * N && C[(size_t)N * N - 1] == 2.0f * N;

    starpu_free_noflag(A, size);
    starpu_free_noflag(B, size);
    starpu_free_noflag(C, size);
#ifdef STARPU_USE_CUDA
    starpu_cublas_shutdown();
#endif
    starpu_shutdown();
    return ok ? 0 : 1;
}
