#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <starpu.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Matrix order and block width are now runtime arguments (argv[1] and
 * argv[2]) instead of compile-time #defines, so the demo can be scaled
 * up on the fly without editing the source and rebuilding. The number
 * of blocks (and hence StarPU tasks) is (matrix_width/block_width)^2 —
 * scale block_width up together with matrix_width, or a bigger matrix
 * turns into millions of tiny tasks instead of a bigger, more visible
 * workload (each task's own work is block_width^2, so shrinking it
 * while growing the matrix makes things slower, not more parallel).
 * The defaults (10, 2) match the original, intentionally tiny demo. */
#define DEFAULT_MATRIX_WIDTH 10
#define DEFAULT_BLOCK_WIDTH 2

#define CHECK_CUBLAS(x) if(x!=CUBLAS_STATUS_SUCCESS){printf("Cublass error: %d\n", x);};
cublasHandle_t cublas_mainhandle;

/* Set once in main() before any task is submitted; codelets read it. */
static int g_block_width;

void func_gpu(void *buffers[], void *args)
{
    float alpha_beta = 1;
    float *A = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    float *B = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
    float *C = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);

    cublasSetStream(cublas_mainhandle, starpu_cuda_get_local_stream());
    CHECK_CUBLAS(cublasSgeam(cublas_mainhandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                g_block_width, g_block_width,
                &alpha_beta,
                A,
                g_block_width,
                &alpha_beta,
                B,
                g_block_width,
                C,
                g_block_width));

   cudaStreamSynchronize(starpu_cuda_get_local_stream());
}


void func_cpu(void *buffers[], void *args)
{
    float *A = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    float *B = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
    float *C = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int block_total_size = g_block_width * g_block_width;
    for(int i=0; i<block_total_size; i++){
        C[i] = A[i] + B[i];
    }
}


struct starpu_codelet codelet_soma =
{
    .cpu_funcs = { func_cpu },
    .cuda_funcs = { func_gpu },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_W },
    .name = "soma_bloco",
    .where = STARPU_CUDA
};

int main(int argc, char *argv[]){

    int matrix_width = DEFAULT_MATRIX_WIDTH;
    int block_width = DEFAULT_BLOCK_WIDTH;
    if (argc > 1) {
        matrix_width = atoi(argv[1]);
    }
    if (argc > 2) {
        block_width = atoi(argv[2]);
    }
    if (matrix_width <= 0 || block_width <= 0 || matrix_width % block_width != 0) {
        fprintf(stderr, "Usage: %s [matrix_width] [block_width]\n", argv[0]);
        fprintf(stderr, "  matrix_width must be a positive multiple of block_width; got matrix_width=%d block_width=%d\n",
                matrix_width, block_width);
        fprintf(stderr, "  number of tasks = (matrix_width/block_width)^2 -- scale block_width up together\n");
        fprintf(stderr, "  with matrix_width, or a bigger matrix turns into millions of tiny tasks.\n");
        return 1;
    }
    g_block_width = block_width;
    int number_blocks_width = matrix_width / block_width;
    int number_blocks = number_blocks_width * number_blocks_width;
    fprintf(stderr, "Matrix order: %dx%d (%d blocks of %dx%d)\n",
            matrix_width, matrix_width, number_blocks, block_width, block_width);

    cublasCreate(&cublas_mainhandle);

    starpu_init(NULL);
    starpu_topology_print(stdout);

    float** matrix_a = malloc(number_blocks * sizeof(float*));
    float** matrix_b = malloc(number_blocks * sizeof(float*));
    float** matrix_c = malloc(number_blocks * sizeof(float*));

    starpu_data_handle_t* matrix_a_handle = malloc(number_blocks * sizeof(starpu_data_handle_t));
    starpu_data_handle_t* matrix_b_handle = malloc(number_blocks * sizeof(starpu_data_handle_t));
    starpu_data_handle_t* matrix_c_handle = malloc(number_blocks * sizeof(starpu_data_handle_t));

    for(int i=0; i<number_blocks; i++){
        matrix_a[i] = (float*)malloc(block_width * block_width * sizeof(float));
        matrix_b[i] = (float*)malloc(block_width * block_width * sizeof(float));
        matrix_c[i] = (float*)malloc(block_width * block_width * sizeof(float));
    }

    for(int i=0; i<matrix_width; i++){
        for(int y=0; y<matrix_width; y++){
            int bi = i/block_width;
            int by = y/block_width;
            int ci = i%block_width;
            int cy = y%block_width;
            matrix_a[by * number_blocks_width + bi][cy * block_width + ci] = 1;
            matrix_b[by * number_blocks_width + bi][cy * block_width + ci] = 2;
        }
    }

    for(int i=0; i<number_blocks; i++){
        starpu_matrix_data_register(&matrix_a_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_a[i], block_width, block_width, block_width, sizeof(float));
        starpu_matrix_data_register(&matrix_b_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_b[i], block_width, block_width, block_width, sizeof(float));
        starpu_matrix_data_register(&matrix_c_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_c[i], block_width, block_width, block_width, sizeof(float));
    }

    for(int i=0; i<number_blocks; i++){
        starpu_task_insert(&codelet_soma,
              STARPU_R, matrix_a_handle[i],
              STARPU_R, matrix_b_handle[i],
              STARPU_W, matrix_c_handle[i],
              0);
    }

    starpu_task_wait_for_all();

    for(int i=0; i<number_blocks; i++){
        starpu_data_unregister(matrix_a_handle[i]);
        starpu_data_unregister(matrix_b_handle[i]);
        starpu_data_unregister(matrix_c_handle[i]);
    }

    starpu_shutdown();

    /* Dumping every element only makes sense for small demo sizes;
     * for a bigger matrix_width just confirm correctness instead of
     * flooding the terminal. */
    if (matrix_width <= 20) {
        for(int i=0; i<matrix_width; i++){
            for(int y=0; y<matrix_width; y++){
                int bi = i/block_width, by = y/block_width;
                int ci = i%block_width, cy = y%block_width;
                printf("%f ", matrix_c[by * number_blocks_width + bi][cy * block_width + ci]);
            }
            printf("\n");
        }
    } else {
        int bi = 0, by = 0;
        fprintf(stderr, "First block[0][0..%d]: ", block_width-1);
        for(int ci=0; ci<block_width; ci++){
            fprintf(stderr, "%f ", matrix_c[by * number_blocks_width + bi][ci]);
        }
        fprintf(stderr, "(expected 3.000000 everywhere)\n");
    }

    for(int i=0; i<number_blocks; i++){
        free(matrix_a[i]);
        free(matrix_b[i]);
        free(matrix_c[i]);
    }
    free(matrix_a);
    free(matrix_b);
    free(matrix_c);
    free(matrix_a_handle);
    free(matrix_b_handle);
    free(matrix_c_handle);

    return 0;
}
