#include <stdlib.h>
#include <limits.h>
#include <starpu.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MATRIX_WIDTH 10
#define BLOCK_WIDTH 2
#define BLOCK_TOTAL_SIZE (BLOCK_WIDTH*BLOCK_WIDTH)
#define NUMBER_BLOCKS_WIDTH (MATRIX_WIDTH/BLOCK_WIDTH)
#define NUMBER_BLOCKS (NUMBER_BLOCKS_WIDTH*NUMBER_BLOCKS_WIDTH)

#define CHECK_CUBLAS(x) if(x!=CUBLAS_STATUS_SUCCESS){printf("Cublass error: %d\n", x);};
cublasHandle_t cublas_mainhandle;

void func_gpu(void *buffers[], void *args)
{
    float alpha_beta = 1;
    float *A = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    float *B = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
    float *C = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);

    cublasSetStream(cublas_mainhandle, starpu_cuda_get_local_stream());
    CHECK_CUBLAS(cublasSgeam(cublas_mainhandle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                BLOCK_WIDTH, BLOCK_WIDTH,
                &alpha_beta,
                A,
                BLOCK_WIDTH,
                &alpha_beta,
                B,
                BLOCK_WIDTH,
                C,
                BLOCK_WIDTH));

   cudaStreamSynchronize(starpu_cuda_get_local_stream());
}


void func_cpu(void *buffers[], void *args)
{
    float *A = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    float *B = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
    float *C = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);

    for(int i=0; i<BLOCK_TOTAL_SIZE; i++){
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

int main(){

    cublasCreate(&cublas_mainhandle);

    starpu_init(NULL);
    starpu_topology_print(stdout);

    float* matrix_a[NUMBER_BLOCKS];
    float* matrix_b[NUMBER_BLOCKS];
    float* matrix_c[NUMBER_BLOCKS];

    starpu_data_handle_t matrix_a_handle[NUMBER_BLOCKS];
    starpu_data_handle_t matrix_b_handle[NUMBER_BLOCKS];
    starpu_data_handle_t matrix_c_handle[NUMBER_BLOCKS];

    for(int i=0; i<NUMBER_BLOCKS; i++){
        matrix_a[i] = (float*)malloc(BLOCK_WIDTH * BLOCK_WIDTH * sizeof(float));
        matrix_b[i] = (float*)malloc(BLOCK_WIDTH * BLOCK_WIDTH * sizeof(float));
        matrix_c[i] = (float*)malloc(BLOCK_WIDTH * BLOCK_WIDTH * sizeof(float));
    }

    for(int i=0; i<MATRIX_WIDTH; i++){
        for(int y=0; y<MATRIX_WIDTH; y++){
            int bi = i/BLOCK_WIDTH;
            int by = y/BLOCK_WIDTH;
            int ci = i%BLOCK_WIDTH;
            int cy = y%BLOCK_WIDTH;
            matrix_a[by * NUMBER_BLOCKS_WIDTH + bi][cy * BLOCK_WIDTH + ci] = 1;
            matrix_b[by * NUMBER_BLOCKS_WIDTH + bi][cy * BLOCK_WIDTH + ci] = 2;
        }
    }

    for(int i=0; i<NUMBER_BLOCKS; i++){
        starpu_matrix_data_register(&matrix_a_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_a[i], BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH, sizeof(float));
        starpu_matrix_data_register(&matrix_b_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_b[i], BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH, sizeof(float));
        starpu_matrix_data_register(&matrix_c_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_c[i], BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH, sizeof(float));
    }

    for(int i=0; i<NUMBER_BLOCKS; i++){
        starpu_task_insert(&codelet_soma,
              STARPU_R, matrix_a_handle[i],
              STARPU_R, matrix_b_handle[i],
              STARPU_W, matrix_c_handle[i],
              0);
    }

    starpu_task_wait_for_all();

    for(int i=0; i<NUMBER_BLOCKS; i++){
        starpu_data_unregister(matrix_a_handle[i]);
        starpu_data_unregister(matrix_b_handle[i]);
        starpu_data_unregister(matrix_c_handle[i]);
    }

    starpu_shutdown();

    for(int i=0; i<MATRIX_WIDTH; i++){
        for(int y=0; y<MATRIX_WIDTH; y++){
            int bi = i/BLOCK_WIDTH;
            int by = y/BLOCK_WIDTH;
            int ci = i%BLOCK_WIDTH;
            int cy = y%BLOCK_WIDTH;
            printf("%f ", matrix_c[by * NUMBER_BLOCKS_WIDTH + bi][cy * BLOCK_WIDTH + ci]);
        }
        printf("\n");
    }

}
