#include <stdlib.h>
#include <limits.h>
#include <starpu.h>

/* Reference solution for the Session 2, Part B exercise (see
 * ../../../handout.org), matching the skeleton in ../skeleton/matrix_add.c. */

#define MATRIX_WIDTH 10
#define BLOCK_WIDTH 2
#define BLOCK_TOTAL_SIZE (BLOCK_WIDTH*BLOCK_WIDTH)
#define NUMBER_BLOCKS_WIDTH (MATRIX_WIDTH/BLOCK_WIDTH)
#define NUMBER_BLOCKS (NUMBER_BLOCKS_WIDTH*NUMBER_BLOCKS_WIDTH)

void func_cpu(void *buffers[], void *args)
{
    float *A = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    float *B = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
    float *C = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);

    for(int i=0; i<BLOCK_TOTAL_SIZE; i++){
        C[i] = A[i] + B[i];
    }
}

struct starpu_codelet codelet_add =
{
    .cpu_funcs = { func_cpu },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_W },
    .name = "block_add",
};

int main(){

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
            int bi = i/BLOCK_WIDTH, by = y/BLOCK_WIDTH;
            int ci = i%BLOCK_WIDTH, cy = y%BLOCK_WIDTH;
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
        starpu_task_insert(&codelet_add,
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
            int bi = i/BLOCK_WIDTH, by = y/BLOCK_WIDTH;
            int ci = i%BLOCK_WIDTH, cy = y%BLOCK_WIDTH;
            printf("%f ", matrix_c[by * NUMBER_BLOCKS_WIDTH + bi][cy * BLOCK_WIDTH + ci]);
        }
        printf("\n");
    }
}
