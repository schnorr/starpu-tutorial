#include <stdlib.h>
#include <limits.h>
#include <starpu.h>

#define MATRIX_WIDTH 1000
#define BLOCK_WIDTH 200
#define BLOCK_TOTAL_SIZE (BLOCK_WIDTH*BLOCK_WIDTH)
#define NUMBER_BLOCKS_WIDTH (MATRIX_WIDTH/BLOCK_WIDTH)
#define NUMBER_BLOCKS (NUMBER_BLOCKS_WIDTH*NUMBER_BLOCKS_WIDTH)

void func_cpu(void *buffers[], void *args)
{
    float *A = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    float *B = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
    float *C = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);

    for(int i=0; i<BLOCK_WIDTH; i++){
        for(int j=0; j<BLOCK_WIDTH; j++){
            for(int k=0; k<BLOCK_WIDTH; k++){
                C[j * BLOCK_WIDTH + i] += A[j * BLOCK_WIDTH + k] * B[k * BLOCK_WIDTH + i];
            }
        }
    }
}

struct starpu_codelet codelet_multi =
{
    .cpu_funcs = { func_cpu },
    .cpu_funcs_name = { "func_cpu" },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_W },
	.where = STARPU_CPU,
	.name = "multiplica_bloco"
};

int main(){

    starpu_init(NULL);

    float* matrix_a[NUMBER_BLOCKS];
    float* matrix_b[NUMBER_BLOCKS];
    float* matrix_c[NUMBER_BLOCKS];

    starpu_data_handle_t matrix_a_handle[NUMBER_BLOCKS];
    starpu_data_handle_t matrix_b_handle[NUMBER_BLOCKS];
    starpu_data_handle_t matrix_c_handle[NUMBER_BLOCKS];

    for(int i=0; i<NUMBER_BLOCKS; i++){
        matrix_a[i] = (float*)malloc(BLOCK_WIDTH * BLOCK_WIDTH * sizeof(float));
        starpu_matrix_data_register(&matrix_a_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_a[i], BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH, sizeof(float));
	starpu_data_set_coordinates(matrix_a_handle[i], 2, i/NUMBER_BLOCKS_WIDTH, i%NUMBER_BLOCKS_WIDTH); 

        matrix_b[i] = (float*)malloc(BLOCK_WIDTH * BLOCK_WIDTH * sizeof(float));
        starpu_matrix_data_register(&matrix_b_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_b[i], BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH, sizeof(float));
	starpu_data_set_coordinates(matrix_b_handle[i], 2, i/NUMBER_BLOCKS_WIDTH, i%NUMBER_BLOCKS_WIDTH); 

        matrix_c[i] = (float*)malloc(BLOCK_WIDTH * BLOCK_WIDTH * sizeof(float));
        starpu_matrix_data_register(&matrix_c_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_c[i], BLOCK_WIDTH, BLOCK_WIDTH, BLOCK_WIDTH, sizeof(float));
	starpu_data_set_coordinates(matrix_c_handle[i], 2, i/NUMBER_BLOCKS_WIDTH, i%NUMBER_BLOCKS_WIDTH); 

    }

    for(int i=0; i<MATRIX_WIDTH; i++){
        for(int y=0; y<MATRIX_WIDTH; y++){
            int bi = i/BLOCK_WIDTH;
            int by = y/BLOCK_WIDTH;
            int ci = i%BLOCK_WIDTH;
            int cy = y%BLOCK_WIDTH;
            matrix_a[by * NUMBER_BLOCKS_WIDTH + bi][cy * BLOCK_WIDTH + ci] = 1;
            matrix_b[by * NUMBER_BLOCKS_WIDTH + bi][cy * BLOCK_WIDTH + ci] = 2;
            matrix_c[by * NUMBER_BLOCKS_WIDTH + bi][cy * BLOCK_WIDTH + ci] = 0;
        }
    }

    for(int i=0; i<NUMBER_BLOCKS_WIDTH; i++){
        for(int j=0; j<NUMBER_BLOCKS_WIDTH; j++){
            for(int k=0; k<NUMBER_BLOCKS_WIDTH; k++){
                starpu_task_insert(&codelet_multi,
                    STARPU_R, matrix_a_handle[j * NUMBER_BLOCKS_WIDTH + k],
                    STARPU_R, matrix_b_handle[k * NUMBER_BLOCKS_WIDTH + i],
                    STARPU_W, matrix_c_handle[j * NUMBER_BLOCKS_WIDTH + i],
                    0);
            }
        }
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
