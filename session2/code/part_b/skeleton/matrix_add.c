#include <stdlib.h>
#include <limits.h>
#include <starpu.h>

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

    /* TODO 1: compute C[i] = A[i] + B[i] for every element of this block
     * (BLOCK_TOTAL_SIZE elements). */
}

struct starpu_codelet codelet_add =
{
    .cpu_funcs = { func_cpu },
    /* TODO 2: how many buffers does this codelet take, and with which
     * access modes (STARPU_R / STARPU_W / STARPU_RW)? Fill in .nbuffers
     * and .modes accordingly. */
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

    /* Fill A with 1s and B with 2s, block by block, at the right global
     * (i, y) position. Already done for you. */
    for(int i=0; i<MATRIX_WIDTH; i++){
        for(int y=0; y<MATRIX_WIDTH; y++){
            int bi = i/BLOCK_WIDTH, by = y/BLOCK_WIDTH;
            int ci = i%BLOCK_WIDTH, cy = y%BLOCK_WIDTH;
            matrix_a[by * NUMBER_BLOCKS_WIDTH + bi][cy * BLOCK_WIDTH + ci] = 1;
            matrix_b[by * NUMBER_BLOCKS_WIDTH + bi][cy * BLOCK_WIDTH + ci] = 2;
        }
    }

    for(int i=0; i<NUMBER_BLOCKS; i++){
        /* TODO 3: register matrix_a[i], matrix_b[i] and matrix_c[i] with
         * StarPU using starpu_matrix_data_register(), producing
         * matrix_a_handle[i]/matrix_b_handle[i]/matrix_c_handle[i].
         * Signature: starpu_matrix_data_register(&handle, STARPU_MAIN_RAM,
         *   (uintptr_t)pointer, ld, nx, ny, elemsize)
         * Here ld == nx == ny == BLOCK_WIDTH (blocks are stored
         * contiguously with no extra padding). */
    }

    for(int i=0; i<NUMBER_BLOCKS; i++){
        /* TODO 4: submit one task per block with starpu_task_insert(),
         * passing &codelet_add and the three handles for block i with
         * the correct access mode for each (matching TODO 2). Don't
         * forget the trailing 0. */
    }

    starpu_task_wait_for_all();

    for(int i=0; i<NUMBER_BLOCKS; i++){
        starpu_data_unregister(matrix_a_handle[i]);
        starpu_data_unregister(matrix_b_handle[i]);
        starpu_data_unregister(matrix_c_handle[i]);
    }

    starpu_shutdown();

    /* Print the result: every value should be 3.0 */
    for(int i=0; i<MATRIX_WIDTH; i++){
        for(int y=0; y<MATRIX_WIDTH; y++){
            int bi = i/BLOCK_WIDTH, by = y/BLOCK_WIDTH;
            int ci = i%BLOCK_WIDTH, cy = y%BLOCK_WIDTH;
            printf("%f ", matrix_c[by * NUMBER_BLOCKS_WIDTH + bi][cy * BLOCK_WIDTH + ci]);
        }
        printf("\n");
    }
}
