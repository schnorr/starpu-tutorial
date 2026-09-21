#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <starpu.h>

/* Reference solution for the Session 3, Part A exercise (see
 * ../../handout.org): same block matrix multiplication as
 * ../exercise/multi_matrix.c, with a second CPU implementation
 * (func_cpu_ikj) added to the codelet — loop order i,k,j instead of
 * i,j,k, same result, different memory access pattern.
 *
 * Matrix order and block width are runtime arguments (argv[1] and
 * argv[2]) instead of compile-time #defines, so this can be scaled up
 * without editing the source and rebuilding. This is *matrix
 * multiplication*, not addition, so the number of blocks/tasks is
 * (matrix_width/block_width)^3, not ^2 — an even steeper relationship
 * than the addition demo in ../demo/. Scale block_width up together
 * with matrix_width, or a bigger matrix turns into an explosion of
 * tiny tasks (and O(N^3) of them). The defaults (1000, 200) match the
 * original exercise size (125 tasks). */
#define DEFAULT_MATRIX_WIDTH 1000
#define DEFAULT_BLOCK_WIDTH 200

/* Set once in main() before any task is submitted; codelets read it. */
static int g_block_width;

void func_cpu(void *buffers[], void *args)
{
    float *A = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    float *B = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
    float *C = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int block_width = g_block_width;
    for(int i=0; i<block_width; i++){
        for(int j=0; j<block_width; j++){
            for(int k=0; k<block_width; k++){
                C[j * block_width + i] += A[j * block_width + k] * B[k * block_width + i];
            }
        }
    }
}

/* Same computation, loop order i,k,j instead of i,j,k. */
void func_cpu_ikj(void *buffers[], void *args)
{
    float *A = (float *)STARPU_VECTOR_GET_PTR(buffers[0]);
    float *B = (float *)STARPU_VECTOR_GET_PTR(buffers[1]);
    float *C = (float *)STARPU_VECTOR_GET_PTR(buffers[2]);

    int block_width = g_block_width;
    for(int i=0; i<block_width; i++){
        for(int k=0; k<block_width; k++){
            for(int j=0; j<block_width; j++){
                C[j * block_width + i] += A[j * block_width + k] * B[k * block_width + i];
            }
        }
    }
}

struct starpu_codelet codelet_multi =
{
    .cpu_funcs = { func_cpu, func_cpu_ikj },
    .cpu_funcs_name = { "func_cpu", "func_cpu_ikj" },
    .nbuffers = 3,
    .modes = { STARPU_R, STARPU_R, STARPU_W },
	.where = STARPU_CPU,
	.name = "multiplica_bloco"
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
        fprintf(stderr, "  number of tasks = (matrix_width/block_width)^3 -- this is matrix\n");
        fprintf(stderr, "  MULTIPLICATION, so it grows even faster than the addition demo does;\n");
        fprintf(stderr, "  scale block_width up together with matrix_width.\n");
        return 1;
    }
    g_block_width = block_width;
    int number_blocks_width = matrix_width / block_width;
    int number_blocks = number_blocks_width * number_blocks_width;
    long number_tasks = (long)number_blocks_width * number_blocks_width * number_blocks_width;
    fprintf(stderr, "Matrix order: %dx%d (%d blocks of %dx%d, %ld tasks)\n",
            matrix_width, matrix_width, number_blocks, block_width, block_width, number_tasks);

    starpu_init(NULL);

    float** matrix_a = malloc(number_blocks * sizeof(float*));
    float** matrix_b = malloc(number_blocks * sizeof(float*));
    float** matrix_c = malloc(number_blocks * sizeof(float*));

    starpu_data_handle_t* matrix_a_handle = malloc(number_blocks * sizeof(starpu_data_handle_t));
    starpu_data_handle_t* matrix_b_handle = malloc(number_blocks * sizeof(starpu_data_handle_t));
    starpu_data_handle_t* matrix_c_handle = malloc(number_blocks * sizeof(starpu_data_handle_t));

    for(int i=0; i<number_blocks; i++){
        matrix_a[i] = (float*)malloc(block_width * block_width * sizeof(float));
        starpu_matrix_data_register(&matrix_a_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_a[i], block_width, block_width, block_width, sizeof(float));
	starpu_data_set_coordinates(matrix_a_handle[i], 2, i/number_blocks_width, i%number_blocks_width);

        matrix_b[i] = (float*)malloc(block_width * block_width * sizeof(float));
        starpu_matrix_data_register(&matrix_b_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_b[i], block_width, block_width, block_width, sizeof(float));
	starpu_data_set_coordinates(matrix_b_handle[i], 2, i/number_blocks_width, i%number_blocks_width);

        matrix_c[i] = (float*)malloc(block_width * block_width * sizeof(float));
        starpu_matrix_data_register(&matrix_c_handle[i], STARPU_MAIN_RAM, (uintptr_t)matrix_c[i], block_width, block_width, block_width, sizeof(float));
	starpu_data_set_coordinates(matrix_c_handle[i], 2, i/number_blocks_width, i%number_blocks_width);

    }

    for(int i=0; i<matrix_width; i++){
        for(int y=0; y<matrix_width; y++){
            int bi = i/block_width;
            int by = y/block_width;
            int ci = i%block_width;
            int cy = y%block_width;
            matrix_a[by * number_blocks_width + bi][cy * block_width + ci] = 1;
            matrix_b[by * number_blocks_width + bi][cy * block_width + ci] = 2;
            matrix_c[by * number_blocks_width + bi][cy * block_width + ci] = 0;
        }
    }

    for(int i=0; i<number_blocks_width; i++){
        for(int j=0; j<number_blocks_width; j++){
            for(int k=0; k<number_blocks_width; k++){
                starpu_task_insert(&codelet_multi,
                    STARPU_R, matrix_a_handle[j * number_blocks_width + k],
                    STARPU_R, matrix_b_handle[k * number_blocks_width + i],
                    STARPU_W, matrix_c_handle[j * number_blocks_width + i],
                    0);
            }
        }
    }

    starpu_task_wait_for_all();

    for(int i=0; i<number_blocks; i++){
        starpu_data_unregister(matrix_a_handle[i]);
        starpu_data_unregister(matrix_b_handle[i]);
        starpu_data_unregister(matrix_c_handle[i]);
    }

    starpu_shutdown();

    /* Dumping every element only makes sense for small demo sizes; for
     * a bigger matrix_width just confirm correctness instead of
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
        fprintf(stderr, "(expected %d.000000 everywhere)\n", 2*matrix_width);
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
