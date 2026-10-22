/* Sequential tiled matrix multiplication C = C + A * B.
   Matrices are N x N, column-major, split into NB x NB blocks of size BS. */
#include <stdio.h>
#include <stdlib.h>

/* C_blk += A_blk * B_blk on BS x BS blocks inside matrices of leading dimension ld */
static void gemm_block(unsigned bs, unsigned ld,
                       const float *A, const float *B, float *C)
{
    for (unsigned j = 0; j < bs; j++)
        for (unsigned l = 0; l < bs; l++) {
            float b = B[l + j * ld];
            for (unsigned i = 0; i < bs; i++)
                C[i + j * ld] += A[i + l * ld] * b;
        }
}

int main(int argc, char *argv[])
{
    unsigned N  = argc > 1 ? atoi(argv[1]) : 2048;
    unsigned NB = argc > 2 ? atoi(argv[2]) : 8;
    unsigned BS = N / NB;

    float *A = malloc((size_t)N * N * sizeof(float));
    float *B = malloc((size_t)N * N * sizeof(float));
    float *C = malloc((size_t)N * N * sizeof(float));
    for (size_t x = 0; x < (size_t)N * N; x++) {
        A[x] = 1.0f; B[x] = 2.0f; C[x] = 0.0f;
    }

#define BLK(M, i, j) (&(M)[(size_t)(i) * BS + (size_t)(j) * BS * N])
    for (unsigned i = 0; i < NB; i++)
        for (unsigned j = 0; j < NB; j++)
            for (unsigned k = 0; k < NB; k++)
                gemm_block(BS, N, BLK(A, i, k), BLK(B, k, j), BLK(C, i, j));

    printf("C[0] = %.1f (expected %.1f)\n", C[0], 2.0f * N);
    free(A); free(B); free(C);
    return 0;
}
