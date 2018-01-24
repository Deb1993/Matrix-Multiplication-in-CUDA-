/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 16
#define BLOCK_SIZE_2 37
// #define BLOCK_SIZE 719
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i)
        /* For each column j of B */ 
        for (int j = 0; j < N; ++j) 
        {
            /* Compute C(i,j) */
            double cij = C[i*lda+j];
            for (int k = 0; k < K; ++k)
#ifdef TRANSPOSE
                cij += A[i*lda+k] * B[j*lda+k];
#else
            cij += A[i*lda+k] * B[k*lda+j];
#endif
            C[i*lda+j] = cij;
        }
}


void do_block1 (int lda, int M, int N, int K, double *A, double *B, double *C)
{
    /* For each block-row of A */ 
    for (int i = 0; i < M; i += BLOCK_SIZE)
        /* For each block-column of B */
        for (int j = 0; j < N; j += BLOCK_SIZE)
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < K; k += BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M_1 = min (BLOCK_SIZE, M-i);
                int N_1 = min (BLOCK_SIZE, N-j);
                int K_1 = min (BLOCK_SIZE, K-k);

                /* Perform individual block dgemm */
                do_block(lda, M_1, N_1, K_1, A + i*lda + k, B + k*lda + j, C + i*lda + j);
            }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
    /* For each block-row of A */ 
    for (int i = 0; i < lda; i += BLOCK_SIZE_2)
        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE_2)
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += BLOCK_SIZE_2)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min (BLOCK_SIZE_2, lda-i);
                int N = min (BLOCK_SIZE_2, lda-j);
                int K = min (BLOCK_SIZE_2, lda-k);

                /* Perform individual block dgemm */
                do_block1(lda, M, N, K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
            }
#ifdef TRANSPOSE
    for (int i = 0; i < lda; ++i)
        for (int j = i+1; j < lda; ++j) {
            double t = B[i*lda+j];
            B[i*lda+j] = B[j*lda+i];
            B[j*lda+i] = t;
        }
#endif
}
