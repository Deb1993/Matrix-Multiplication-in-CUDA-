/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <emmintrin.h>

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
void do_block (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
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

/* void do_block_vector (int lda, int M, int N, int K, double*  restrict A, double* restrict B, double* restrict C)
{
    // For each row i of A
         for (int i = 0; i < M; i = i + 2)
                // For each column j of B 
                        for (int j = 0; j < N; j = j + 2) 
                                {
                                            for (int k = 0; k < K; k = k + 2)
                                                         	do_vector(lda, i, j , k, A + i*lda + k, B + k*lda + j, C + i*lda + j);
                                                         	        }
    }
*/


void do_vector (int lda, int ii, int jj, int kk, double* restrict A, double* restrict B, double* restrict C)
{
			register __m128d c00_c01 = _mm_loadu_pd(C);
			register __m128d c10_c11 = _mm_loadu_pd(C + lda);

				for (int i = 0; i < 2; ++i)
					{
						__m128d a1 = _mm_load1_pd(A + i);  
						__m128d	a2 = _mm_load1_pd(A + i + lda);
						__m128d b  = _mm_load_pd(B + i*lda);
						c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a1,b));
						c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a2,b));
				}

_mm_storeu_pd(C, c00_c01);
_mm_storeu_pd(C + lda, c10_c11);

}

void do_block_vector (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
{
	for (int i = 0; i < M; i = i + 2)
		for( int j = 0 ; j < N ; j = j + 2)
			{
				for( int k = 0 ; k < K ; k = k + 2)
					do_vector(lda, i, j , k ,A + i*lda + k, B + k*lda + j, C + i*lda + j);
				}
}

void do_block1 (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
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
		if(M_1 && 1 == 0 && N_1 && 1 == 0 && K_1 && 1 == 0)
			do_block_vector(lda,M_1,N_1,K_1, A + i*lda + k, B + k*lda + j, C + i*lda + j);
		else
                	do_block(lda, M_1, N_1, K_1, A + i*lda + k, B + k*lda + j, C + i*lda + j);
            }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
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
