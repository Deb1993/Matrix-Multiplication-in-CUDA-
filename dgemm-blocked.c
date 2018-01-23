/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <emmintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#define L1_CACHE 32*1024
#define L2_CACHE 4*1024*1024
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 16
#define BLOCK_SIZE_2 576
// #define BLOCK_SIZE 719
#endif

#define min(a,b) (((a)<(b))?(a):(b))

int calculate_block_size(int n, int cache_size) {
    double num_blocks = n*sqrt(3/(1.0*cache_size));
    return num_blocks >= 1 ? sqrt(cache_size/3) : n;
}

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

void do_vector (int lda, double* restrict A, double* restrict B, double* restrict C)
{
			register __m128d c00_c01 = _mm_loadu_pd(C);
			register __m128d c10_c11 = _mm_loadu_pd(C + lda);
			register __m128d c20_c21 = _mm_loadu_pd(C + 2*lda);
			register __m128d c30_c31 = _mm_loadu_pd(C + 3*lda);
                        register __m128d c02_c03 = _mm_loadu_pd(C + 2);
                        register __m128d c12_c13 = _mm_loadu_pd(C + lda + 2);
                        register __m128d c22_c23 = _mm_loadu_pd(C + 2*lda + 2);
                        register __m128d c32_c33 = _mm_loadu_pd(C + 3*lda + 2);

				for (int i = 0; i < 4; i++)
					{
						register __m128d a1 = _mm_load1_pd(A + i);  
						register __m128d a2 = _mm_load1_pd(A + i + lda);
						register __m128d a3 = _mm_load1_pd(A + i + 2*lda);
						register __m128d a4 = _mm_load1_pd(A + i + 3*lda);
						register __m128d b  = _mm_loadu_pd(B + i*lda);
						//register __m128d a4 = _mm_load1_pd(A + i + 3*lda);
						register __m128d b1 = _mm_loadu_pd(B + i*lda + 2);
						c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a1,b));
						c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a2,b));
						c20_c21 = _mm_add_pd(c20_c21, _mm_mul_pd(a3,b));
						c30_c31 = _mm_add_pd(c30_c31, _mm_mul_pd(a4,b));
                                                c02_c03 = _mm_add_pd(c02_c03, _mm_mul_pd(a1,b1));
                                                c12_c13 = _mm_add_pd(c12_c13, _mm_mul_pd(a2,b1));
                                                c22_c23 = _mm_add_pd(c22_c23, _mm_mul_pd(a3,b1));
                                                c32_c33 = _mm_add_pd(c32_c33, _mm_mul_pd(a4,b1));
				}
_mm_storeu_pd(C, c00_c01);
_mm_storeu_pd(C + lda, c10_c11);
_mm_storeu_pd(C + 2*lda, c20_c21);
_mm_storeu_pd(C + 3*lda, c30_c31);
_mm_storeu_pd(C + 2, c02_c03);
_mm_storeu_pd(C + lda + 2, c12_c13);
_mm_storeu_pd(C + 2*lda + 2, c22_c23);
_mm_storeu_pd(C + 3*lda + 2, c32_c33);

}

void do_block_vector (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
{
	for (int i = 0; i < M; i = i + 4)
		for( int j = 0 ; j < N ; j = j + 4)
			{
				for( int k = 0 ; k < K ; k = k + 4) {
					do_vector(lda, A + i*lda + k, B + k*lda + j, C + i*lda + j);

				}
}
}

void do_block1 (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
{

    int block_size_2 = calculate_block_size(lda, L2_CACHE);
    int block_size = calculate_block_size(block_size_2, L1_CACHE);
    /* For each block-row of A */ 
    for (int i = 0; i < M; i += block_size)
        /* For each block-column of B */
        for (int j = 0; j < N; j += block_size)
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < K; k += block_size)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M_1 = min (block_size, M-i);
                int N_1 = min (block_size, N-j);
                int K_1 = min (block_size, K-k);

                /* Perform individual block dgemm */
		if((M_1 % 4) == 0 && (N_1 % 4) == 0 && (K_1 % 4) == 0)
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
    int block_size_2 = calculate_block_size(lda, L2_CACHE);
    printf("Block Size 2: %d\n", block_size_2);
    /* For each block-row of A */ 
    for (int i = 0; i < lda; i += block_size_2)
        /* For each block-column of B */
        for (int j = 0; j < lda; j += block_size_2)
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < lda; k += block_size_2)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min (block_size_2, lda-i);
                int N = min (block_size_2, lda-j);
                int K = min (block_size_2, lda-k);

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
