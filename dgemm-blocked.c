/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#define _GNU_SOURCE_
#include <math.h>
#include <stdio.h>
#include <x86intrin.h>
#include <string.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#define BLOCK_SIZE_2_K 32
#define BLOCK_SIZE_2 416
#endif


#define min(a,b) (((a)<(b))?(a):(b))

static double A_padded[BLOCK_SIZE_2*BLOCK_SIZE_2_K] __attribute__ ((aligned (16)));
static double B_padded[BLOCK_SIZE_2*BLOCK_SIZE_2_K] __attribute__ ((aligned (16)));
static double C_padded[BLOCK_SIZE_2*BLOCK_SIZE_2] __attribute__ ((aligned (16)));

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{
    /* For each row i of A */
    for (int i = 0; i < M; ++i) {
        /* For each column j of B */ 
        for (int j = 0; j < N; ++j) 
        {
            /* Compute C(i,j) */
            double cij = C[i*lda+j];
            for (int k = 0; k < K;) {
                cij += A[k*lda+i] * B[k*lda+j];
                k+=1;
                if(k < K)
                    cij += A[k*lda+i] * B[k*lda+j];
                k+=1;
                if(k < K)
                    cij += A[k*lda+i] * B[k*lda+j];
                k+=1;
                if(k < K)
                    cij += A[k*lda+i] * B[k*lda+j];
                k+=1;
            __builtin_prefetch(B + (k+1)*lda);
	    __builtin_prefetch(A + (k+1)*lda);
            }
            C[i*lda+j] = cij;
        }
    __builtin_prefetch(C + (i+1)*lda);
    }
}

void do_vector (int lda, double* restrict A, double* restrict B, double* restrict C)
{			
			register __m128d c00_c01 = _mm_load_pd(C);
			//printf("Hello\n");
			register __m128d c10_c11 = _mm_load_pd(C + lda);
			register __m128d c20_c21 = _mm_load_pd(C + 2*lda);
			register __m128d c30_c31 = _mm_load_pd(C + 3*lda);
                        register __m128d c02_c03 = _mm_load_pd(C + 2);
                        register __m128d c12_c13 = _mm_load_pd(C + lda + 2);
                        register __m128d c22_c23 = _mm_load_pd(C + 2*lda + 2);
                        register __m128d c32_c33 = _mm_load_pd(C + 3*lda + 2);
    
        register __m128d b  = _mm_load_pd(B);
        register __m128d b1 = _mm_load_pd(B + 2);

        register __m128d a1 = _mm_load1_pd(A);
        register __m128d a2 = _mm_load1_pd(A + 1);
        c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a1,b));
        c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a2,b));

        register __m128d a3 = _mm_load1_pd(A + 2);
        register __m128d a4 = _mm_load1_pd(A + 3);
        c20_c21 = _mm_add_pd(c20_c21, _mm_mul_pd(a3,b));
        c30_c31 = _mm_add_pd(c30_c31, _mm_mul_pd(a4,b));

        c02_c03 = _mm_add_pd(c02_c03, _mm_mul_pd(a1,b1));
        c12_c13 = _mm_add_pd(c12_c13, _mm_mul_pd(a2,b1));
        c22_c23 = _mm_add_pd(c22_c23, _mm_mul_pd(a3,b1));
        c32_c33 = _mm_add_pd(c32_c33, _mm_mul_pd(a4,b1));


        b  = _mm_load_pd(B + 1*lda);
        b1 = _mm_load_pd(B + 1*lda + 2);

        a1 = _mm_load1_pd(A + 1*lda);
        a2 = _mm_load1_pd(A + 1*lda + 1);
        c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a1,b));
        c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a2,b));

        a3 = _mm_load1_pd(A + 1*lda + 2);
        a4 = _mm_load1_pd(A + 1*lda + 3);
        c20_c21 = _mm_add_pd(c20_c21, _mm_mul_pd(a3,b));
        c30_c31 = _mm_add_pd(c30_c31, _mm_mul_pd(a4,b));


        c02_c03 = _mm_add_pd(c02_c03, _mm_mul_pd(a1,b1));
        c12_c13 = _mm_add_pd(c12_c13, _mm_mul_pd(a2,b1));


        c22_c23 = _mm_add_pd(c22_c23, _mm_mul_pd(a3,b1));
        c32_c33 = _mm_add_pd(c32_c33, _mm_mul_pd(a4,b1));

        b  = _mm_load_pd(B + 2*lda);
        b1 = _mm_load_pd(B + 2*lda + 2);

        a1 = _mm_load1_pd(A + 2*lda);
        a2 = _mm_load1_pd(A + 2*lda + 1);
        c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a1,b));
        c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a2,b));

        a3 = _mm_load1_pd(A + 2*lda + 2);
        a4 = _mm_load1_pd(A + 2*lda + 3);
        c20_c21 = _mm_add_pd(c20_c21, _mm_mul_pd(a3,b));
        c30_c31 = _mm_add_pd(c30_c31, _mm_mul_pd(a4,b));


        c02_c03 = _mm_add_pd(c02_c03, _mm_mul_pd(a1,b1));
        c12_c13 = _mm_add_pd(c12_c13, _mm_mul_pd(a2,b1));


        c22_c23 = _mm_add_pd(c22_c23, _mm_mul_pd(a3,b1));
        c32_c33 = _mm_add_pd(c32_c33, _mm_mul_pd(a4,b1));

	b  = _mm_load_pd(B + 3*lda);
        b1 = _mm_load_pd(B + 3*lda + 2);

        a1 = _mm_load1_pd(A + 3*lda);
        a2 = _mm_load1_pd(A + 3*lda + 1);
        c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a1,b));
        c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a2,b));

        a3 = _mm_load1_pd(A + 3*lda + 2);
        a4 = _mm_load1_pd(A + 3*lda + 3);
        c20_c21 = _mm_add_pd(c20_c21, _mm_mul_pd(a3,b));
        c30_c31 = _mm_add_pd(c30_c31, _mm_mul_pd(a4,b));


        c02_c03 = _mm_add_pd(c02_c03, _mm_mul_pd(a1,b1));
        c12_c13 = _mm_add_pd(c12_c13, _mm_mul_pd(a2,b1));


        c22_c23 = _mm_add_pd(c22_c23, _mm_mul_pd(a3,b1));
        c32_c33 = _mm_add_pd(c32_c33, _mm_mul_pd(a4,b1));
			
			_mm_store_pd(C, c00_c01);
			_mm_store_pd(C + lda, c10_c11);
			_mm_store_pd(C + 2*lda, c20_c21);
			_mm_store_pd(C + 3*lda, c30_c31);
			_mm_store_pd(C + 2, c02_c03);
			_mm_store_pd(C + lda + 2, c12_c13);
			_mm_store_pd(C + 2*lda + 2, c22_c23);
			_mm_store_pd(C + 3*lda + 2, c32_c33);

}


void do_block_vector (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
{
	for (int i = 0; i < M; i = i + 4)
		for( int j = 0 ; j < N ; j = j + 4)
			{
				for( int k = 0 ; k < K ; k = k + 4) {
					do_vector(lda, A + k*lda + i, B + k*lda + j, C + i*lda + j);
				}
}
}

void do_block1 (int lda, int M, int N, int K, double *restrict A, double *restrict B, double *restrict C){
 
     for (int i = 0; i < M; i += BLOCK_SIZE) {
         int M_1 = min (BLOCK_SIZE, M-i);
         for (int j = 0; j < N; j += BLOCK_SIZE) {
             int N_1 = min (BLOCK_SIZE, N-j);
             for (int k = 0; k < K; k += BLOCK_SIZE)
             {
                 int K_1 = min (BLOCK_SIZE, K-k);
                 if((M_1 % 4) == 0 && (N_1 % 4) == 0 && (K_1 % 4) == 0)
                     do_block_vector(lda,M_1,N_1,K_1, A + k*lda + i, B + k*lda + j, C + i*lda + j);
                 else
                     do_block(lda, M_1, N_1, K_1, A + k*lda + i, B + k*lda + j, C + i*lda + j);

             }
         }
		    __builtin_prefetch(C + (i+BLOCK_SIZE)*lda);
     }
 }

void do_transpose(int lda, double *A) {
    for(int i = 0; i < lda; i+=1) {
        for(int j = i+1; j < lda; j+=1) {
            double  temp = A[i*lda + j];
            A[i*lda + j] = A[j*lda + i];
            A[j*lda + i] = temp;
        }
        __builtin_prefetch(A + (i+1)*lda);
    }
}


void do_copy(int lda, int M, int N, double *C) {

    double *src = C_padded, *dst = C;
    for(int i = 0; i < M; i+=1) {
        memcpy(dst, src, N*sizeof(double));
        src+=BLOCK_SIZE_2;
        dst+=lda;
        __builtin_prefetch(src);
    }
}

void pad_matrices(int lda, int M, int N, int K, double *A, double *B) {

    memset(A_padded, 0, BLOCK_SIZE_2*BLOCK_SIZE_2_K*sizeof(double));
    memset(B_padded, 0, BLOCK_SIZE_2*BLOCK_SIZE_2_K*sizeof(double));
 
     double *src = A;
     double *dst = A_padded;
     for(int i = 0; i < K; i+=1) {
         memcpy(dst, src, M*sizeof(double));
         src+=lda;
         dst+=BLOCK_SIZE_2;
        __builtin_prefetch(src);
     }
     src = B;
     dst = B_padded;
     for(int i = 0; i < K; i+=1) {
         memcpy(dst, src, N*sizeof(double));
         src+=lda;
         dst+=BLOCK_SIZE_2;
        __builtin_prefetch(src);
     }
}

void pad_C(int lda, int M, int N, double *C) {
    double *src = C, *dst = C_padded;
    for(int i = 0; i < M; i+=1) {
        memcpy(dst, src, N*sizeof(double));
        src+=lda;
        dst+=BLOCK_SIZE_2;
        __builtin_prefetch(src);
    }
}


void printMatrix(double *A, int M, int N) {
    for(int i = 0; i < M; i+=1) {
        for(int j = 0; j < N; j+=1) {
            printf("%f ", A[i*M + j]);
        }
            printf("\n");
    }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double *restrict A, double *restrict B, double *restrict C)
{
    do_transpose(lda, A);

    /* For each block-row of A */ 
    for (int i = 0; i < lda; i += BLOCK_SIZE_2) {
        /* For each block-column of B */
        for (int j = 0; j < lda; j += BLOCK_SIZE_2) {
            /* Accumulate block dgemms into block of C */
            int M = min (BLOCK_SIZE_2, lda-i);
            int N = min (BLOCK_SIZE_2, lda-j);
            pad_C(lda, M, N, C + i*lda + j);
            for (int k = 0; k < lda; k += BLOCK_SIZE_2_K)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int K = min (BLOCK_SIZE_2_K, lda-k);
                pad_matrices(lda, M, N, K, A + k*lda + i, B + k*lda + j);
                /* Perform individual block dgemm */
                int M_1 = M, N_1 = N, K_1 = K;
                if(M_1 < BLOCK_SIZE_2) {
                    M_1 = (M_1/4)*4 + 4;
                    M_1 = min(BLOCK_SIZE_2, M_1);
                }
                if(N_1 < BLOCK_SIZE_2) {
                    N_1 = (N_1/4)*4 + 4;
                    N_1 = min(BLOCK_SIZE_2, N_1);
                }
                if(K_1 < BLOCK_SIZE_2) {
                    K_1 = (K_1/4)*4 + 4;
                    K_1 = min(BLOCK_SIZE_2_K, K_1);
                }
                do_block1(BLOCK_SIZE_2, M_1, N_1, K_1, A_padded, B_padded, C_padded);
     
		__builtin_prefetch(A + (k+1)*BLOCK_SIZE_2_K);
		__builtin_prefetch(B + (k+1)*BLOCK_SIZE_2_K);
            }
            do_copy(lda, M, N, C + i*lda + j);
        }
	__builtin_prefetch(C + (i+1)*BLOCK_SIZE_2);
    }
    do_transpose(lda, A);
}
