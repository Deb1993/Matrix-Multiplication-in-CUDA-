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

#define L1_CACHE 32*1024
#define L2_CACHE 4*1024*1024
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 32
#define BLOCK_SIZE_2 416
#define BLOCK_SIZE_2_K 416
#endif

#define TRANSPOSE 1
#define AVX 1

#define min(a,b) (((a)<(b))?(a):(b))

static double A_padded[BLOCK_SIZE_2*BLOCK_SIZE_2] __attribute__ ((aligned (16)));
static double B_padded[BLOCK_SIZE_2*BLOCK_SIZE_2] __attribute__ ((aligned (16)));
static double C_padded[BLOCK_SIZE_2*BLOCK_SIZE_2] __attribute__ ((aligned (16)));

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
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
                cij += A[k*lda+i] * B[k*lda+j];
#else
            cij += A[i*lda+k] * B[k*lda+j];
#endif
            C[i*lda+j] = cij;
        }
}
 void do_vector_avx (int lda, double* restrict A, double* restrict B, double* restrict C)
 {
 
             register __m256d c00_c03 = _mm256_load_pd(C);
             //printf("Hello\n");
             register __m256d c10_c13 = _mm256_load_pd(C + lda);
             register __m256d c20_c23 = _mm256_load_pd(C + 2*lda);
             register __m256d c30_c33 = _mm256_load_pd(C + 3*lda);
 
             for( int i = 0 ; i < 4 ; i++) {
 
                 register __m256d a0 = _mm256_set1_pd(A[i*lda]);
                 //register __m256d a1 = _mm256_shuffle_pd(a0,a0,0);
                 register __m256d b1 = _mm256_load_pd(B + i*lda);
                 c00_c03 = _mm256_add_pd(c00_c03, _mm256_mul_pd(a0,b1));
                 a0 = _mm256_set1_pd(A[i*lda +1]);
                 //a1 = _mm256_shuffle_pd(a0,a0,0);
                 c10_c13 = _mm256_add_pd(c10_c13, _mm256_mul_pd(a0,b1));
                 a0 = _mm256_set1_pd(A[i*lda + 2]);
                 //a1 = _mm256_shuffle_pd(a0,a0,0);
                 c20_c23 = _mm256_add_pd(c20_c23, _mm256_mul_pd(a0,b1));
                 a0 = _mm256_set1_pd(A[i*lda + 3]);
                 //a1 = _mm256_shuffle_pd(a0,a0,0);
                 c30_c33 = _mm256_add_pd(c30_c33, _mm256_mul_pd(a0,b1));
             }
 
 _mm256_store_pd(C, c00_c03);
 _mm256_store_pd(C + 1*lda, c10_c13);
 _mm256_store_pd(C + 2*lda, c20_c23);
 _mm256_store_pd(C + 3*lda, c30_c33);
 }
 
 void do_block_vector_avx(int lda,int M,int N,int K, double* restrict A, double* restrict B, double* restrict C)
 {
     for (int i = 0; i < M; i = i + 4)
         for( int j = 0 ; j < N ; j = j + 4)
             {
                 for( int k = 0 ; k < K ; k = k + 4) {
                     do_vector_avx(lda, A + k*lda + i, B + k*lda + j, C + i*lda + j);
 
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
                    #ifdef SSE
                        do_block_vector(lda,M_1,N_1,K_1, A + k*lda + i, B + k*lda + j, C + i*lda + j);
                    #endif
                    
                    #ifdef AVX
                        do_block_vector_avx(lda,M_1,N_1,K_1, A + k*lda + i, B + k*lda + j, C + i*lda + j);
                    #endif
                 else
                     do_block(lda, M_1, N_1, K_1, A + k*lda + i, B + k*lda + j, C + i*lda + j);

             }
         }
     }
 }


void do_transpose(int lda, double *A) {
    for(int i = 0; i < lda; i+=1) {
        for(int j = i+1; j < lda; j+=1) {
            double  temp = A[i*lda + j];
            A[i*lda + j] = A[j*lda + i];
            A[j*lda + i] = temp;
        }
    }
}


void do_copy(int lda, int M, int N, double *C) {

    for(int i = 0; i < M; i+=1) {
        for(int j = 0; j < N; j+=1) {
            C[i*lda + j] = C_padded[i*BLOCK_SIZE_2 + j];
        }
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
     }
     src = B;
     dst = B_padded;
     for(int i = 0; i < K; i+=1) {
         memcpy(dst, src, N*sizeof(double));
         src+=lda;
         dst+=BLOCK_SIZE_2;
     }
}

void pad_C(int lda, int M, int N, double *C) {
    memset(C_padded, 0, BLOCK_SIZE_2*BLOCK_SIZE_2*sizeof(double));
    double *src = C, *dst = C_padded;
    for(int i = 0; i < M; i+=1) {
        memcpy(dst, src, N*sizeof(double));
        src+=lda;
        dst+=BLOCK_SIZE_2;
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
    int size = lda;

    /* For each block-row of A */ 
    for (int i = 0; i < size; i += BLOCK_SIZE_2) {
        /* For each block-column of B */
        for (int j = 0; j < size; j += BLOCK_SIZE_2) {
            /* Accumulate block dgemms into block of C */
            int M = min (BLOCK_SIZE_2, size-i);
            int N = min (BLOCK_SIZE_2, size-j);
            pad_C(lda, M, N, C + i*lda + j);
            for (int k = 0; k < size; k += BLOCK_SIZE_2_K)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int K = min (BLOCK_SIZE_2_K, size-k);
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
            }
            do_copy(lda, M, N, C + i*lda + j);
        }
    }
    do_transpose(lda, A);
}

