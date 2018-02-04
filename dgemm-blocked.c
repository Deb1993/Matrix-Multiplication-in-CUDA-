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
#include <emmintrin.h>
#include <string.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#define L1_CACHE 32*1024
#define L2_CACHE 4*1024*1024
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 36
#define BLOCK_SIZE_2 416
#endif

#define TRANSPOSE 1

#define min(a,b) (((a)<(b))?(a):(b))

static double A_padded[BLOCK_SIZE_2*BLOCK_SIZE_2] __attribute__ ((aligned (16)));
static double B_padded[BLOCK_SIZE_2*BLOCK_SIZE_2] __attribute__ ((aligned (16)));
static double C_padded[BLOCK_SIZE_2*BLOCK_SIZE_2] __attribute__ ((aligned (16)));

static double A_p2[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (16)));
static double B_p2[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (16)));
static double C_p2[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (16)));

static void do_vectorize (int lda, int ii, int jj, int kk, double *restrict A, double *restrict B, double *restrict C) {

        __m128d c00_c01 = _mm_loadu_pd (C + ii*lda + jj);
        __m128d c10_c11 = _mm_loadu_pd(C + (ii + 1)*lda + jj);
        for(int i = 0; i < 2; i+=1)  {
            __m128d a1 = _mm_load1_pd(A + ii*lda + kk + i);
            __m128d a2 = _mm_load1_pd(A + (ii + 1)*lda + kk + i);

            __m128d b = _mm_loadu_pd(B + (kk + i)*lda + jj);

            c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a1, b));
            c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a2, b));
        }

        _mm_storeu_pd(C + ii*lda + jj, c00_c01);
        _mm_storeu_pd(C + (ii + 1)*lda + jj, c10_c11);
}


static void do_block_vectorized (int lda, int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{
    for (int i = 0; i < M; i+=2) {
        for (int j = 0; j < N; j+=2) {
            for (int k = 0; k < K; k+=2){
                do_vectorize(lda, i, j, k, A, B, C);
            }
        }
    }
}

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

void do_vector (int lda, double* restrict A, double* restrict B, double* restrict C)
{
    register __m128d c00_c01 = _mm_load_pd(C);
    register __m128d c10_c11 = _mm_load_pd(C + lda);
    register __m128d c20_c21 = _mm_load_pd(C + 2*lda);
    register __m128d c30_c31 = _mm_load_pd(C + 3*lda);
    register __m128d c02_c03 = _mm_load_pd(C + 2);
    register __m128d c12_c13 = _mm_load_pd(C + lda + 2);
    register __m128d c22_c23 = _mm_load_pd(C + 2*lda + 2);
    register __m128d c32_c33 = _mm_load_pd(C + 3*lda + 2);

    register __m128d temp1, temp2;

    for (int i = 0; i < 4; i++)
    {
        register __m128d b  = _mm_load_pd(B + i*lda);
        register __m128d b1 = _mm_load_pd(B + i*lda + 2);

        register __m128d a1 = _mm_load1_pd(A + i*lda);  
        temp1 = _mm_mul_pd(a1,b);
        register __m128d a2 = _mm_load1_pd(A + i*lda + 1);
        temp2 = _mm_mul_pd(a2,b);
        c00_c01 = _mm_add_pd(c00_c01, temp1);
        c10_c11 = _mm_add_pd(c10_c11, temp2);

        register __m128d a3 = _mm_load1_pd(A + i*lda + 2);
        temp1 = _mm_mul_pd(a3,b);
        register __m128d a4 = _mm_load1_pd(A + i*lda + 3);
        temp2 = _mm_mul_pd(a4,b);
        c20_c21 = _mm_add_pd(c20_c21, temp1);
        c30_c31 = _mm_add_pd(c30_c31, temp2);


        temp1 = _mm_mul_pd(a1,b1);
        temp2 = _mm_mul_pd(a2,b1);
        c02_c03 = _mm_add_pd(c02_c03, temp1);
        c12_c13 = _mm_add_pd(c12_c13, temp2);


        temp1 = _mm_mul_pd(a3,b1);
        temp2 = _mm_mul_pd(a4,b1);
        c22_c23 = _mm_add_pd(c22_c23, temp1);
        c32_c33 = _mm_add_pd(c32_c33, temp2);
    }
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
					 //int k2 = k + 4;
					 //__builtin_prefetch(A + k2*lda + i, 0, 0);
					 //__builtin_prefetch(B + k2*lda + j, 0, 0);

				}
}
}

void do_copy_p2(int lda, int M, int N, double *C) {

    for(int i = 0; i < M; i+=1) {
        for(int j = 0; j < N; j+=1) {
            C[i*lda + j] = C_p2[i*BLOCK_SIZE + j];
        }
    }
}

void pad_matrices_p2(int lda, int M, int N, int K, double *A, double *B) {

    for(int i = 0; i < BLOCK_SIZE; i+=1) {
        for(int j = 0; j < BLOCK_SIZE; j+=1) {
            if(i >= K || j >= M) {
                A_p2[i*BLOCK_SIZE + j] = 0;
            } else {
                A_p2[i*BLOCK_SIZE + j] = A[i*lda + j];
            }

            if(i >= K || j >= N) {
                B_p2[i*BLOCK_SIZE + j] = 0;
            } else {
	        B_p2[i*BLOCK_SIZE + j] = B[i*lda + j];
            }

        }
    }
}

void pad_C_p2(int lda, int M, int N, double *C) {
    for(int i = 0; i < BLOCK_SIZE; i+=1) {
        for(int j = 0; j < BLOCK_SIZE; j+=1) {
                if(i >= M || j >= N) 
                    C_p2[i*BLOCK_SIZE + j] = 0;
                else 
	            C_p2[i*BLOCK_SIZE + j] = C[i*lda + j];
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
     }
 }


/*
void do_block1 (int lda, int M, int N, int K, double *restrict A, double *restrict B, double *restrict C)
{

    for (int i = 0; i < M; i += BLOCK_SIZE) {
        int M_1 = min (BLOCK_SIZE, M-i);
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            int N_1 = min (BLOCK_SIZE, N-j);
            pad_C_p2(lda, M_1, N_1, C + i*lda + j);
            for (int k = 0; k < K; k += BLOCK_SIZE)
            {
                int K_1 = min (BLOCK_SIZE, K-k);
                pad_matrices_p2(lda, M_1, N_1, K_1, A + k*lda + i, B + k*lda + j);
                do_block_vector(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE, A_p2, B_p2, C_p2);
                 int k2 = k + BLOCK_SIZE;
                __builtin_prefetch(A + k2*BLOCK_SIZE + i, 0, 0);
                __builtin_prefetch(B + k2*BLOCK_SIZE + j, 0, 0);
            }
            do_copy_p2(lda, M_1, N_1, C + i*lda + j);
        }
    }
}

*/
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

    memset(A_padded, 0, BLOCK_SIZE_2*BLOCK_SIZE_2);
    memset(B_padded, 0, BLOCK_SIZE_2*BLOCK_SIZE_2);
 
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
    memset(C_padded, 0, BLOCK_SIZE_2*BLOCK_SIZE_2);
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
/*
    int size = lda % 16 == 0 ? lda : 16*(lda/16 + 1);
    double *A_padded, *B_padded, *C_padded;

    posix_memalign((void **)&A_padded, 16, size*size*sizeof(double));
    posix_memalign((void **)&B_padded, 16, size*size*sizeof(double));
    posix_memalign((void **)&C_padded, 16, size*size*sizeof(double));

    for(int i = 0; i < size; i+=1) {
        for(int j = 0; j < size; j+=1) {
            C_padded[i*size + j] = 0;
            if(i == lda || j == lda) {
                A_padded[i*size + j] = 0;
                B_padded[i*size + j] = 0;
            } else {
                A_padded[i*size + j] = A[i*lda + j];
                B_padded[i*size + j] = B[i*lda + j];
            }
        }
    }

*/
    /* For each block-row of A */ 
    for (int i = 0; i < size; i += BLOCK_SIZE_2) {
        /* For each block-column of B */
        for (int j = 0; j < size; j += BLOCK_SIZE_2) {
            /* Accumulate block dgemms into block of C */
            int M = min (BLOCK_SIZE_2, size-i);
            int N = min (BLOCK_SIZE_2, size-j);
            pad_C(lda, M, N, C + i*lda + j);
            for (int k = 0; k < size; k += BLOCK_SIZE_2)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int K = min (BLOCK_SIZE_2, size-k);
                pad_matrices(lda, M, N, K, A + k*lda + i, B + k*lda + j);
                //printf("NORMAL\n");
                //printMatrix(A, M, K);
                //printf("PADDED\n");
                //printMatrix(A_padded, M, K);
                /* Perform individual block dgemm */
                do_block1(BLOCK_SIZE_2, M, N, K, A_padded, B_padded, C_padded);
            }
            do_copy(lda, M, N, C + i*lda + j);
        }
    }
/*
    for(int i = 0; i < lda; i+=1) {
        for(int j = 0; j < lda; j+=1) {
            C[i*lda + j] = C_padded[i*size + j];
        }
    }
    free(A_padded);
    free(B_padded);
    free(C_padded);
*/   
    do_transpose(lda, A);

}
