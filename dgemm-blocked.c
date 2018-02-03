/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <emmintrin.h>
#include <immintrin.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 16
#define BLOCK_SIZE_2 576
// #define BLOCK_SIZE 719
#define TRANSPOSE 1 
#endif

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
void do_block (int lda, int M, int N, int K, double* restrict A, double* restrict  B, double* restrict C)
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
			//printf("Hello\n");
			register __m128d c10_c11 = _mm_load_pd(C + lda);
			register __m128d c20_c21 = _mm_load_pd(C + 2*lda);
			register __m128d c30_c31 = _mm_load_pd(C + 3*lda);
                        register __m128d c02_c03 = _mm_load_pd(C + 2);
                        register __m128d c12_c13 = _mm_load_pd(C + lda + 2);
                        register __m128d c22_c23 = _mm_load_pd(C + 2*lda + 2);
                        register __m128d c32_c33 = _mm_load_pd(C + 3*lda + 2);

				//for (int i = 0; i < 4; i = i + 2)
				//	{
						register __m128d b  = _mm_load_pd(B);
						register __m128d b1 = _mm_load_pd(B + 2);
						register __m128d a1 = _mm_load1_pd(A);  
						register __m128d b2  = _mm_load_pd(B + 1*lda);
						register __m128d b3 = _mm_load_pd(B + 1*lda + 2);
						register __m128d a2 = _mm_load1_pd(A + 1*lda);  
						c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a1,b));
                                                c02_c03 = _mm_add_pd(c02_c03, _mm_mul_pd(a1,b1));
						c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a2,b2));
                                                c02_c03 = _mm_add_pd(c02_c03, _mm_mul_pd(a2,b3));
						b  = _mm_load_pd(B + 2*lda);
						b1 = _mm_load_pd(B + 2*lda + 2);
						a1 = _mm_load1_pd(A + 2*lda);  
						b2  = _mm_load_pd(B + 3*lda);
						b3 = _mm_load_pd(B + 3*lda + 2);
						a2 = _mm_load1_pd(A + 3*lda);  
						c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a1,b));
                                                c02_c03 = _mm_add_pd(c02_c03, _mm_mul_pd(a1,b1));
						c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a2,b2));
                                                c02_c03 = _mm_add_pd(c02_c03, _mm_mul_pd(a2,b3));
						a1 = _mm_load1_pd(A + 2*lda + 1);
						a2 = _mm_load1_pd(A + 3*lda + 1);
						c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a1,b));
                                                c12_c13 = _mm_add_pd(c12_c13, _mm_mul_pd(a1,b1));
						c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a2,b2));
                                                c12_c13 = _mm_add_pd(c12_c13, _mm_mul_pd(a2,b3));
						b  = _mm_load_pd(B);
						b1 = _mm_load_pd(B + 2);
						a1 = _mm_load1_pd(A + 1);
						b2  = _mm_load_pd(B + 1*lda);
						b3 = _mm_load_pd(B + 1*lda + 2);
						a2 = _mm_load1_pd(A + 1*lda + 1);
						c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a1,b));
                                                c12_c13 = _mm_add_pd(c12_c13, _mm_mul_pd(a1,b1));
						c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a2,b2));
                                                c12_c13 = _mm_add_pd(c12_c13, _mm_mul_pd(a2,b3));
						a1 = _mm_load1_pd(A + 2);
						a2 = _mm_load1_pd(A + 1*lda + 2);
						c20_c21 = _mm_add_pd(c20_c21, _mm_mul_pd(a1,b));
                                                c22_c23 = _mm_add_pd(c22_c23, _mm_mul_pd(a1,b1));
						c20_c21 = _mm_add_pd(c20_c21, _mm_mul_pd(a2,b2));
                                                c22_c23 = _mm_add_pd(c22_c23, _mm_mul_pd(a2,b3));
						b  = _mm_load_pd(B + 2*lda);
						b1 = _mm_load_pd(B + 2*lda + 2);
						a1 = _mm_load1_pd(A + 2*lda + 2);
						b2  = _mm_load_pd(B + 3*lda);
						b3 = _mm_load_pd(B + 3*lda + 2);
						a2 = _mm_load1_pd(A + 3*lda + 2);
						c20_c21 = _mm_add_pd(c20_c21, _mm_mul_pd(a1,b));
                                                c22_c23 = _mm_add_pd(c22_c23, _mm_mul_pd(a1,b1));
						c20_c21 = _mm_add_pd(c20_c21, _mm_mul_pd(a2,b2));
                                                c22_c23 = _mm_add_pd(c22_c23, _mm_mul_pd(a2,b3));
						a1 = _mm_load1_pd(A + 2*lda + 3);
						a2 = _mm_load1_pd(A + 3*lda + 3);
						c30_c31 = _mm_add_pd(c30_c31, _mm_mul_pd(a1,b));
                                                c32_c33 = _mm_add_pd(c32_c33, _mm_mul_pd(a1,b1));
						c30_c31 = _mm_add_pd(c30_c31, _mm_mul_pd(a2,b2));
                                                c32_c33 = _mm_add_pd(c32_c33, _mm_mul_pd(a2,b3));
						b  = _mm_load_pd(B);
						b1 = _mm_load_pd(B + 2);
						a1 = _mm_load1_pd(A + 3);
						b2  = _mm_load_pd(B + 1*lda);
						b3 = _mm_load_pd(B + 1*lda + 2);
						a2 = _mm_load1_pd(A + 1*lda + 3);
						c30_c31 = _mm_add_pd(c30_c31, _mm_mul_pd(a1,b));
                                                c32_c33 = _mm_add_pd(c32_c33, _mm_mul_pd(a1,b1));
						c30_c31 = _mm_add_pd(c30_c31, _mm_mul_pd(a2,b2));
                                                c32_c33 = _mm_add_pd(c32_c33, _mm_mul_pd(a2,b3));
						//register __m128d a2 = _mm_load1_pd(A + i*lda + 1);
						//register __m128d a3 = _mm_load1_pd(A + i*lda + 2);
						//register __m128d a4 = _mm_load1_pd(A + i*lda + 3);
						//register __m128d b  = _mm_load_pd(B + i*lda);
						//register __m128d a4 = _mm_load1_pd(A + i + 3*lda);
						//register __m128d b1 = _mm_load_pd(B + i*lda + 2);
						//c00_c01 = _mm_add_pd(c00_c01, _mm_mul_pd(a1,b));
						//c10_c11 = _mm_add_pd(c10_c11, _mm_mul_pd(a2,b));
						//c20_c21 = _mm_add_pd(c20_c21, _mm_mul_pd(a3,b));
						//c30_c31 = _mm_add_pd(c30_c31, _mm_mul_pd(a4,b));
                                                //c02_c03 = _mm_add_pd(c02_c03, _mm_mul_pd(a1,b1));
                                                //c12_c13 = _mm_add_pd(c12_c13, _mm_mul_pd(a2,b1));
                                                //c22_c23 = _mm_add_pd(c22_c23, _mm_mul_pd(a3,b1));
                                                //c32_c33 = _mm_add_pd(c32_c33, _mm_mul_pd(a4,b1));
				//}
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

void do_block1 (int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C)
{
    /* For each block-row of A */ 
    for (int i = 0; i < M; i += BLOCK_SIZE)
        /* For each block-column of B */
        for (int j = 0; j < N; j += BLOCK_SIZE)
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < K; k += 2*BLOCK_SIZE)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M_1 = min (BLOCK_SIZE, M-i);
                int N_1 = min (BLOCK_SIZE, N-j);
                int K_1 = min (2*BLOCK_SIZE, K-k);

                /* Perform individual block dgemm */
		//printf("M_1 = %d N_1 = %d K_1 = %d\n",M_1,N_1,K_1);
	if(K_1 == 2*BLOCK_SIZE) {
		if((M_1 % 4) == 0 && (N_1 % 4) == 0 && (K_1 % 4) == 0){
			do_block_vector(lda,M_1,N_1,K_1/2, A + k*lda + i, B + k*lda + j, C + i*lda + j);
			do_block_vector(lda,M_1,N_1,K_1/2, A + (k+BLOCK_SIZE)*lda + i, B + (k+BLOCK_SIZE)*lda + j, C + i*lda + j); }
		else {
                	do_block(lda, M_1, N_1, K_1, A + k*lda + i, B + k*lda + j, C + i*lda + j);
      	 	     }
	}
	else
                	do_block(lda, M_1, N_1, K_1, A + k*lda + i, B + k*lda + j, C + i*lda + j);
		
  }
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* restrict A, double* restrict B, double* restrict C)
{
int size;

if(lda % 2 == 0){
	size = lda;
}else{
	size = lda + 1;
}

double* A_padded = (double*)malloc(size * size * sizeof(double));
double* B_padded = (double*)malloc(size * size * sizeof(double));
double* C_padded = (double*)malloc(size * size * sizeof(double));


/*#ifdef TRANSPOSE
    for (int i = 0; i < size; ++i)
        for (int j = i+1; j < size; ++j) {
            double t = A[i*size+j];
            A[i*size+j] = A[j*size+i];
            A[j*size+i] = t;
        }
#endif
*/
for(int i = 0; i < size; i+=1) {
    A_padded[size*(size-1) + i] = 0;
    B_padded[size*(size-1) + i] = 0;
    A_padded[i*(size) + size - 1] = 0;
    B_padded[i*(size) + size - 1] = 0;
}
//if(lda % 2 != 0) {
	/* For each block-row of A */ 
    for (int i = 0; i < size; i += 1)
	{
        /* For each block-column of B */
        for (int j = 0; j < size;)
            /* Accumulate block dgemms into block of C */
            {
		
		if(j < size - 8) {
				
				/*memcpy(A_padded + i*size + j , A + i*lda + j,   8);
				memcpy(B_padded + i*size + j , B + i*lda + j++ ,8);
				memcpy(A_padded + i*size + j , A + i*lda + j ,  8);
				memcpy(B_padded + i*size + j , B + i*lda + j++ ,8);
				memcpy(A_padded + i*size + j , A + i*lda + j ,  8);
				memcpy(B_padded + i*size + j , B + i*lda + j++ ,8);
				memcpy(A_padded + i*size + j , A + i*lda + j ,  8);
				memcpy(B_padded + i*size + j , B + i*lda + j++ ,8);
				memcpy(A_padded + i*size + j , A + i*lda + j ,  8);
				memcpy(B_padded + i*size + j , B + i*lda + j++ ,8);
				memcpy(A_padded + i*size + j , A + i*lda + j ,  8);
				memcpy(B_padded + i*size + j , B + i*lda + j++ ,8);
				memcpy(A_padded + i*size + j , A + i*lda + j ,  8);
				memcpy(B_padded + i*size + j , B + i*lda + j++ ,8);
				memcpy(A_padded + i*size + j , A + i*lda + j ,  8);
				memcpy(B_padded + i*size + j , B + i*lda + j++ ,8);
				*/

				A_padded[i*size + j] = A[i*lda + j];
				B_padded[i*size + j] = B[i*lda + j++];
				A_padded[i*size + j] = A[i*lda + j];
				B_padded[i*size + j] = B[i*lda + j++];
				A_padded[i*size + j] = A[i*lda + j];
				B_padded[i*size + j] = B[i*lda + j++];
				A_padded[i*size + j] = A[i*lda + j];
				B_padded[i*size + j] = B[i*lda + j++];
				A_padded[i*size + j] = A[i*lda + j];
				B_padded[i*size + j] = B[i*lda + j++];
				A_padded[i*size + j] = A[i*lda + j];
				B_padded[i*size + j] = B[i*lda + j++];
				A_padded[i*size + j] = A[i*lda + j];
				B_padded[i*size + j] = B[i*lda + j++];
				A_padded[i*size + j] = A[i*lda + j];
				B_padded[i*size + j] = B[i*lda + j++];
		} else {
			while(j < size) {
				//memcpy(A_padded + i*size + j , A + i*lda + j ,8);
				//memcpy(B_padded + i*size + j , B + i*lda + j++ , 8);
				
				A_padded[i*size + j] = A[i*lda + j];
				B_padded[i*size + j] = B[i*lda + j++];
			}
		}
		
	}
}
//}

#ifdef TRANSPOSE
    for (int i = 0; i < size; ++i)
        for (int j = i+1; j < size; ++j) {
            double t = A_padded[i*size+j];
            A_padded[i*size+j] = A_padded[j*size+i];
            A_padded[j*size+i] = t;
        }
#endif

	/* For each block-row of A */ 
    for (int i = 0; i < size; i += BLOCK_SIZE_2)
        /* For each block-column of B */
        for (int j = 0; j < size; j += BLOCK_SIZE_2)
            /* Accumulate block dgemms into block of C */
            for (int k = 0; k < size; k += BLOCK_SIZE_2)
            {
                /* Correct block dimensions if block "goes off edge of" the matrix */
                int M = min (BLOCK_SIZE_2, size-i);
                int N = min (BLOCK_SIZE_2, size-j);
                int K = min (BLOCK_SIZE_2, size-k);

                /* Perform individual block dgemm */
                do_block1(size, M, N, K, A_padded + k*size + i, B_padded + k*size + j, C_padded + i*size + j);
            }

	for( int i = 0 ; i < lda ; i++) {
		for( int j = 0 ; j < lda ; j++) {
			C[i*lda + j] = C_padded[i*size + j];
		}
	}
}
