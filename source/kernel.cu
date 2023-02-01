#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "DirectXMath.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


void MatrixMatrixMultiplyCpu(const float* a, const float* b, float *c, int width_a, int height_a, int width_b)
{
	for (int i = 0; i < height_a; i++)
	{
		for (int j = 0; j < width_b; j++)
		{
			for (int k = 0; k < width_a; k++)
			{
				c[i * width_b + j] += a[i * width_a + k] * b[k * width_b + j];
			}
		}
	}
}

void MatrixMatrixMultiplyCpuOptimized(const float* a, const float* b, float *c, int width_a, int height_a, int width_b)
{
	// optimized through cache-blocking
# define CPU_BLOCK_SIZE 16
	#pragma omp parallel for
	for (int i = 0; i < height_a; i += CPU_BLOCK_SIZE)
	{
		for (int j = 0; j < width_b; j += CPU_BLOCK_SIZE)
		{
			for (int k = 0; k < width_a; k += CPU_BLOCK_SIZE)
			{
				for (int ii = i; ii < i + CPU_BLOCK_SIZE; ii++)
				{
					for (int jj = j; jj < j + CPU_BLOCK_SIZE; jj++)
					{
						for (int kk = k; kk < k + CPU_BLOCK_SIZE; kk++)
						{
							c[ii * width_b + jj] += a[ii * width_a + kk] * b[kk * width_b + jj];
						}
					}
				}
			}
		}
	}
}

__global__ void MatrixMatrixMultiplySimple(const float* a, const float* b, float *c, int width_a, int width_b)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	c[i * width_b + j] = 0;
	for (int k = 0; k < width_a; k++)
	{
		c[i * width_b + j] += a[i * width_a + k] * b[k * width_b + j];
	}
}

#define N 1024
#define BLOCK_SIZE 16

__global__ void MatrixMatrixMultiplyTiled(const float* a, const float* b, float *c, int width_a, int width_b)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	c[y * width_b + x] = 0;

	// loop over all tiles required to compute this thread group result tile in C
	for (unsigned int i = 0; i < width_a; i += blockDim.x)
	{
		__shared__ float matrix_a_shared[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float matrix_b_shared[BLOCK_SIZE][BLOCK_SIZE];

		// load tiles into shared memory
		matrix_a_shared[threadIdx.x][threadIdx.y] = a[y * width_a + threadIdx.x + i];
		matrix_b_shared[threadIdx.x][threadIdx.y] = b[(threadIdx.y + i) * width_b + x];

		// wait for all threads to finish loading into shared memory
		__syncthreads();

		for (unsigned int k = 0; k < BLOCK_SIZE; k++)
		{
			c[y * width_b + x] += matrix_a_shared[k][threadIdx.x] * matrix_b_shared[threadIdx.y][k];
		}
		// wait so shared memory does not get overwritten before all threads in group are finished
		__syncthreads();
	}
}



int main()
{
	int width_a = N;
	int height_a = N;
	size_t mem_size_a = width_a * height_a * sizeof(float);
	
	int width_b = N;
	int height_b = width_a;
	size_t mem_size_b = width_b * height_b * sizeof(float);
	
	int width_c = width_b;
	int height_c = height_a;
	size_t mem_size_c = width_c * height_c * sizeof(float);

	// Allocate host memory
	auto* matrix_a = static_cast<float*>(malloc(mem_size_a));
	auto* matrix_b = static_cast<float*>(malloc(mem_size_b));
	auto* matrix_c = static_cast<float*>(malloc(mem_size_c));
	auto* matrix_c_2 = static_cast<float*>(malloc(mem_size_c));
	memset(matrix_c, 0, mem_size_c);
	memset(matrix_c_2, 0, mem_size_c);

	// Allocate device memory
	float *d_matrix_a, *d_matrix_b, *d_matrix_c;
	cudaMalloc(&d_matrix_a, mem_size_a);
	cudaMalloc(&d_matrix_b, mem_size_b);
	cudaMalloc(&d_matrix_c, mem_size_c);
	
	// matrices stored in row-major order
	FillMatrixRandom(matrix_a, width_a, height_a);
	FillMatrixRandom(matrix_b, width_b, height_b);
	//PrintMatrix(matrix_a, width_a, height_a, "A");
	//PrintMatrix(matrix_b, width_b, height_b, "B");


	// CPU version
	clock_t begin, stop;
	begin = clock();
	MatrixMatrixMultiplyCpu(matrix_a, matrix_b, matrix_c, width_a, height_a, width_b);
	stop = clock();
	//PrintMatrix(matrix_c, width_b, height_a, "C");
	printf("CPU time in milliseconds: %f\n", 1000 * (stop - begin) / static_cast<double>(CLOCKS_PER_SEC));
	// Copy result for later verification of GPU computations
	auto* matrix_c_cpu_result = static_cast<float*>(malloc(mem_size_c));
	memcpy(matrix_c_cpu_result, matrix_c, mem_size_c);

	// CPU version optimized
	begin = clock();
	MatrixMatrixMultiplyCpuOptimized(matrix_a, matrix_b, matrix_c_2, width_a, height_a, width_b);
	stop = clock();
	//PrintMatrix(matrix_c_2, width_b, height_a, "C");
	printf("Optimized CPU time in milliseconds: %f\n", 1000 * (stop - begin) / static_cast<double>(CLOCKS_PER_SEC));
	// Verify correctness
	float error = MatrixElementDifference(matrix_c_2, matrix_c_cpu_result, width_c * height_c);
	printf("Computation error: %f\n", error);


	// simple GPU version
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 numBlocks((width_c + (BLOCK_SIZE - 1)) / BLOCK_SIZE, (height_c + (BLOCK_SIZE - 1)) / BLOCK_SIZE);
	cudaEvent_t start, end;
	
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaMemcpy(d_matrix_a, matrix_a, mem_size_a, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_b, matrix_b, mem_size_b, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	MatrixMatrixMultiplySimple<<<numBlocks, threadsPerBlock>>>(d_matrix_a, d_matrix_b, d_matrix_c, width_a, width_b);
	cudaMemcpy(matrix_c, d_matrix_c, mem_size_c, cudaMemcpyDeviceToHost);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, end);
	//PrintMatrix(matrix_c, width_c, height_c, "C");
	printf("Simple GPU time in milliseconds: %f\n", milliseconds);
	// Verify correctness
	error = MatrixElementDifference(matrix_c, matrix_c_cpu_result, width_c * height_c);
	printf("Computation error: %f\n", error);


	// tiled GPU version
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaMemcpy(d_matrix_a, matrix_a, mem_size_a, cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrix_b, matrix_b, mem_size_b, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	MatrixMatrixMultiplyTiled<<<numBlocks, threadsPerBlock>>>(d_matrix_a, d_matrix_b, d_matrix_c, width_a, width_b);
	cudaMemcpy(matrix_c, d_matrix_c, mem_size_c, cudaMemcpyDeviceToHost);
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, end);
	//PrintMatrix(matrix_c, width_c, height_c, "C");
	printf("Tiled GPU time in milliseconds: %f\n", milliseconds);
	// Verify correctness
	error = MatrixElementDifference(matrix_c, matrix_c_cpu_result, width_c * height_c);
	printf("Computation error: %f\n", error);

	free(matrix_a);
	free(matrix_b);
	free(matrix_c);

    return 0;
}