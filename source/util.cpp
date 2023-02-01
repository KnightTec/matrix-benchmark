#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void PrintMatrix(const float* matrix, int width, int height, const char* matrix_name)
{
	printf("%s =", matrix_name);
	for (int i = 0; i < height; i++)
	{
		printf("\t");
		for (int j = 0; j < width; j++)
		{
			printf("%.2f ", matrix[i * width + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void FillMatrixRandom(float* matrix, int width, int height)
{
	srand((unsigned)time(nullptr));
	for (int i = 0; i < width * height; i++)
	{
		matrix[i] = (int)rand() % 10;
	}
}

float MatrixElementDifference(const float* matrix_a, const float* matrix_b, size_t num_elements)
{
	float diff = 0;
	for (int i = 0; i < num_elements; i++)
	{
		diff += matrix_a[i] - matrix_b[i];
	}
	return diff;
}