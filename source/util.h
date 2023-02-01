#pragma once

void PrintMatrix(const float* matrix, int width, int height, const char* matrix_name);

void FillMatrixRandom(float* matrix, int width, int height);

float MatrixElementDifference(const float* matrix_a, const float* matrix_b, size_t num_elements);