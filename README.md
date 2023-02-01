# matrix-benchmark
A runtime performance comparison of various versions of matrix-matrix multiplication implementations:

- Naive CPU version
- Optimized CPU version  with cache blocking and OpenMP's loop parallelization
- Naive CUDA kernel
- Tiled CUDA kernel utilizing shared memory

This benchmark was part of my [term paper](paper.pdf) "GPGU for High-Performance Neural Networks" in the summer semester 2020.


