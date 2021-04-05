// Original source reproduced unmodified here from: 
// https://github.com/olcf/vector_addition_tutorials/blob/master/CUDA/vecAdd.cu

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA kernel. Each thread takes care of one element of c
__global__ void vecAdd(double *devA, double *devB, double *devC, int n) {
  // Get our global thread ID
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    devC[i] = devA[i] + devB[i];
}

int main(int argc, char *argv[]) {
  // Size of vectors
  constexpr const size_t N = 100000;
  // Size, in bytes, of each vector
  constexpr const size_t bytes = N * sizeof(double);


  // Device vectors
  double *devA, *devB, *devC;

  
  // Allocate memory for each vector on GPU
  cudaMallocManaged(&devA, bytes);
  cudaMallocManaged(&devB, bytes);
  cudaMallocManaged(&devC, bytes);

  // Initialize vectors on host
  for (int i = 0; i < N; i++) {
    devA[i] = sin(i) * sin(i);
    devB[i] = cos(i) * cos(i);
  }

  // Number of threads in each thread block
  int blockSize = 1024;
  // Number of thread blocks in grid
  int gridSize = (int)ceil((float)N / blockSize);
  // Execute the kernel
  vecAdd<<<gridSize, blockSize>>>(devA, devB, devC, N);

  cudaDeviceSynchronize();

  // Sum up vector c and print result divided by n, this should equal 1 within
  // error
  double sum = 0;
  for (int i = 0; i < N; i++)
    sum += devC[i];
  printf("final result: %f\n", sum / N);

  // Release device memory
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);

  return 0;
}
