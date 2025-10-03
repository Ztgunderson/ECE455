#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h> // Added for malloc/free

__global__ void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}

int main()
{
    int N = 10000000;
    float a = 2.0f; // Explicitly set a=2.0f as per problem description
    size_t size = N * sizeof(float);

    // Host memory pointers
    float *h_x = (float *)malloc(size); // Renamed x to h_x to avoid conflict with kernel argument
    float *h_y = (float *)malloc(size); // Renamed y to h_y to avoid conflict with kernel argument

    for (int i = 0; i < N; i++)
    {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // Device memory pointers
    float *d_x, *d_y;

    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    saxpy<<<blocksPerGrid, threadsPerBlock>>>(N, a, d_x, d_y);

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    printf("y[0] = %f\n", h_y[0]);

    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}