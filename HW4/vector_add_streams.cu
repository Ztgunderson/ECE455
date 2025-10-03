#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h> // Added for malloc/free

__global__ void vector_add(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        // if this is over N we want to stop as thread is useless/empty and if it runs this it will crash
        C[i] = A[i] + B[i];
}

// How to split into different streams for your algorithm
int main()
{
    int N = 10000000;
    size_t size = N * sizeof(float);

    // Host memory pointers
    float *h_A = (float *)malloc(size); // Renamed A to h_A to avoid conflict with kernel argument
    float *h_B = (float *)malloc(size); // Renamed B to h_B to avoid conflict with kernel argument
    float *h_C = (float *)malloc(size); // Renamed C to h_C to avoid conflict with kernel argument

    for (int i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device memory pointers
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int half = N / 2;
    size_t half_size = size / 2;
    int threads = 256;
    int blocks_half = (half + threads - 1) / threads;

    // Stream 1: First half data transfer H->D and kernel launch
    cudaMemcpyAsync(d_A, h_A, half_size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B, h_B, half_size, cudaMemcpyHostToDevice, stream1);
    vector_add<<<blocks_half, threads, 0, stream1>>>(d_A, d_B, d_C, half);
    cudaMemcpyAsync(h_C, d_C, half_size, cudaMemcpyDeviceToHost, stream1);

    // Stream 2: Second half data transfer H->D and kernel launch
    cudaMemcpyAsync(d_A + half, h_A + half, half_size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B + half, h_B + half, half_size, cudaMemcpyHostToDevice, stream2);
    vector_add<<<blocks_half, threads, 0, stream2>>>(d_A + half, d_B + half, d_C + half, half);
    cudaMemcpyAsync(h_C + half, d_C + half, half_size, cudaMemcpyDeviceToHost, stream2);

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    printf("C[0] = %f, C[N-1] = %f\n", h_C[0], h_C[N - 1]);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}