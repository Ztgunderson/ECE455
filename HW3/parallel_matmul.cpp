#include <iostream>
#include <vector>
#include <omp.h>

int main()
{
    // N=1024 is a common size, good for cache alignment (power of 2)
    const int N = 1024;

    // Initialize matrices A (all 1s) and B (all 2s)
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 2));
    // Initialize result matrix C (all 0s)
    std::vector<std::vector<int>> C(N, std::vector<int>(N, 0));

// Parallelize the outermost loop (over rows 'i')
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        // Inner loops ('j' and 'k') run sequentially within each thread
        for (int j = 0; j < N; ++j)
        {
            int sum = 0;
            for (int k = 0; k < N; ++k)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    // Expected result: C[i][j] = sum(1 * 2) N times = 2 * N = 2 * 1024 = 2048
    std::cout << "C[0][0] = " << C[0][0] << std::endl;

    return 0;
}