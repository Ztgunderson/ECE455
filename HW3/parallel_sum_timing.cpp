#include <iostream>
#include <vector>
#include <omp.h>

int main()
{
    const int N = 100000000;
    std::vector<double> data(N, 1.0);

    // Loop to test with 1, 2, 4, and 8 threads
    for (int threads = 1; threads <= 8; threads *= 2)
    {
        double sum = 0.0;

        // Start timing
        double t0 = omp_get_wtime();

// Parallel region: uses 'threads' number of threads,
// with reduction for the sum.
#pragma omp parallel for reduction(+ : sum) num_threads(threads)
        for (int i = 0; i < N; ++i)
        {
            sum += data[i];
        }

        // End timing
        double t1 = omp_get_wtime();

        // Output results
        std::cout << "Threads: " << threads
                  << ", Time: " << t1 - t0
                  << " sec, Sum: " << sum << std::endl;
    }

    // Expected Sum = N * 1.0 = 100,000,000
    return 0;
}