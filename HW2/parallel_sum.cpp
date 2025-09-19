#include <iostream>
#include <vector>
#include <thread>
#include <numeric>
#include <random>
#include <chrono>

/*
Task:
Create a large array of 107 ints, split into T segments, and sum in parallel.
Combine partial sums and compare timing with a single-threaded baseline

Hint:
Use std::accumulate for segments and std::chrono for timing.
Store partials in a std::vector<long long>.
*/
void partial_sum(const std ::vector<int> &data,
                 size_t start, size_t end, long long &out)
{
    out = std ::accumulate(data.begin() + start, data.begin() + end, 0LL);
}
int main()
{
    const size_t N = 10'000'000;
    const int T = std ::thread ::hardware_concurrency() ? std ::thread ::hardware_concurrency() : 4;
    std ::vector<int> data(N);
    // C++ way of random number generation
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, 100);
    for (auto &x : data)
        x = dist(rng);

    // Baseline ( single - threaded )
    auto t0 = std ::chrono ::high_resolution_clock::now();
    long long baseline = std ::accumulate(data.begin(), data.end(), 0LL);
    auto t1 = std ::chrono ::high_resolution_clock ::now();
    // Parallel
    std ::vector<long long> partials(T, 0);
    std ::vector<std ::thread> threads;
    threads.reserve(T);
    size_t chunk = N / T;
    auto p0 = std ::chrono ::high_resolution_clock::now();
    for (int i = 0; i < T; ++i)
    {
        size_t s = i * chunk;
        size_t e = (i == T - 1) ? N : s + chunk;
        threads.emplace_back(partial_sum, std ::cref(data), s, e, std ::ref(partials[i]));
    }
    for (auto &th : threads)
        th.join();
    long long total = std ::accumulate(partials.begin(), partials.end(), 0LL);
    auto p1 = std ::chrono ::high_resolution_clock ::now();
    std ::chrono ::duration<double> t_base = t1 - t0;
    std ::chrono ::duration<double> t_par = p1 - p0;
    std ::cout << "Baseline sum : " << baseline
               << "Time : " << t_base.count() << " s\n ";
    std ::cout << "Parallel sum : " << total
               << "Time : " << t_par.count() << " s\n ";
    return 0;
}