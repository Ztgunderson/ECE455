#include <iostream>
#include <thread>
#include <vector>

/*
---Task---
Spawn N threads.
Each thread prints "Hello from thread X of N" where X is the thread’s ID (0-based).
Join all threads.

---Hints---
Pass the thread ID as a function argument;
store threads in a std::vector<std::thread> and call join() on each.
*/

void hello(int id, int total)
{
    std ::cout << " Hello from thread " << id << " of " << total << " \n ";
}
int main()
{
    const int N = 5;
    std ::vector<std ::thread> threads;
    threads.reserve(N); // Create 5 spaces for the compiler to have memory preallocated in the code

    // Make a vector of all the threads
    for (int i = 0; i < N; ++i)
        threads.emplace_back(hello, i, N); // This constructs the threads and adds it to the vector

    // Join all the threads
    for (auto &t : threads)
        t.join();

    return 0;
}