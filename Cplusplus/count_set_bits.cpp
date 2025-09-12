#include <iostream>
#include <thread>
#include <chrono>
#include <functional>
#include <limits>
#include <atomic>

using SetBitCounter = int(*)(int);


int countSetBits_Shift(int v) {
    int c = 0;
    for (; v; v >>= 1)
        c += (v & 1);
    return c;
}

int countSetBits_BK(int v) {
    int c = 0;
    for (; v; v &= (v - 1))
        ++c;
    return c;
}

int countSetBits_MultithreadingTree(int v){
    // Stupidly Parallizable but over head is too large to make sense to do this effectively

    // Define threads
    const int NUM_THREADS = 4; // Divide into 4 parts (8 bits each for 32-bit int)
    const int BITS_PER_THREAD = 32 / NUM_THREADS;

    std::atomic<int> count{0};
    std::vector<std::thread> threads; // Create a vector of threads

    for (int i = 0; i < NUM_THREADS; ++i){
        threads.emplace_back( // Stores thread pointer in place in the vector
            [=, &count]() { // Lambda function that will run in thread
                int local = 0;
                int startBit = i * BITS_PER_THREAD;
                int mask = ((i << BITS_PER_THREAD) - 1) << startBit;
                int segment = (v & mask) >> startBit;

                // Count bits in this segment
                while (segment){
                    local += (segment & 1);
                    segment >>= 1;
                }

                count += local;
            }
        );
    }

    for (auto& t : threads) { 
        t.join();
    }

    return count.load();
}

void benchmark(SetBitCounter func, const std::string& name, int iterations = 1'000'000, int value = std::numeric_limits<int>::max()){
    int total = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i){
        total += func(value);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;

    std::cout << name << ":\n";
    std::cout << "  Total time: " << duration.count() << " microseconds\n";
    std::cout << "  Avg per call: " << duration.count() / iterations << " microseconds\n";
    std::cout << "  Dummy total (ignore): " << total << "\n\n";

}

int main() {
    benchmark(countSetBits_Shift, "Shift-Based Loop");
    benchmark(countSetBits_BK, "Brian Kernighan");
    benchmark(countSetBits_MultithreadingTree, "Bad multithreading example");
    return 0;
}