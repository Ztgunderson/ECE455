#include <iostream>
#include <thread>

// function for the first thread
void printNumbers() {
    for (int i = 1; i <= 5; ++i) {
        printf("Number: %d\n", i);
    }
}

// function for the second thread
void printLetters() {
    for (char letter = 'A'; letter <= 'E'; ++letter) {
        printf("Letter: %c\n", letter);
    }
}

int main() {
    // create two threads, each running a function
    std::thread t1(printNumbers);
    std::thread t2(printLetters);
    // join each thread
    t1.join();
    t2.join();
    std::cout << "Both threads finished!" << std::endl;
    return 0;
}