#include <stdio.h>

int main(int argc, char* argv[]) {
    int i, j, z, N;
    z = 0;

    int A[100];

    for(int i = 0;i < 100; i++) {
        A[i] = 0;
    }

    for(int i = 1;i < 100; i++) {
        A[i] = A[i-1] + 1;
    }

    return z;
}