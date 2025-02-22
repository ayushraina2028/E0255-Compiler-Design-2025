#include <stdio.h>

int main(int argc, char* argv[]) {
    int i, j, z, N;
    z = 0;

    scanf("%d",&N);

    for(i = 0;i < N; ++i) {
        for(j = 0;j < N; ++j) {
            z++;
        }
    }

    return z;
}