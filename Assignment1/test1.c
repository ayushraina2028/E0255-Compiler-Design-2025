#include <stdio.h>

int main() {
    int a = 5, b = 10;
    for (int i = 0; i < 3; i++) {
        int x = a * b;  // Recomputed in every loop iteration
        printf("%d\n", x);
    }
    return 0;
}
