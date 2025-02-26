#include <stdio.h>

void loop_example(int n) {
    for (int i = 0; i < n; i++) {
        printf("Iteration %d\n", i);
    }
}

int main() {
    loop_example(5);
    return 0;
}
