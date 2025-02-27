// test2.c - More complex patterns
int test_complex() {
    // Nested if/else
    int a = 10, b = 20, c = 30;
    if (a > b) {
        if (b > c) {
            a = b + c;
        } else {
            b = a + c;
        }
    } else {
        c = a + b;
    }

    // Nested loops
    int arr[10][10];
    for(int i = 0; i < 10; i++) {
        int outer_invariant = a * b;  // Loop invariant for outer loop
        for(int j = 0; j < 10; j++) {
            arr[i][j] = outer_invariant + j;
        }
    }

    // Multiple redundant expressions
    int sum1 = a + b;
    int sum2 = b + c;
    int sum3 = a + b;  // Redundant with sum1
    int sum4 = b + c;  // Redundant with sum2

    return sum1 + sum2 + sum3 + sum4;
}