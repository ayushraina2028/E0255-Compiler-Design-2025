// test_comprehensive.c
int test_function() {
    // Memory operations
    int arr[10];
    int *p = arr;
    *p = 5;
    int val = *p;

    // Redundant expressions
    int a = 10, b = 20;
    int sum1 = a + b;  // First computation
    int sum2 = a + b;  // Redundant computation

    // If/else with anticipated expressions
    if (a > b) {
        sum1 = a + b;  // Same expression in both branches
        val = sum1 * 2;
    } else {
        sum1 = a + b;  // Same expression in both branches
        val = sum1 * 3;
    }

    // Switch statement
    switch(val) {
        case 1:
            sum1 = a + b;
            break;
        case 2:
            sum2 = a + b;
            break;
        default:
            val = 0;
    }

    // Loop with invariant
    int invariant = 42;
    for(int i = 0; i < 10; i++) {
        arr[i] = invariant * 2;  // Loop invariant expression
        sum1 += i;               // Loop varying expression
    }

    return sum1 + sum2;
}