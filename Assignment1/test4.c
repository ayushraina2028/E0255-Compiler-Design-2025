int for_loop_invariant_expr(int x, int *ptr) {
    int i = 0;

    // Loop runs while i < 10
    while (i < 10) {
        int temp = (x * x) % x;  // Invariant expression inside loop
        i++;
    }

    return (x * x) % x;  // Final computation after loop exits
}
