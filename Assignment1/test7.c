int if_else_multiple_redundant_exprs(int x, int *ptr) {
    int result1, result2;

    if (x > 2) {
        result1 = x * x;
        result2 = result1 + x;
        result2 = result2 + 5;
        result2 = result2 + 5; // Redundant addition
    } else {
        result1 = x * x;
        result2 = result1 + x;
        result2 = result2 + 3;
        result1 = x * x; // Redundant multiplication
    }

    return result1 + result2;
}
