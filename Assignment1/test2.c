int simple_if_else_multiple(int x, int *ptr) {
    int result;
    
    if (x > 3) {
        result = x * x + x + 5;
    } else if (x > 2) {
        result = x * x + x + 5;
    } else {
        result = x * x + x + 5;
    }

    return result;
}
