int arithmetic_ops(int a, int b) {
    int sum = a + b;
    int diff = a - b;
    int prod = a * b;
    int quot = (b != 0) ? a / b : 0; // Avoid division by zero
    return sum + diff + prod + quot;
}
