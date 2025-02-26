int foo() {
    int a = 10 + 5;  // Should be replaced with 15
    int b = a * 3;   // Should be replaced with 45
    return b;
}
