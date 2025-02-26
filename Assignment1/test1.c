int test1(int a, int b, int e) {
    int c, d, f;
    
    c = a + b;  // Anticipated: (a + b), should be in V_USE
    d = c + e;  // Anticipated: (c + e), should be in V_USE
    f = a + b;  // Reuse of (a + b), should be in V_DEF

    return d + f;
}
