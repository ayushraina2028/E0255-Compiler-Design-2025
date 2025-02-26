int test3(int x, int y, int z) {
    int a, b, c;

    a = x + y;   // Anticipated (x + y), should be in V_USE

    if (z > 0) {
        b = a * z;  // Anticipated (a * z), should be in V_USE
    } else {
        c = x + y;  // Same as (x + y), should now be in V_DEF
    }

    return a + b + c;
}
