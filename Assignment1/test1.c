void example(int x, int y) {
    int a, b, c;
    
    a = x + y;  // (x + y) is computed here
    if (x > 0) {
        b = x + y;  // (x + y) is redundantly computed again
    } else {
        c = x + y;  // (x + y) is also computed here
    }
}
