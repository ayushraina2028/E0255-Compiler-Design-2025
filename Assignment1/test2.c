int test_function(int a, int b, int e) {
 int c, d, f;
c = a + b; // First anticipated expression (used later)*
d = c + e; // Second anticipated expression (used later)*
f = a + b;// Third anticipated expression (same as first)*
return d + f; // Use of previously computed values*
}