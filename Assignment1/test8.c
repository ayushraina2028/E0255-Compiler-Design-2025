int not_anticipated_for_loop(int x, int *ptr) {
    int result;
    int i = 0;

    do {
        result = (x * x) % x; // This simplifies to 0 (modulo self)
        i++;
    } while (i < 10);

    return result;
}
