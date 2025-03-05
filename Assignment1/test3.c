int if_else_memory(int x, int *ptr) {
    int result = 0;  // Corresponds to store i32 0, ptr %3
    int temp = x;    // Store x in memory first

    if (temp > 2) {
        temp = temp * temp + temp + 5;
    } else {
        temp = temp * temp + temp + 3;
    }

    return temp;
}
