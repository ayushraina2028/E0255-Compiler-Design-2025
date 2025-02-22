int add(int a, int b) {
    if (a > b) {
        return a + b;
    } else {
        return b + a;
    }
}

int main() {
    int x = 5, y = 10;
    int result = add(x, y);
    return result;
}