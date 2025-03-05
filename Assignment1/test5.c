int switch_case(int x, int *ptr) {
    int result;

    switch (x) {
        case 0:
        case 1:
            result = (x * x) + x + 3;
            break;
        case 2:
        case 3:
            result = (x * x) + x + 3;
            break;
        default:
            result = (x * x) + x + 3;
            break;
    }

    return result;
}
