int not_anticipated_switch(int x, int *ptr) {
    int result;
    
    switch (x) {
        case 0:
        case 1:
        case 2:
            result = ((x % 2) * (x % 2)) + (x % 2) + 3;
            break;
        default:
            result = (x * x) + x + 2;
            break;
    }
    
    return result;
}
