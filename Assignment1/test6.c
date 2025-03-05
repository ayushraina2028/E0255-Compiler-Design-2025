#include <math.h>

int if_else_math_call(int x, int *ptr) {
    double result;

    if (x > 3) {
        result = (x * x) + exp(x) + 5.0;
    } else if (x > 2) {
        result = (x * x) + exp(x) + 3.0;
    } else {
        result = (x * x) + exp(x) + 1.0;
    }

    return (int)result;
}
