#include <stdio.h>

int main() {
    int c;

    printf("Enter characters (Press Ctrl+D to stop on Linux/macOS or Ctrl+Z on Windows):\n");

    while ((c = getchar()) != EOF) {  // Read characters until EOF
        printf("You entered: %c (ASCII: %d)\n", c, c);
    }

    printf("End of input detected. Exiting...\n");

    return 0;
}
