#include <stdio.h>
#include <stdlib.h>

void time1Dlinear();
void time1Dtexture();
void time2Dlinear();
void time2Dtexture();
void time3Dlinear();
void time3Dtexture();

int main(int argc, char *argv[]) {
    time1Dlinear();
    time1Dtexture();
    time2Dlinear();
    time2Dtexture();
    time3Dlinear();
    time3Dtexture();
    return 0;
}
