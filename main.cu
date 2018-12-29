#include <stdlib.h>
#include <stdio.h>

__global__ void hello()
{
    printf("hello world from the gpu\n");
}

void launchhello()
{
    hello<<<1,1>>>();
    cudaDeviceSynchronize();
}
