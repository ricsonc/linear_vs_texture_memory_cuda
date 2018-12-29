#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <cuda.h>

texture<char, 1> texref;
texture<char, 2> texref2;
texture<char, 3> texref3;

#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* 1D linear memory */

__global__ void touch1Dlinear(void* devPtr_, void* outPtr_, long M)
{
    long N = gridDim.x * blockDim.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    char* devPtr = (char*) devPtr_;
    char* outPtr = (char*) outPtr_;

    for(; i < M-2; i += N) {
        outPtr[i] = devPtr[i] + devPtr[i+1] + devPtr[i+2] + devPtr[i+3];
    }
}

void time1Dlinear()
{
    void* devPtr;
    void* outPtr;
    long M = 1000L*1000L*100L;
    int blocks = 65536;
    int threads = 64;

    cudaMalloc(&devPtr, M);
    cudaMalloc(&outPtr, M);

    auto start = std::chrono::system_clock::now();

    for(int i = 0; i < 10; i ++){
        touch1Dlinear<<<blocks,threads>>>(devPtr, outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );
    
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaFree(devPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("linear 1D: %f ms\n", 1000*delta.count());
}

/* 1D texture memory */

__global__ void touch1Dtexture(void* outPtr_, long M){
    long N = gridDim.x * blockDim.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    char* outPtr = (char*) outPtr_;

    for(; i < M-2; i += N) {
        outPtr[i] = (
            tex1Dfetch(texref, i) +
            tex1Dfetch(texref, i+1) +
            tex1Dfetch(texref, i+2) + 
            tex1Dfetch(texref, i+3)
        );
    }
}

void time1Dtexture()
{
    void* refPtr;
    void* outPtr;
    long M = 1000L*1000L*100L;
    int blocks = 65536;
    int threads = 64;

    cudaMalloc(&refPtr, M);
    cudaMalloc(&outPtr, M);
    gpuCheck( cudaBindTexture(NULL, texref, refPtr, M) );
    
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++){
        touch1Dtexture<<<blocks, threads>>>(outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaUnbindTexture(texref) );    
    gpuCheck( cudaFree(refPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("texture 1D: %f ms\n", 1000*delta.count());
}

/* 2D linear memory */

__global__ void touch2Dlinear(void* devPtr_, void* outPtr_, long M)
{
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;

    char* devPtr = (char*) devPtr_;
    char* outPtr = (char*) outPtr_;

    for(; ix < M-1; ix += N) {
        for(; iy < M-1; iy += N) {
            outPtr[ix*M+iy] = (
                devPtr[ix*M+iy] +
                devPtr[(ix+1)*M+iy] +
                devPtr[ix*M+iy+1] + 
                devPtr[(ix+1)*M+iy+1]
            );
        }
    }
}

void time2Dlinear()
{
    void* devPtr;
    void* outPtr;
    long M = 10000;
    dim3 blocks(256,256);
    dim3 threads(8,8);

    cudaMalloc(&devPtr, M*M);
    cudaMalloc(&outPtr, M*M);

    auto start = std::chrono::system_clock::now();

    for(int i = 0; i < 10; i ++){
        touch2Dlinear<<<blocks,threads>>>(devPtr, outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );
    
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaFree(devPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("linear 2D: %f ms\n", 1000*delta.count());
}

/* 2D texture memory */

__global__ void touch2Dtexture(void* outPtr_, long M){
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;

    char* outPtr = (char*) outPtr_;

    for(; ix < M-1; ix += N) {
        for(; iy < M-1; iy += N) {
            outPtr[ix*M+iy] = (
                tex2D(texref2, ix, iy) +
                tex2D(texref2, ix+1, iy) +
                tex2D(texref2, ix, iy+1) +
                tex2D(texref2, ix+1, iy+1)
            );
        }
    }
}

void time2Dtexture()
{
    void* refPtr;
    void* outPtr;
    long M = 10000;
    dim3 blocks(256, 256);
    dim3 threads(8,8);

    cudaMalloc(&refPtr, M*M);
    cudaMalloc(&outPtr, M*M);
    gpuCheck( cudaBindTexture(NULL, texref2, refPtr, M*M) );
    
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++){
        touch2Dtexture<<<blocks, threads>>>(outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaUnbindTexture(texref) );    
    gpuCheck( cudaFree(refPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("texture 2D: %f ms\n", 1000*delta.count());
}

/* 3D linear memory */

__global__ void touch3Dlinear(void* devPtr_, void* outPtr_, long M)
{
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;
    long iz = blockIdx.z * blockDim.z + threadIdx.z;

    char* devPtr = (char*) devPtr_;
    char* outPtr = (char*) outPtr_;

    for(; ix < M-1; ix += N) {
        for(; iy < M-1; iy += N) {
            for(; iz < M-1; iz += N) {
                outPtr[ix*M*M+iy*M+iz] = (
                    devPtr[ix*M*M+iy*M+iz] +
                    devPtr[ix*M*M+iy*M+(iz+1)] +
                    devPtr[ix*M*M+(iy+1)*M+iz] +
                    devPtr[(ix+1)*M*M+iy*M+iz]
                );
            }
        }
    }
}

void time3Dlinear()
{
    void* devPtr;
    void* outPtr;
    long M = 465;
    dim3 blocks(32,32,32);
    dim3 threads(4, 4, 4);

    cudaMalloc(&devPtr, M*M*M);
    cudaMalloc(&outPtr, M*M*M);

    auto start = std::chrono::system_clock::now();

    for(int i = 0; i < 10; i ++){
        touch3Dlinear<<<blocks,threads>>>(devPtr, outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );
    
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaFree(devPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("linear 2D: %f ms\n", 1000*delta.count());
}

/* 3D texture memory */

__global__ void touch3Dtexture(void* outPtr_, long M){
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;
    long iz = blockIdx.z * blockDim.z + threadIdx.z;

    char* outPtr = (char*) outPtr_;

    for(; ix < M-1; ix += N) {
        for(; iy < M-1; iy += N) {
            for(; iz < M-1; iz += N) {
                outPtr[ix*M*M+iy*M+iz] = (
                    tex3D(texref3, ix, iy, iz) +                    
                    tex3D(texref3, ix, iy, iz+1) +
                    tex3D(texref3, ix, iy+1, iz) +
                    tex3D(texref3, ix+1, iy, iz)
                );
            }
        }
    }
}

void time3Dtexture()
{
    void* refPtr;
    void* outPtr;
    long M = 465;
    dim3 blocks(32,32,32);
    dim3 threads(4,4,4);

    cudaMalloc(&refPtr, M*M*M);
    cudaMalloc(&outPtr, M*M*M);
    gpuCheck( cudaBindTexture(NULL, texref3, refPtr, M*M*M) );
    
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < 10; i++){
        touch3Dtexture<<<blocks, threads>>>(outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaUnbindTexture(texref) );    
    gpuCheck( cudaFree(refPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("texture 3D: %f ms\n", 1000*delta.count());
}
