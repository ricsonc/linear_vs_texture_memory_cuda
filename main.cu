#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <cuda.h>

typedef float4 typ;

texture<typ, 1> texref;
texture<typ, 2> texref2;
texture<typ, 3> texref3;
int RUNS = 1000;

/* some utilities */

#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x+b.x, a.y+b.y);
}

__device__ float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

/* 1D linear memory */

__global__ void touch1Dlinear(void* devPtr_, void* outPtr_, long M)
{
    long N = gridDim.x * blockDim.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    typ* devPtr = (typ*) devPtr_;
    typ* outPtr = (typ*) outPtr_;

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

    gpuCheck( cudaMalloc(&devPtr, M*sizeof(typ)) );
    gpuCheck( cudaMalloc(&outPtr, M*sizeof(typ)) );

    auto start = std::chrono::system_clock::now();

    for(int i = 0; i < RUNS; i ++){
        touch1Dlinear<<<blocks,threads>>>(devPtr, outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );
    
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaFree(devPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("linear 1D: %.1f ms\n", delta.count());
}

/* 1D texture memory */

__global__ void touch1Dtexture(void* outPtr_, long M){
    long N = gridDim.x * blockDim.x;
    long i = blockIdx.x * blockDim.x + threadIdx.x;

    typ* outPtr = (typ*) outPtr_;

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

    gpuCheck( cudaMalloc(&refPtr, M*sizeof(typ)) );
    gpuCheck( cudaMalloc(&outPtr, M*sizeof(typ)) );
    gpuCheck( cudaBindTexture(NULL, texref, refPtr, M) );
    
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < RUNS; i++){
        touch1Dtexture<<<blocks, threads>>>(outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaUnbindTexture(texref) );    
    gpuCheck( cudaFree(refPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("texture 1D: %.1f ms\n", delta.count());
}

/* 2D linear memory */

__global__ void touch2Dlinear(void* devPtr_, void* outPtr_, long M)
{
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;

    typ* devPtr = (typ*) devPtr_;
    typ* outPtr = (typ*) outPtr_;

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

    gpuCheck( cudaMalloc(&devPtr, M*M*sizeof(typ)) );
    gpuCheck( cudaMalloc(&outPtr, M*M*sizeof(typ)) );

    auto start = std::chrono::system_clock::now();

    for(int i = 0; i < RUNS; i ++){
        touch2Dlinear<<<blocks,threads>>>(devPtr, outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );
    
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaFree(devPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("linear 2D: %.1f ms\n", delta.count());
}

/* 2D texture memory */

__global__ void touch2Dtexture(void* outPtr_, long M){
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;

    typ* outPtr = (typ*) outPtr_;

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
    long M = 10000;
    dim3 blocks(256, 256);
    dim3 threads(8,8);

    void* outPtr;
    gpuCheck( cudaMalloc(&outPtr, M*M*sizeof(typ)) );

    cudaArray *refPtr;
    gpuCheck( cudaMallocArray(&refPtr, &texref2.channelDesc, M, M) );
    gpuCheck( cudaBindTextureToArray(texref2, refPtr) );
    
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < RUNS; i++){
        touch2Dtexture<<<blocks, threads>>>(outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaUnbindTexture(texref) );    
    gpuCheck( cudaFreeArray(refPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("texture 2D: %.1f ms\n", delta.count());
}

/* 3D linear memory */

__global__ void touch3Dlinear(void* devPtr_, void* outPtr_, long M)
{
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;
    long iz = blockIdx.z * blockDim.z + threadIdx.z;

    typ* devPtr = (typ*) devPtr_;
    typ* outPtr = (typ*) outPtr_;

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

    gpuCheck( cudaMalloc(&devPtr, M*M*M*sizeof(typ)) );
    gpuCheck( cudaMalloc(&outPtr, M*M*M*sizeof(typ)) );

    auto start = std::chrono::system_clock::now();

    for(int i = 0; i < RUNS; i ++){
        touch3Dlinear<<<blocks,threads>>>(devPtr, outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );
    
    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaFree(devPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("linear 2D: %.1f ms\n", delta.count());
}

/* 3D texture memory */

__global__ void touch3Dtexture(void* outPtr_, long M){
    long N = gridDim.x * blockDim.x;
    long ix = blockIdx.x * blockDim.x + threadIdx.x;
    long iy = blockIdx.y * blockDim.y + threadIdx.y;
    long iz = blockIdx.z * blockDim.z + threadIdx.z;

    typ* outPtr = (typ*) outPtr_;

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
    unsigned long M = 465;
    dim3 blocks(32,32,32);
    dim3 threads(4,4,4);

    void* outPtr;
    gpuCheck( cudaMalloc(&outPtr, M*M*M*sizeof(typ)) );
    
    cudaArray* refPtr;
    gpuCheck( cudaMalloc3DArray(&refPtr, &texref2.channelDesc, {M, M, M}) );
    gpuCheck( cudaBindTextureToArray(texref3, refPtr) );
    
    auto start = std::chrono::system_clock::now();
    for(int i = 0; i < RUNS; i++){
        touch3Dtexture<<<blocks, threads>>>(outPtr, M);
    }
    
    gpuCheck( cudaPeekAtLastError() );
    gpuCheck( cudaDeviceSynchronize() );

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> delta = end-start;

    gpuCheck( cudaUnbindTexture(texref) );    
    gpuCheck( cudaFreeArray(refPtr) );
    gpuCheck( cudaFree(outPtr) );
    
    printf("texture 3D: %.1f ms\n", delta.count());
}
