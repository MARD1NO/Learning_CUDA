#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * This example demonstrates the impact of misaligned writes on performance by
 * forcing misaligned writes to occur on a float*.
 */

void checkResult(float *hostRef, float *gpuRef, const int N, const int offset)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = offset; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("different on %dth element: host %f gpu %f\n", i, hostRef[i],
                    gpuRef[i]);
            break;
        }
    }

    if (!match)  printf("Arrays do not match.\n\n");
}

void initialData(float *ip,  int size)
{
    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 100.0f;
    }

    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int n, int offset)
{
    for (int idx = offset, k = 0; idx < n; idx++, k++)
    {
        C[idx] = A[k] + B[k];
    }
}

__global__ void readWriteOffset(float *A, float *B, float *C, const int n, int offset){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    unsigned int k = i + offset; 
    if(k < n){
        C[k] = A[k] + B[k];
    }
}

__global__ void readWriteOffsetUnroll4(float *A, float *B, float *C, const int n, int offset){
    unsigned int i = blockIdx.x * blockDim.x*4 + threadIdx.x; 
    unsigned int k = i + offset; 
    if(k < n){
        C[k] = A[k] + B[k];
    }
    if(k + blockDim.x < n){
        C[k + blockDim.x] = A[k + blockDim.x] + B[k + blockDim.x];
    }
    if(k + blockDim.x*2 < n){
        C[k + blockDim.x*2] = A[k + blockDim.x*2] + B[k + blockDim.x*2];
    }
    if(k + blockDim.x*3 < n){
        C[k + blockDim.x*3] = A[k + blockDim.x*3] + B[k + blockDim.x*3];
    }
}

__global__ void warmup(float *A, float *B, float *C, const int n, int offset)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int k = i + offset;

    if (k < n) C[k] = A[i] + B[i];
}



int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up array size
    int nElem = 1 << 22; // total number of elements to reduce
    printf(" with array size %d\n", nElem);
    size_t nBytes = nElem * sizeof(float);

    // set up offset for summary
    int blocksize = 512;
    int offset = 0;

    if (argc > 1) offset    = atoi(argv[1]);

    if (argc > 2) blocksize = atoi(argv[2]);

    // execution configuration
    dim3 block (blocksize, 1);
    dim3 grid  ((nElem + block.x - 1) / block.x, 1);

    // allocate host memory
    float *h_A = (float *)malloc(nBytes);
    float *h_B = (float *)malloc(nBytes);
    float *hostRef = (float *)malloc(nBytes);
    float *gpuRef  = (float *)malloc(nBytes);

    // initialize host array
    initialData(h_A, nElem);
    memcpy(h_B, h_A, nBytes);

    // summary at host side
    sumArraysOnHost(h_A, h_B, hostRef, nElem, offset);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // copy data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_A, nBytes, cudaMemcpyHostToDevice));

    // warmup
    double iStart = cpuSecond();
    warmup<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    double iElaps = cpuSecond() - iStart;
    printf("warmup      <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
           block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // readWriteOffset
    iStart = cpuSecond();
    readWriteOffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("readWriteOffset <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x,
           block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // readWriteOffset
    iStart = cpuSecond();
    readWriteOffsetUnroll4<<<grid.x/4, block>>>(d_A, d_B, d_C, nElem, offset);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("readWriteOffsetUnroll4 <<< %4d, %4d >>> offset %4d elapsed %f sec\n", grid.x/4,
           block.x, offset, iElaps);
    CHECK(cudaGetLastError());

    // free host and device memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
