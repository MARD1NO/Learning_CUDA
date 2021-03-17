#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * simpleDivergence demonstrates divergent code on the GPU and its impact on
 * performance and CUDA metrics.
 */

__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    if (tid % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void mathKernel2(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel3(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    bool ipred = (tid % 2 == 0);

    if (ipred)
    {
        ia = 100.0f;
    }

    if (!ipred)
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void mathKernel4(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    int itid = tid >> 5;

    if (itid & 0x01 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}

__global__ void warmingup(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }

    c[tid] = ia + ib;
}


int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s using Device %d: %s\n", argv[0], dev, deviceProp.name);

    // set up data size
    int size = 64;
    int blocksize = 64;

    if(argc > 1) blocksize = atoi(argv[1]);

    if(argc > 2) size = atoi(argv[2]);

    printf("Data size %d ", size);

    // set up execution configuration
    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // allocate cpu memory
    float *h_C;
    h_C = (float *)malloc(nBytes);

    // run a warmup kernel to remove overhead
    double iStart, iElaps;

    // CHECK(cudaDeviceSynchronize());
    // iStart = cpuSecond();
    // warmingup<<<grid, block>>>(d_C);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = cpuSecond() - iStart;
    // printf("warmup      <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x, iElaps);
    // CHECK(cudaGetLastError());

    // run kernel 1
    iStart = cpuSecond();
    mathKernel1<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("mathKernel1 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x, iElaps);
    CHECK(cudaGetLastError());

    // cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
    // for(int i=0; i<size; i++){
    //     printf(" %f ", h_C[i]);
    // }

    // run kernel 2
    iStart = cpuSecond();
    mathKernel2<<<grid, block>>>(d_C);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("mathKernel2 <<< %4d %4d >>> elapsed %f sec \n", grid.x, block.x, iElaps);
    CHECK(cudaGetLastError());

    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
    for(int i=0; i<size; i++){
        printf(" %f ", h_C[i]);
    }

    // // run kernel 3
    // iStart = cpuSecond();
    // mathKernel3<<<grid, block>>>(d_C);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = cpuSecond() - iStart;
    // printf("mathKernel3 <<< %4d %4d >>> elapsed %d sec \n", grid.x, block.x,
    //        iElaps);
    // CHECK(cudaGetLastError());

    // // run kernel 4
    // iStart = cpuSecond();
    // mathKernel4<<<grid, block>>>(d_C);
    // CHECK(cudaDeviceSynchronize());
    // iElaps = cpuSecond() - iStart;
    // printf("mathKernel4 <<< %4d %4d >>> elapsed %d sec \n", grid.x, block.x,
    //        iElaps);
    // CHECK(cudaGetLastError());

    // free gpu memory and reset divece
    free(h_C);
    CHECK(cudaFree(d_C));
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}