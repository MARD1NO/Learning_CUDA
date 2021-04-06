#include "../common/common.h"
#include "cuda_runtime.h"
#include "stdio.h"

#define BDIMX 32 
#define BDIMY 32
#define IPAD 1

void printData(char *msg, int *in, const int size){
    printf("%s: ", msg);
    for(int i=0; i < size; i++){
        printf("%5d", in[i]);
        fflush(stdout);
    }
    printf("\n");
    return;
}

__global__ void setRowReadRow(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx; 

    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}

__global__ void setColReadCol(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.x][threadIdx.y] = idx; 

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx; 

    __syncthreads();

    out[idx] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char** argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s at ", argv[0]);
    printf("device %d: %s", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    cudaSharedMemConfig pConfig;
    CHECK(cudaDeviceGetSharedMemConfig(&pConfig));
    printf("with Bank Mode: %s ", pConfig == 1 ? "4-Byte": "8-Byte");

    int nx = BDIMX; 
    int ny = BDIMY; 
    bool iprintf = 0;

    if(argc > 1) iprintf = atoi(argv[1]);

    size_t nBytes = nx * ny * sizeof(int);

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1); 
    printf("<<< grid (%d, %d) block (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

    int *d_C;
    CHECK(cudaMalloc((int**)&d_C, nBytes));
    int *gpuRef = (int *)malloc(nBytes);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setColReadCol<<<grid, block>>>(d_C); 
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if(iprintf) printData("set col read col: ", gpuRef, nx*ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadRow<<<grid, block>>>(d_C); 
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if(iprintf) printData("set row read row: ", gpuRef, nx*ny);

    CHECK(cudaMemset(d_C, 0, nBytes));
    setRowReadCol<<<grid, block>>>(d_C);
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));
    if(iprintf)  printData("set row read col: ", gpuRef, nx * ny);

    CHECK(cudaFree(d_C)); 
    free(gpuRef);
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}