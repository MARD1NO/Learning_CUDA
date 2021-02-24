#include "cuda_runtime.h"
#include "stdio.h"

#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess){                                                  \
        printf("Error: %s: %d, ", __FILE__, __LINE__);                          \
        printf("code: %d, reason: %s \n", error, cudaGetErrorString(error));    \
        exit(1);                                                                \
    }                                                                           \
}

void  checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1e-8;
    bool match = 1; 
    for(int i=0; i<N; i++){
        if(abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0; 
            printf("Arrays do not match! \n");
            printf("host %5.2f gpu %5.2fat current %d \n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) { 
        printf("Arrays match! \n");
    }
}

void sumArraysOnHost(float *A, float *B, float *C, const int N){
    // Do elementwise add
    for(int idx=0; idx<N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C){
    // Do elementwise add
    int idx = threadIdx.x;
    // int idx = blockIdx.x*blockDim.x + threadIdx.x;
    C[idx] = A[idx] + B[idx];
}

void initialData(float *ip, int size){
    // Generate different seed for random number
    time_t t; 
    srand((unsigned int) time(&t));
    for(int i=0; i<size; i++){
        ip[i] = (float)(rand() &0xFF) /10.0f; 
    }
}

int main(){
    printf("starting... \n");

    // set up device 
    int dev = 0; 
    cudaSetDevice(dev);

    // set up data size of vectors
    int nElem = 32; 
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initial data
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);


    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // invoke kernel at host side
    dim3 block(nElem);
    dim3 grid(nElem/block.x);

    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C);
    printf("Execution configuration <<<%d %d>>> \n", grid.x, block.x);
    
    // copy kernel result back to host side
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // add vector at host side for result
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    //free host memory
    free(h_A);
    free(h_B);
    free(gpuRef);
    free(hostRef);
    return 0;
}
