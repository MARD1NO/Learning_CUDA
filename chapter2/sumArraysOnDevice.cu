#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

__global__ void sumArraysOnHost(float *A, float *B, float *C, const int N){
    // Do elementwise add
    int idx = threadIdx.x;
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

int main(int argc, char **argv){
    int nElem = 1024; 
    size_t nBytes = nElem * sizeof(float);
    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, nBytes);
    cudaMalloc((float **)&d_B, nBytes);
    cudaMalloc((float **)&d_C, nBytes);

    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    sumArraysOnHost<<<1, nElem>>>(d_A, d_B, d_C, nElem);
    cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost);
    cudaDeviceReset();
    for(int i=0; i<5; i++){
        printf("A： %f  \n", h_A[i]);
        printf("B： %f  \n", h_B[i]);    
        printf("C： %f  \n", h_C[i]);    
    }
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}