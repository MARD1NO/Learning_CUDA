#include "cuda_runtime.h"
#include "stdio.h"

__device__ float devData[5];

__global__ void checkGlobalVariable(){
    devData[threadIdx.x] += 2.0f; 
}

int main(void){
    float value[5] = {3.14, 3.14, 3.14, 3.14, 3.14};
    cudaMemcpyToSymbol(devData, &value, sizeof(float)*5);
    printf("Copy \n");
    checkGlobalVariable<<<1, 5>>>();

    cudaMemcpyFromSymbol(&value, devData, sizeof(float)*5);

    for(int i=0; i<5; i++){
        printf("%d num is %f \n", i, value[i]);
    }
    cudaDeviceReset();
    return EXIT_SUCCESS;
}