#include "cuda_runtime.h"
#include "stdio.h"

__device__ float devData; 

__global__ void checkGlobalVariable(){
    printf("Device: the value of the global variable is: %f\n", devData);
    devData += 2.0f;
}

int main(void){
    float value = 3.14f; 
    float *dptr = NULL; 
    cudaGetSymbolAddress((void **)&dptr, devData);
    cudaMemcpy(dptr, &value, sizeof(float), cudaMemcpyHostToDevice);
    printf("Host: copied %f to the global variable \n", value);

    checkGlobalVariable<<<1, 1>>>();

    // copy back 
    cudaMemcpy(&value, dptr, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Host the value changed by the kernel to %f \n", value);

    cudaDeviceReset(); 
    return EXIT_SUCCESS; 

}