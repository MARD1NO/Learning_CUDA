#include <stdio.h>

__global__ void helloFromGPU(void){
    if(threadIdx.x == 5){
        printf("Hello World from GPU! thread: %d \n", threadIdx.x);
    }
}

int main(void){
    // Hello from CPU
    printf("Hello World from CPU! \n");
    // helloFromGPU<<<1, 10>>>();
    helloFromGPU<<<1, 10>>>();
    // cudaDeviceReset();
    cudaDeviceSynchronize();
    return 0;
}