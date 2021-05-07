#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>


__device__ float myAtomicAdd(float *address, float incr)
{
    unsigned int *typedAddress = (unsigned int *)address; 

    float currentVal = *address; 
    unsigned int expected = __float2uint_rn(currentVal);
    unsigned int desired = __float2uint_rn(currentVal + incr);
    printf("Desired is: %d \n", desired);

    int oldIntValue = atomicCAS(typedAddress, expected, desired);
    
    // Loop while the guess is incorrect.
    while (oldIntValue != expected)
    {
        expected = oldIntValue;
        desired = __float2uint_rn(__uint2float_rn(oldIntValue)+incr);
        oldIntValue = atomicCAS(typedAddress, expected, desired);
    }
    printf("Old int Value is: %d \n", oldIntValue);
    return __uint2float_rn(oldIntValue);
}

__global__ void kernel(float *sharedFloat)
{
    float result = myAtomicAdd(sharedFloat, 2.0f);
    // printf("Result is: %f", result);
}

int main(int argc, char **argv)
{
    // float h_sharedFloat;
    // float *d_sharedFloat;
    // CHECK(cudaMalloc((void **)&d_sharedFloat, sizeof(float)));
    // CHECK(cudaMemset(d_sharedFloat, 0.0, sizeof(float)));

    // kernel<<<1, 32>>>(d_sharedFloat);

    // CHECK(cudaMemcpy(&h_sharedFloat, d_sharedFloat, sizeof(float),
    //                  cudaMemcpyDeviceToHost));

    // printf("1 x 32 increments led to value of %f\n", float(h_sharedFloat));


    int h_sharedFloat;
    float *d_sharedFloat;
    CHECK(cudaMalloc((void **)&d_sharedFloat, sizeof(float)));
    CHECK(cudaMemset(d_sharedFloat, 0.0, sizeof(float)));

    kernel<<<1, 32>>>(d_sharedFloat);

    CHECK(cudaMemcpy(&h_sharedFloat, d_sharedFloat, sizeof(int),
                     cudaMemcpyDeviceToHost));

    printf("1 x 32 increments led to value of %d\n", h_sharedFloat);

    return 0;
}