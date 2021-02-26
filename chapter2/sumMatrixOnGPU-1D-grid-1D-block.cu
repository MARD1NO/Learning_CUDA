#include "cuda_runtime.h"
#include "stdio.h"
#include "sys/time.h"

#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess){                                                  \
        printf("Error: %s: %d, ", __FILE__, __LINE__);                          \
        printf("code: %d, reason: %s \n", error, cudaGetErrorString(error));    \
        exit(1);                                                                \
    }                                                                           \
}

double cpuSecond(){
    struct timeval tp; 
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void printMatrix(float *C, const int nx, const int ny){
    float *ic = C; 
    printf("\n Matrix: (%d, %d)\n", nx, ny);
    for(int iy=0; iy<ny; iy++){
        for(int ix=0; ix<nx; ix++){
            printf("%.3f ", ic[ix]);
        }
        ic += nx; 
        printf("\n");
    } 
    printf("\n");
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1e-8;
    bool match = 1; 
    for(int i=0; i<N; i++){
        if(abs(hostRef[i] - gpuRef[i]) > epsilon) {
            match = 0; 
            printf("Arrays do not match! \n");
            printf("host %5.2f gpu %5.2f at current %d \n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match) { 
        printf("Arrays match! \n");
    }
}

void initialData(float *ip, int size){
    // Generate different seed for random number
    time_t t; 
    srand((unsigned int) time(&t));
    for(int i=0; i<size; i++){
        ip[i] = (float)(rand() &0xFF) /10.0f; 
    }
}

void sumMatrixOnHost(float *A, float *B, float*C, const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;
    
    for(int iy=0; iy<ny; iy++){
        for(int ix=0; ix<nx; ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; 
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnDevice(float *MatA, float *MatB, float *MatC, const int nx, const int ny){
    unsigned int ix = threadIdx.x + blockIdx.x*blockDim.x; 
    if(ix < nx){
        for(int iy=0; iy<ny; iy++){
            int idx = iy*nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

int main(){
    printf("Starting... \n");

    // set up device 
    int dev = 0; 
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d:  %s \n", dev, deviceProp.name);
    cudaSetDevice(dev);

    // set matrix dimension 
    int nx = 1<<14;
    int ny = 1<<14; 
    int nxy = nx*ny; 
    int nBytes = nxy*sizeof(float);

    // malloc host memory 
    float *h_A, *h_B, *host_ref, *gpu_ref; 
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    host_ref = (float *)malloc(nBytes);
    gpu_ref = (float *)malloc(nBytes);


    // initialize host matrix
    initialData(h_A, nxy);
    // printMatrix(h_A, nx, ny);

    initialData(h_B, nxy);
    // printMatrix(h_B, nx, ny);

    double iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, host_ref, nx, ny);
    double iElaps = cpuSecond();
    double cpu_elapse = iElaps-iStart;
    printf("SumMatrixOnCPU2D elapsed %f sec \n", cpu_elapse);
    // printMatrix(host_ref, nx, ny);
    // malloc device memory
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    // transfer data from host to device 
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // set up execution configuration
    dim3 block(256, 1); 

    dim3 grid((nx+block.x-1)/block.x, 1);

    iStart = cpuSecond();
    //invoke the kernel
    sumMatrixOnDevice<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond();
    double gpu_elapse = iElaps - iStart ; 
    printf("SumMatrixOnGPU2D <<<grid(%d %d), block(%d, %d)>>> elapsed %f sec \n", grid.x, grid.y, \
            block.x, block.y, gpu_elapse);
    printf("The accelerate: %f \n", cpu_elapse/gpu_elapse);
    cudaMemcpy(gpu_ref, d_MatC, nBytes, cudaMemcpyDeviceToHost);
    // printMatrix(gpu_ref, nx, ny);
    // free host and device memory 
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);

    free(h_A);
    free(h_B);
    free(host_ref);
    free(gpu_ref);

    cudaDeviceReset();
    return 0;
}