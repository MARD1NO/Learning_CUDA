#include "../common/common.h"
#include "cuda_runtime.h"
#include "stdio.h"

int recursiveReduce(int *data, int const size){
    // teminate recur
    if(size == 1){
        return data[0];
    }

    int stride = size / 2;

    for(int i = 0; i < stride; i++){
        data[i] += data[i+stride];
    }
    return recursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    int tid = threadIdx.x; 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to local block data pointer
    int *idata = g_idata + blockIdx.x*blockDim.x;

    // boundary check
    if(idx >= n) return; 

    for(int stride=1; stride<blockDim.x; stride *= 2){
        if((tid%(2*stride)) == 0){
            idata[tid] += idata[tid+stride];
        }

        // synchonize within threadBlock
        __syncthreads();
    }

    //  write output data
    if(tid == 0){
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    int tid = threadIdx.x; 
    int idx = blockIdx.x*blockDim.x + threadIdx.x; 

    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockDim.x*blockIdx.x; 

    // boundary check
    if(idx >= n) return; 

    for(int stride=1; stride < blockDim.x; stride*=2){
        int index = 2*stride*tid; 
        if(index<blockDim.x){
            idata[index] += idata[index+stride];
        }
        // synchronize within threadBlock
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid==0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    int tid = threadIdx.x;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x*blockDim.x; 

    // boudary check 
    if(idx >= n) return; 

    // in-place reduction in global memory 
    // >>= 1 等价于 /2
    for(int stride = blockDim.x / 2; stride > 0; stride >>=1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    int tid = threadIdx.x;
    int idx = blockIdx.x*blockDim.x*2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x*blockDim.x*2; 

    // unrolling 2 data blocks
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx+blockDim.x];
    __syncthreads();
    
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    int tid = threadIdx.x;
    int idx = blockIdx.x*blockDim.x*4 + threadIdx.x;

    // convert global data pointer to the local pointer of this block 
    int *idata = g_idata + blockIdx.x*blockDim.x*4; 

    // unrolling 4 data blocks
    if (idx + blockDim.x*3 < n) g_idata[idx] =  g_idata[idx] + g_idata[idx+blockDim.x] + g_idata[idx+blockDim.x*2] + g_idata[idx+blockDim.x*3];
    __syncthreads();
    
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }
        __syncthreads();
    }
    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}


__global__ void reduceUnrolling8 (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceUnrollWarps8(int *g_idata, int *g_odata, int n){
    // set thread ID
    int tid = threadIdx.x; 
    int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer
    int *idata = g_idata + blockIdx.x*blockDim.x*8;

    // unrolling 8
    if(idx + 7*blockDim.x < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    // inplace reduction in global memory 
    for(int stride=blockDim.x / 2; stride > 32; stride >>= 1){
        if(tid < stride){
            idata[tid] += idata[tid+stride];
        }

        __syncthreads();
    }

    // unrolling warp 
    if (tid < 32){
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8 (int *g_idata, int *g_odata,
        unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceCompleteUnrollWarps8 (int *g_idata, int *g_odata,
    int n){
    // set thread ID
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

template<int iBlocksize>
__global__ void reduceCompleteUnroll(int *g_idata, int *g_odata,
    int n){
    // set thread ID
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        int b2 = g_idata[idx + 5 * blockDim.x];
        int b3 = g_idata[idx + 6 * blockDim.x];
        int b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (iBlocksize >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (iBlocksize >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (iBlocksize >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (iBlocksize >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


int main(int argc, char **argv){
    // set up device 
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s", dev, deviceProp.name);
    cudaSetDevice(dev);

    bool bResult = false; 
    // initialization 
    int size = 1<<18; // total number of elements to reduce 
    printf("    with array size %d    ", size);

    // execution configuration
    int blocksize = 512; 
    if(argc > 1){
        blocksize = atoi(argv[1]);
    }
    dim3 block(blocksize, 1);
    dim3 grid((size+block.x-1)/block.x, 1);

    printf("grid %d block %d \n", grid.x, block.x);

    // allocate host memory 
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x*sizeof(int));
    int *tmp = (int *) malloc(bytes);

    // intialize array 
    for(int i = 0; i < size; i++){
        h_idata[i] = (int)(rand()&0xFF);
    }
    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps; 
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL; 
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x*sizeof(int));

    //cpu reduction 
    iStart = cpuSecond();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = cpuSecond() - iStart;
    printf("CPU Reduce elapsed %f ms cpu_sum: %d \n", iElaps, cpu_sum);

    // kernel 1
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size); 
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart; 
    cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i<grid.x; i++){
        gpu_sum += h_odata[i];
    }
    printf("gpu Neighbored elapsed %f ms gpu_sum %d <<<grid %d block %d>>> \n", iElaps, gpu_sum, grid.x, block.x);
    cudaDeviceSynchronize();

    // kernel 2: reduceNeighbored with less divergence
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 4: Unrolling 2
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrolling2<<<grid.x/2, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x/2 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/2; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling2 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x/2, block.x);

    // kernel 5: Unrolling 4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrolling4<<<grid.x/4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x/4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/4; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling4 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x/4, block.x);

    // kernel 6: Unrolling 8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrolling8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling8 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

    // kernel 7: reduceUnrollWarps8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];

    printf("gpu reduceUnrollWarps8 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

    // kernel 8: reduceCompleteUnrollWarps8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();
    reduceUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x/8; i++) gpu_sum += h_odata[i];

    printf("gpu reduceCompleteUnrollWarps8 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);

    // kernel 9: reduceCompleteUnroll
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = cpuSecond();

    switch (blocksize)
    {
    case 1024:
        reduceCompleteUnroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 512:
        reduceCompleteUnroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 256:
        reduceCompleteUnroll<256><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 128:
        reduceCompleteUnroll<128><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 64:
        reduceCompleteUnroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    }

    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // free host memory 
    free(h_idata);
    free(h_odata);
    free(tmp); 

    // free device memory 
    cudaFree(d_idata);
    cudaFree(d_odata);

    // reset device
    cudaDeviceReset(); 

    // check Results 
    bResult = (gpu_sum == cpu_sum);
    if(!bResult){
        printf("Test failed \n");
    }
    return EXIT_SUCCESS;
}

