### 异构计算

CPU上的线程通常是重量级的实体，**上下文切换缓慢且开销大**，适合处理控制密集型任务

GPU上的线程是高度轻量级的，相比CPU，GPU可以并发支持更多线程，适合处理计算密集型任务

### CUDA程序组成
CUDA是一种异构计算平台，包含以下两个部分

- 在CPU上运行的主机代码（host），通过C编译器进行编译，如gcc，cl
- 在GPU上运行的设备代码（device），通过nvcc编译器进行编译

### 用GPU输出Hello World

首先编写一个内核函数，由 `__global__` 修饰的函数表示，**该函数会从CPU中调用，然后在GPU上执行**

```cpp
__global__ void helloFromGPU(void){
    if(threadIdx.x == 5){
        printf("Hello World from GPU! thread: %d \n", threadIdx.x);
    }
}
```

三重尖括号里面的参数是执行配置，第一个参数是线程块的个数，第二个参数是线程的个数，具体会在下一章更详细讲解

最后使用内置函数 `cudaDeviceRest()`来显式地释放和清空当前进程与设备用关的资源

完整的程序如下

```cpp 
#include <stdio.h>

__global__ void helloFromGPU(void){
    printf("Hello World from GPU! thread: %d \n", threadIdx.x);
}

int main(void){
    // Hello from CPU
    printf("Hello World from CPU! \n");
    // helloFromGPU<<<1, 10>>>();
    helloFromGPU<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}
```
这里我们使用了10个线程，因此GPU这里会打印10次

将该文件记作 `hello.cu`，使用下面命令编译
```shell
nvcc hello.cu -o hello
```
然后运行生成后的文件
```shell
./hello
```
就会出现相关结果

### CUDA编程结构
1. 分配GPU显存
2. 从CPU内存将数据拷贝到GPU显存
3. 调用核函数来计算
4. 将数据拷贝回CPU内存
5. 释放GPU显存
