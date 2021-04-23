CUDA编程有两个级别的并发
- 内核级并发
- 网格级并发

前面几章探讨的是内核级别，而本章将探讨网格级的并发，**实现多个内核再同一设备上同时执行**

# 6.1 流和事件概述
CUDA流是一系列异步的CUDA操作，这些操作**按照主机代码确定的顺序**在设备上执行

流能封装这些操作并保持操作的顺序，流中操作的执行相对于主机总是异步的。我们的任务是使用CUDA的API来确保一个异步操作在**运行结果被使用之前可以完成**。

同一个CUDA流操作是严格顺序的，**不同CUDA流中的操作在执行顺序上不受限制**。使用多个流同时启动多个内核，可以实现网格级的并发。

CUDA编程经常是一个 数据迁移->执行内核->数据拷贝回主机。执行内核比传输数据耗时更多，我们可以通过将内核执行和数据传输调度到不同的流中，将操作进行重叠，让程序的总运行时间缩短。

从软件的角度看，CUDA操作是在不同的流上并发执行。**在硬件上却不一定如此**，需要根据PCIe总线争用SM资源，完成不同流的CUDA操作可能需要互相等待

## 6.1.1 CUDA流
流分为两种类型
- 隐式声明的流（空流） 如果没显式指定，则内核启动和数据传输都默认使用空流
- 显式声明的流（非空流） 想要重叠就必须显示声明流

基于流的异步的内核启动/数据传输 支持以下类型的粗粒度并发
- 重叠主机计算和设备计算
- 重叠主机计算和主机与设备间的数据传输
- 重叠主机与设备间的数据传输和设备计算
- 并发设备计算

数据传输可以被异步发布，但是必须显式地设置一个CUDA流来装载它们

cudaMemcpy的异步版本函数如下
```cpp
cudaError_t cudaMemcpyAsync(void *dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream=0);
```
第五个参数为流标识符，默认被设置为默认流。该函数与主机是异步的，调用发布后，控制权立即返回到主机。

使用如下代码创建一个非空流
```cpp
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
```
在使用异步CUDA函数时，它们可能从**先前启动的异步操作中返回错误代码**，因此返回错误的API调用**并不一定是产生错误的那个调用**

执行异步数据传输时，必须使用固定主机内存（使用`cudaMallocHost`和`cudaHostAlloc`）
```cpp
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags);
```
在主机虚拟内存中固定分配，**可以确保其在CPU内存中的物理位置在应用程序的整个生命周期中保持不变**。否则，操作系统可以随时自由改变主机虚拟内存的物理位置。

在非默认流启动内核，需要在内核执行配置指定第四个参数
```cpp
kernel_name<<<grid, block, sharedMemsize, stream>>>(argument list);
```
非默认流声明如下
```cpp
cudaStream_t stream;
```
使用如下代码创建
```cpp
cudaStreamCreate(&stream);
```
释放流
```cpp
cudaError_t cudaStreamDestroy(cudaStream_t stream);
```
如果该流中仍有未完成的工作，`cudaStreamDestroy`函数将立即返回。若流中工作完成，与流相关的资源将被自动释放

CUDA提供两个函数来检查流中所有操作是否完成
```cpp
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cuudaStreamQuery(cudaStream_t stream);
```
- `cudaStreamSynchronize` 强制阻塞主机，直到给定流中所有的操作都完成
- `cudaStreamQuery` 会检查流中所有操作是否都已经完成

若完成会返回 `cudaSuccess`，若操作还在执行会返回 `cudaErrorNotReady`

## 6.1.2 流调度
### 6.1.2.1 虚假的依赖关系
虽然支持多路并发，但是所有的流最终是被多路复用到单一的硬件工作队列中。这种单一流水线可能会导致虚假的依赖关系，在工作队列中，一个被阻塞的操作会将队列中该操作后面的所有操作都阻塞，即使它们属于不同的流

### 6.1.2.2 HyperQ技术
Kepler GPU使用多个硬件工作队列，减少了虚假的依赖关系。如果创建的流超过了硬件工作队列的数目，那么多个流会共享一个硬件工作队列

## 6.1.3 流的优先级
对计算能力 >= 3.5 的设备，可以给流分配优先级
```cpp
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
```
该函数创建一个具有指定整数优先级的流，并在pStream返回一个句柄。

高优先级的网格队列可以优先占有低优先级流已经执行的工作。**流优先级不会影响数据传输操作，只对计算内核有影响**

可以使用以下函数查询优先级范围
```cpp
cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority, int *greatestPriority); //函数返回值存放在leastxxx和greatestxxx中 
```

## 6.1.4 CUDA事件
CUDA中事件本质是CUDA流的标记，它与该流内操作流中特定点相关联。可以执行以下两个基本任务
- 同步流的执行
- 监控设备的进展
CUDA API提供了在流中任意点插入时间以及查询事件完成的函数。只有当一个给定CUDA流中先前的所有操作都执行结束后，记录在该流内的事件才会起作用

### 6.1.4.1 创建和销毁
```cpp
cudaEvent_t event; // 事件声明
cudaError_t cudaEventCreate(cudaEvent_t* event); // 创建
cudaError_t cudaEventDestroy(cudaEvent_t* event); // 销毁
```

### 6.1.4.2 记录事件和计算运行时间
事件在流执行中标记了一个点，可用来检查**正在执行的流操作是否已经到达了给定点**

一个事件使用如下函数排队进入CUDA流
```cpp
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream=0);
```
等待一个事件会阻塞主机线程的调用
```cpp
cudaError_t cudaEventSynchronize(cudaEvent_t event);
```
可以使用如下代码测试一个事件是否可以不用阻塞主机应用程序来完成
```cpp
cudaError_t cudaEventQuery(cudaEvent_t event);
```
我们可以使用以下函数，计算两个事件标记的CUDA操作执行时间(以毫秒为单位)，事件的启动和停止不必在同一个CUDA流
```cpp
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop);
```
完整的示例代码如下
```cpp
cudaEvent_t start, stop; 
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
kernel<<<grid, block>>>xxx(...);
cudaEventRecord(stop);

cudaEventSynchronize(stop); // sync until the stop event finish
float time; 
cudaEventElapsedTime(&time, start, stop); 

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

## 6.1.5 流同步
从主机的角度，CUDA操作可以分为两大类
- 内存相关操作
- 内核启动

流分两种类型
- 异步流(非空流)
- 同步流(空流/默认流)
在主机上非空流是一个异步流，其上所有的操作都不阻塞主机执行

非空流进一步分为
- 阻塞流
- 非阻塞流

**虽然非空流在主机上是非阻塞的**，但是非空流内的操作**可以被空流中的操作所阻塞**，若一个非空流在主机上是非阻塞的，则空流可以阻塞该非空流中的操作。

### 6.1.5.1 阻塞流和非阻塞流
使用cudaStreamCreate函数创建的流是阻塞流，需要一直等到空流中先前的操作执行结束。

空流是隐式流，**在相同的CUDA上下文中它和其他所有的阻塞流同步**。
```cpp
kernel_1<<<1, 1, 0, stream_1>>>(); 
kernel_2<<<1, 1>>>(); // 创建空流
kernel_3<<<1, 1, 0, stream_2>>>;
```
cuda运行时提供一个控制流的行为
```cpp
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* p_stream, unsigned int flags);
```
flags参数决定是否阻塞
- cudaStreamDefault 默认行为，即阻塞
- cudaStreamNonBlocking 使非空流对空流的阻塞行为失效
若 `stream_1/2` 使用 `cudaStreamNonBlocking` 创建，则所有核函数执行都不会被阻塞

### 6.1.5.2 隐式同步
`cudaMemcpy`可以隐式同步设备和主机，这是由于主机的应用程序在数据传输完成之前被阻塞。
然而该函数主要目的不是同步，所以它的同步是隐式的

无意中调用隐式同步主机和设备的函数，可能会导致意想不到的性能下降

许多与内存相关的操作意味着在当前设备上所有先前的操作上都有阻塞，如
- 锁页主机内存分配
- 设备内存分配
- 设备内存初始化
- 同一设备上两个地址之间的内存复制
- 一级缓存/共享内存配置的修改

### 6.1.5.3 显式同步
显式同步CUDA程序的几种方法
- 同步设备 `cudaDeviceSynchronize(void);`
- 同步流 `cudaStreamSynchronize(stream)`  / `cudaStreamQuery(stream)`
- 同步流中的事件 `cudaEventSynchronize(event)`  / `cudaEventQuery(event)`
- 使用事件跨流同步 `cudaStreamWaitEvent(stream, event)` 能使指定流等待指定事件

### 6.1.5.4 可配置事件
我们可用`cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags)`来定制事件的行为和性能

标志包括
- cudaEventDefault 
- cudaEventBlockingSync 指定使用 cudaEventSynchronize 函数同步事件
- cudaEventDisableTiming 表明创建的事件只能用来同步
- cudaEventInterprocess 表明创建的事件可能被用作进程间事件

# 6.2 并发内核执行
## 6.2.1 非空流中的并发内核
我们先创建4个简单的内核
```cpp
__global__ void kernel_1()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}
```
然后用for循环创建4个流
```cpp
for (int i = 0 ; i < n_streams ; i++)
    {
        CHECK(cudaStreamCreate(&(streams[i])));
    }
```
在默认流里创建两个事件，并在内核执行前，开始记录
```cpp
// creat events
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));

// record start event
CHECK(cudaEventRecord(start, 0));
```
将四个核函数分别在四个流中调用
```cpp
for (int i = 0; i < n_streams; i++)
{
    kernel_1<<<grid, block, 0, streams[i]>>>();
    kernel_2<<<grid, block, 0, streams[i]>>>();
    kernel_3<<<grid, block, 0, streams[i]>>>();
    kernel_4<<<grid, block, 0, streams[i]>>>();
}
```
停止事件，并同步，计算时间
```cpp
// record stop event
CHECK(cudaEventRecord(stop, 0));
CHECK(cudaEventSynchronize(stop));

// calculate elapsed time
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
```
## 6.2.2 Fermi GPU上的虚假依赖关系
如果采用深度优先的方法，它会在当前流i进行任务调度，当最后一个任务被启动时，调度下一个任务，即流i+1的第一个任务。因为这两个任务不依赖，所以可以启动，这就导致重叠执行只有每个流的结束与下一个流的开始处。

采用深度优先的代码如下
```cpp
for (int i = 0; i < n_streams; i++)
{
    kernel_1<<<grid, block, 0, streams[i]>>>();
    kernel_2<<<grid, block, 0, streams[i]>>>();
    kernel_3<<<grid, block, 0, streams[i]>>>();
    kernel_4<<<grid, block, 0, streams[i]>>>();
}

// K1 K2 K3 K4 K1 K2 K3 K4 ...
```

而采用广度优先的方法，会让工作队列中相邻的任务来自于不同的流，实现加速。
```cpp
for (int i = 0; i < n_streams; i++)
    kernel_1<<<grid, block, 0, streams[i]>>>();

for (int i = 0; i < n_streams; i++)
    kernel_2<<<grid, block, 0, streams[i]>>>();

for (int i = 0; i < n_streams; i++)
    kernel_3<<<grid, block, 0, streams[i]>>>();

for (int i = 0; i < n_streams; i++)
    kernel_4<<<grid, block, 0, streams[i]>>>();

// K1 K1 K1 K1 K2 K2 ...
```


## 6.2.3 使用OpenMP的调度操作
OpenMP是CPU的并行编程模型，通过编译器指令来识别并行区域
```cpp
omp_set_num_threads(n_streams); // 指定并行区域要用到的CPU核心数量
#pragma omp parallel 
{
    ...
}
```
一般情况下，如果每个流在内核执行之前，期间或之后**有额外的工作待完成**，那么它可以包含在同一个OpeenMP并行区域里，并且跨流和线程进行重叠。

## 6.2.4 用环境变量调整流行为
```
export CUDA_DEVICE_MAX_CONNECTIONS=32 
或在程序里 setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1)
```

## 6.2.5 GPU资源的并发限制
实际应用里，内核会创建大量线程。这么多的线程会导致硬件资源缺少，是并发的主要限制因素

## 6.2.6 默认流的阻塞行为
我们在深度优先的基础上，将kernel3设置到默认流中
```cpp
for (int i = 0; i < n_streams; i++)
{
    kernel_1<<<grid, block, 0, streams[i]>>>();
    kernel_2<<<grid, block, 0, streams[i]>>>();
    kernel_3<<<grid, block>>>();
    kernel_4<<<grid, block, 0, streams[i]>>>();
}
```

## 6.2.7 创建流间依赖关系
我们可以使用同步事件来创建流之间的依赖关系
首先将标志设置为 `cudaEventDisableTiming`
```cpp
cudaEvent_t *kernelEvent;
kernelEvent = (cudaEvent_t *) malloc(n_streams * sizeof(cudaEvent_t));

for (int i = 0; i < n_streams; i++)
{
    CHECK(cudaEventCreateWithFlags(&(kernelEvent[i]),
                cudaEventDisableTiming));
}
```
然后将每个事件都排入每个流中，我们只对最后一个流做特殊处理，他需要等待其他流完成后才开始
```cpp
for (int i = 0; i < n_streams; i++)
{
    kernel_1<<<grid, block, 0, streams[i]>>>();
    kernel_2<<<grid, block, 0, streams[i]>>>();
    kernel_3<<<grid, block, 0, streams[i]>>>();
    kernel_4<<<grid, block, 0, streams[i]>>>();

    CHECK(cudaEventRecord(kernelEvent[i], streams[i]));
    CHECK(cudaStreamWaitEvent(streams[n_streams - 1], kernelEvent[i], 0));
}
```



