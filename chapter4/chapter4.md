# 4.1 CUDA内存模型概述

## 4.1.1 内存层次结构的优点

应用程序往往遵循局部性原则，**即可以在任意时间点访问相对较小的局部地址空间**

有两种类型局部性

- 时间局部性认为如果一个数据位置被引用，那么该数据在较短的时间周期内很可能会再次被引用
- 空间局部性认为如果一个内存位置被引用，则附近的位置也可能被引用

CPU和GPU的主存都采用的是DRAM（动态随机存取存储器），而低延迟内存（CPU一级缓存）使用的则是SRAM（静态随机存取存储器）。当数据被处理器频繁使用时，该数据保存在低延迟，低容量的存储器中；如果只是被存储以备后用时，数据就存储在高延迟，大容量的存储器中。

## 4.1.2 CUDA内存模型

一般来说，存储器分为可编程和不可编程，在CPU中，一级缓存和二级缓存都是不可编程的存储器。在CUDA中，可编程内存类型如下

- 寄存器
- 共享内存
- 本地内存
- 常量内存
- 纹理内存
- 全局内存

具体来说：

1. 一个核函数中的线程都有自己**私有的本地内存**
2. 一个线程块有自己的 **共享内存**，对同一线程块的所有线程都可见
3. 所有线程都可以访问全局内存
4. 所有线程都能访问的 **只读内存空间**有：常量内存空间和纹理内存
5. 纹理内存为各种数据布局提供了不同的寻址模式和滤波模式

### 4.1.2.1 寄存器

寄存器是GPU上运行速度最快的内存空间，核函数声明的一个没有其他修饰符的自变量，**通常存储在寄存器中**

在核函数声明的数组中，**如果引用该数组的索引是常量且能在编译时确定**，那么该数组也存储在寄存器中



寄存器变量对于每个线程来说是私有的，一个核函数通常用寄存器来保存频繁访问的线程私有变量。 寄存器变量与核函数生命周期相同，即一旦核函数执行完毕，就不能访问寄存器变量



寄存器是一个在SM中由活跃线程束划分出的较少资源，**在核函数中使用较少的寄存器将使在SM上有更多的常驻线程块**。每个SM上并发线程块越多，使用率和性能越高



如果一个核函数使用了超过硬件限制数量的寄存器，则会用**本地内存**替代多占用的寄存器。为了避免寄存器溢出带来的不利影响，nvcc编译器会使用启发式策略最小化寄存器使用。我们也可以给每个核函数显式加上信息帮助编译器优化

```cpp
__global__ void 
__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
kernel(...){
    ...
}
```

maxThreadsPerBlock指的是每个线程块可以包含的最大线程数，minBlocksPerMultiprocessor指的是每个SM中预期的最小的常驻线程块数量。

我们也可以用`maxrrecount`编译选项，指定所有核函数使用寄存器的最大数量

### 4.1.2.2 本地内存

编译器可能存放到本地内存中的变量有：

- 在编译时使用未知索引引用的本地数组
- 可能会占用大量寄存器空间的较大本地结构体或数组
- 任何不满足核函数寄存器限定条件的变量

溢出到本地内存中的变量本质上与全局内存在同一块存储区域，因此本地内存访问的特点是高延迟和低带宽

### 4.1.2.3 共享内存

核函数中使用 `__shared__`修饰符修饰的变量存放在共享内存中

它具有更高的带宽和更低的延迟。每一个SM都有一定数量的由线程块分配的共享内存，因此如果过度使用共享内存，则会限制活跃线程束的数量

虽然共享内存在核函数范围内声明，但是其生命周期伴随着整个线程块。**一个线程块执行结束后，其分配的共享内存将被释放并重新分配给其他线程块**

共享内存是线程之间相互通信的基本方式，一个块内的线程通过共享内存进行合作，前提是同步使用以下调用

`__syncthreads()`，该函数保证所有线程在其他线程被允许执行前，到达该处。当然这个函数强制SM到空闲状态，会影响性能

### 4.1.2.4 常量内存

常量内存驻留在设备内存中，并在每个SM专用的常量缓存中缓存。使用如下修饰符来修饰：

`__constant__`

常量变量必须在**全局空间内和所有核函数之外**进行声明，对所有计算能力的设备只可以声明64KB的内存，对同一编译单元的所有核函数可见

核函数对于常量内存的操作只有**读**，常量内存需要在**主机端**使用下面的函数来初始化

```cpp
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count)
```

线程束中的所有线程从相同的内存地址中读取数据时，常量内存表现最好。

### 4.1.2.5 纹理内存

忽略

### 4.1.2.6 全局内存

全局内存是GPU中最大，延迟最高且最常使用的内存。

它的声明可以在任何SM设备上被访问到，并且贯穿应用程序的整个生命周期



全局内存变量可以被静态声明或动态声明。在设备代码中，静态地声明一个变量：

`__device__`

动态分配就是之前cudaMalloc那套

全局内存分配空间存在于应用程序的整个生命周期中，并且可以访问所有核函数中的所有线程。由于线程的执行不能跨线程块同步，**不同块内的线程并发修改全局内存的同一位置会出现问题**



全局内存常驻于设备内存中，可通过32/64/128字节的内存事务进行访问，**这些内存事务需要自然对齐**，即首地址必须是32/64/128字节的倍数

一般情况下，用来满足内存请求的事务越多，未使用的字节被传输回的可能性越高，这就造成数据吞吐量降低

### 4.1.2.9 静态全局内存
可参考 `globalVariable.cu` 文件

需要注意的是
- cudaMemcpyToSymbol 函数是存在CUDA运行时API中的，可以使用GPU硬件来访问
- devData作为一个标识符，并不是设备全局内存的变量地址
- 核函数中，devData被当作全局内存中的一个变量

下面这种写法是错误的
```cpp
cudaMemcpy(&devData, &value, sizeof(float), cudaMemcpyHostToDevice)
```
我们不能在主机端的设备变量中使用运算符 &，因为它只是一个在GPU上表示物理位置符号。

但我们可以使用下面的函数来获取全局变量的地址
```cpp
cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol)
```
完整代码如下
```cpp
float *dptr = NULL; 
cudaGetSymbolAddress((void**)&dptr, devData);
cudaMemcpy(dptr, &value, sizeof(float), cudaMemcpyHostToDevice);
```

在CUDA编程中，一般情况下设备核函数不能访问主机变量，主机函数也不能访问设备变量，即使这些变量在同一文件作用域下声明。

# 4.2 内存管理
## 4.2.1 内存分配和释放
我们在主机上使用下列函数分配全局内存
```cpp
cudaError_t cudaMalloc(void **devPtr, size_t count);
```
该函数分配了count字节的全局内存，并用devptr指针**返回该内存的地址**（即地址的地址）

我们可以用
```cpp
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
```
来初始化设备内存

释放的时候调用`cudaFree(void *devPtr)`

设备内存的分配和释放操作成本较高，所以应用程序应重利用设备内存，以减少对整体性能的影响

## 4.2.2 内存传输
使用
```cpp
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);
```
从主机向设备传输数据

kind表示的是复制方向，有
- cudaMemcpyHostToHost
- cudaMemcpyHostToDevice
- cudaMemcpyDeviceToHost
- cudaMemcpyDeviceToDevice

CUDA编程的基本原则是尽可能减少主机与设备之间的传输

## 4.2.3 固定内存
GPU不能直接在可分页主机内存上安全地访问数据

当从可分页主机内存传输数据到设备内存是时，CUDA首先分配临时页面锁定的或固定的主机内存，将主机源数据分配到固定内存中，然后从固定内存传输数据给设备内存

但是CUDA提供了一个函数
```cpp
cudaError_t cudaMallocHost(void **devptr, size_t count)
```
该函数分配了count字节的主机内存，**这些内存是页面锁定的并且对设备来说是可访问的**

固定内存能被设备直接访问，所以它能用比**可分页内存高得多的带宽**进行读写，然而，**分配过多的固定内存可能会降低主机系统的性能**，因为它减少了用于存储虚拟内存数据的可分页内存的数量
```cpp
cudaError_t status = cudaMallocHost((void **)&h_aPinned, bytes);
if (status != cudaSuccess){
    fprintf(stderr, "Error returned from pinned host memory allocation \n");
    exit(1);
}
```
固定主机内存需通过
```cpp 
cudaError_t cudaFreeHost(void *ptr);
```
来释放

tips: 
与可分页内存相比，固定内存的分配和释放成本更高，但它为大规模数据传输提供更高的传输吞吐量
应该尽可能地减少或重叠主机和设备间的数据传输

## 4.2.4 零拷贝内存
主机和设备不能互相访问各自的变量，但有个例外是**零拷贝内存**，主机和设备都可以访问

CUDA核函数中使用零拷贝内存有以下优势
- 当设备内存不足时可利用主机内存
- 避免主机和设备间的显式数据传输
- 提高PCIE传输速率

使用零拷贝内存时，必须**同步主机和设备间的内存访问**

零拷贝内存是固定内存，使用下面函数创建一个到固定内存的映射
```cpp
cudaError_t cudaHostAlloc(void **pHost, size_t count, unsigned int flags);
```

flags参数包括
- cudaHostAllocDefault 使cudaHostAlloc函数行为与cudaMallocHost函数一致
- cudaHostAllocPortable 返回能被所有CUDA上下文使用的固定内存
- cudaHostAllocWriteCombined 返回写结合内存，该内存可以在某些系统配置上通过PCIe总线上更快地传输，但是它在大多数主机上不能被有效读取，写结合内存是缓冲区的一个很好的选择
- cudaHostAllocMapped 返回主机写入和设备读取被映射到设备地址空间中的主机内存

```cpp
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
```
该函数返回了一个在pDevice中的设备指针，该指针可以在设备上被引用以访问映射得到的固定主机内存

在进行频繁的读写操作时，**使用零拷贝内存作为设备内存的补充会显著降低性能，因为映射到内存的传输必须经过PCIe总线。**

```cpp
CHECK(cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocMapped));
CHECK(cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocMapped));
// pass the pointer to device
CHECK(cudaHostGetDevicePointer((void **)&d_A, (void *)h_A, 0));
CHECK(cudaHostGetDevicePointer((void **)&d_B, (void *)h_B, 0));
```

tips: 
异构计算系统架构分为集成架构和离散架构
集成架构即CPU和GPU集成在一个芯片上，在物理地址上共享主存，此时无需在PCIe总线上备份，零拷贝内存在编程和性能上更佳
而离散架构下，设备通过PCIe总线连接到主机，零拷贝内存仅在某些情况下有优势

## 4.2.5 统一虚拟地址
统一虚拟寻址UVA可以让主机内存和设备内存共享同一个虚拟地址空间。

在之前的零拷贝例子中，我们需要
1. 分配映射的固定主机内存
2. 使用CUDA运行时函数`cudaHostGetDevicePointer`获取映射到固定内存的设备指针
3. 将设备指针传递给核函数

而有了UVA，我们无需获取设备指针，可以直接
```cpp
CHECK(cudaHostAlloc((void **)&h_A, nBytes, cudaHostAllocMapped));
CHECK(cudaHostAlloc((void **)&h_B, nBytes, cudaHostAllocMapped));

initialData(h_A, nElem);
initialData(h_B, nElem);

sumArraysZeroCopy<<<grid, block>>>(h_A, h_B, d_C, nElem);
```

## 4.2.6 统一内存寻址
统一内存中创建了托管内存池，内存池中已分配的空间可以用相同的内存地址在CPU和GPU上进行访问。底层系统在统一内存空间中自动在主机和设备之间进行数据传输。

托管内存可以通过静态分配（即添加 `__managed__` 注释），也可以动态分配，即
```cpp
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags=0)
```

