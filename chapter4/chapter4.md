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

# 4.3 内存访问模式
大多数设备端数据访问都是从全局内存开始，因此最大限度利用全局内存带宽是调控核函数性能的基本。

## 4.3.1 对齐与合并访问
核函数的内存请求通常是在DRAM设备和片上内存间以128字节或32字节内存事务实现的

对全局内存的访问都会通过二级缓存，也有部分会通过一级缓存。如果两个缓存都用到，则内存访问是由一个128字节的内存事务实现的，如果只使用了二级缓存，则内存访问时由一个32字节的内存事务实现的

一行一级缓存是128个字节，若线程束每个线程请求一个4字节的值，则每次请求获取128字节数据，恰好对应。

我们需要注意设备内存访问的特性： 
1. 对齐内存访问
2. 合并内存访问

（若启用缓存）当设备内存事务的第一个地址是用于事务服务的缓存粒度的偶数倍时（二级缓存是32字节，一级缓存是128字节），就会出现对齐内存访问

当一个线程束中全部的32个线程访问一个连续的内存块时，就会出现合并内存访问。

## 4.3.2 全局内存读取
SM中，数据通过以下3种缓存进行传输：
1. 一级和二级缓存
2. 常量缓存
3. 只读缓存

可以使用编译器标志禁用一级缓存
```shell
-Xptxas -dlcm=cg # 禁用
-Xptxas -dlcm=ca # 启用
```

总结下，内存加载分为两类
1. 缓存加载（即启用一级缓存）
2. 没有缓存加载（即禁用一级缓存）
如果启用一级缓存，则内存加载被缓存

**对齐与非对齐：若内存访问的第一个地址是32字节的倍数，则是对齐加载**

**合并与未合并：若线程束访问一个连续的数据块，则加载合并**

### 4.3.2.1 缓存加载
GPU一级缓存是专为空间局部性而不是为时间局部性设计的，**频繁访问一个一级缓存的内存位置不会增加数据留在缓存中的概率**


### 4.3.2.2 没有缓存加载
不经过一级缓存，所以它在内存段的粒度上（32个字节）而不是缓存池的粒度上（128个字节）。这种细粒度的加载相较于有缓存的情况下，**对于非对齐和非合并的内存访问有更好的总线利用率**

### 4.3.2.3 非对齐读取的示例
可参考代码 `readSegment.cu`

我们可以用
```shell
nvprof --devices 0 --metrics gld_transactions ./readSegment
```
来获取gld_efficiency指标

#### 4.3.2.4 只读缓存
对于计算能力为3.5及以上的GPU来说，只读缓存支持使用全局内存加载代替一级缓存

只读缓存的加载粒度为32个字节，跟前面内存读取类似，这种更细粒度的加载要优于一级缓存

有两种方式让内存通过只读缓存进行读取： 
1. 使用函数 _ldg
```cpp
__global__ void copyKernel(int *out, int *in){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = __ldg(&in[idx]);
}
```

2. 在间接引用的指针上使用修饰符`__restrict__`
```cpp
__global__ void copyKernel(int * __restrict__ out, const int * __restrict__ in){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in[idx];
}
```

## 4.3.3 全局内存写入
可参考 `writeSegment.cu`

## 4.3.4 结构体数组与数组结构体
定义一个数组结构体
```cpp
struct innerStruct{
    float x;
    float y;
};

struct innerStruct myAOS[N];

// 其内存组织形式如： x y x y x y x y
```
然后我们看下结构体数组
```cpp
struct innerArray{
    float x[N];
    float y[N];
};

// 其内存组织形式如： x x x x y y y y 
```
如果应用程序只需要一个X，那么数组结构体有一半带宽被浪费了（因为有一半需要加载Y），而结构体数组则能利用满带宽。

并行编程如SIMD,CUDA偏向使用结构体数组

关于数组结构体和结构体数组的对比，可分别参考`simpleMathAoS.cu`和`simpleMathSoA.cu`

## 4.3.5 性能调整
优化设备内存带宽利用率有两个目标：
- 对齐及合并内存访问，以减少带宽的浪费
- 足够的并发内存操作，以隐藏内存延迟

实现并发内存访问最大化方式是通过： 
1. 增加每个线程中执行独立内存操作的数量
2. 对核函数启动的执行配置进行实验，以充分体现每个SM的并行性

### 4.3.5.1 展开技术
我们可以跟让每个线程都执行4个独立的内存操作
```cpp
__global__ void readOffsetUnroll4(float *A, float *B, float *C, const int n, int offset){
    unsigned int i = blockIdx.x * blockDim.x * 4 + threadIdx.x; 
    unsigned int k = i + offset; 

    if(k < n){
        C[i] = A[k]+B[k];
    }
    if(k + blockDim.x < n){
        C[i+blockDim.x] = A[k+blockDim.x] + B[k+blockDim.x];
    }
    if(k + blockDim.x*2 < n){
        C[i+blockDim.x*2] = A[k+blockDim.x*2] + B[k+blockDim.x*2];
    }
    if(k + blockDim.x*3 < n){
        C[i+blockDim.x*3] = A[k+blockDim.x*3] + B[k+blockDim.x*3];
    }
}
```
### 4.3.5.2 增大并行性
修改核函数的执行配置

# 4.4 核函数可达到的带宽
一个原本不好的访问模式，仍然可以通过重新设计核函数中的几个部分以实现良好的性能

## 4.4.1 内存带宽
我们一般有两种类型的带宽： 
1. 理论带宽，当前硬件可以实现的绝对最大带宽
2. 有效带宽，即 (读字节数+写字节数)*10^9 / 运行时间

## 4.4.2 矩阵转置问题
一段cpp代码示例为
```cpp
void transposeHost(float *out, float *in, const int nx, const int ny){
    for(int iy=0; iy<ny; ++iy){
        for(int ix=0; ix<nx; ++ix){
            out[ix*ny+iy] = in[iy*nx+ix];
        }
    }
}
```

读：对原矩阵按行读取，是合并访问
写：对转置矩阵的列访问，是交叉访问

我们会考虑两种方法：即按行读取按列存储，按列读取按行存储

如果禁用一级缓存，这两种实现的性能理论上是一致的。如果开启一级缓存，则第二种方法可能会更好（因为按列读取是**不合并的**，开启一级缓存后，下次读取可能直接缓存上执行）

### 4.4.2.1 为转置核函数设置性能的上限和下限
我们使用矩阵拷贝来作为一个性能的上下限，其中上限就是让矩阵按行拷贝，下限就是让矩阵按列拷贝

行拷贝实现如下
```cpp
__global__ void copyRow(float *out, float *in, const int nx, const int ny){
    unsigned int ix = blockDim.x*blockIdx.x + threadIdx.x; 
    unsigned int iy = blockDim.y*blockIdx.y + threadIdx.y; 
    if(ix < nx && iy < ny){
        out[iy*nx + ix] = in[iy*nx + ix];
    }
}
```
列拷贝如下
```cpp
__global__ void copyCol(float *out, float *in, const int nx, const int ny){
    unsigned int ix = blockDim.x*blockIdx.x + threadIdx.x; 
    unsigned int iy = blockDim.y*blockIdx.y + threadIdx.y;

    if(ix < nx && iy < ny){
        out[ix*ny+iy] = in[ix*ny + iy];
    }
}
```

### 4.4.2.2 朴素转置：读取行与读取列
```cpp
__global__ void transposeNaiveRow(float *out, float *in, const int nx, const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x; 
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y; 

    if(ix < nx && iy < ny){
        out[ix*ny + iy] = in[iy*nx + ix];
    }
}
```
我们互换一下索引，就有读取列与存储行的转置形式
```cpp
__global__ void transposeNaiveCol(float *out, float *in, const int nx, const int ny){
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x; 
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y; 

    if(ix < nx && iy < ny){
        out[iy*nx + ix] = in[ix*ny + iy];
    }
}
```
### 4.4.2.3 展开转置：读取行与读取列
接下来我们利用展开技术来提高转置内存带宽的利用率

下面代码展示的是读取行，存储列的形式
```cpp
__global__ void transposeUnroll4Row(float *out, flaot *in, const int nx, const int ny){
    unsigned int ix = 4* blockDim.x * blockIdx.x + threadIdx.x; 
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y; 

    unsigned int ti = iy * nx + ix; 
    unsigned int to = ix * ny + iy; 

    if (ix+3*blockDim.x < nx && iy < ny){
        out[to] = in[ti]; 
        out[to + blockDim.x*ny] = in[ti + blockDim.x]; 
        out[to + 2*blockDim.x*ny] = in[ti + 2*blockDim.x]; 
        out[to + 2*blockDim.x*ny] = in[ti + 2*blockDim.x]; 
    }
}
```
下面代码展示的是读取列，存储行的形式
```cpp
__global__ void transposeUnroll4Col(float *out, float *in, const int nx, const int ny){
    unsigned int ix = 4* blockDim.x * blockIdx.x + threadIdx.x; 
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y; 

    unsigned int ti = iy*nx + ix;
    unsigned int to = ix*ny + iy;

    if(ix + 3*blockDim.x < nx && iy < ny){
        out[ti] = in[to];
        out[ti + blockDim.x] = in[to + blockDim.x*ny];
        out[ti + 2*blockDim.x] = in[to + 2*blockDim.x*ny];
        out[ti + 2*blockDim.x] = in[to + 3*blockDim.x*ny];
    }
}
```

### 4.4.2.4 对角转置
虽然CUDA编程模型可以对网格抽象为一维和二维，但本质上都是一维。每个线程块的唯一bid可以计算为
```cpp
unsigned int bid = gridDim.x*blockIdx.y + blockIdx.x;
```

对角坐标和笛卡尔坐标的转换为：
```cpp
block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
block_y = blockIdx.x
```
其中blockIdx.x和blockIdx.y为对角坐标，block_x和block_y是笛卡尔坐标。（因为数据读取，最后我们还是要转换回来），使用映射回来的笛卡尔坐标来计算线程索引
```cpp
__global__ void transposeDiagonalRow(float *out, float *in, const int nx, const int ny){
    unsigned int blk_y = blockIdx.x; 
    unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x; 

    unsigned int ix = blockDim.x * blk_x + threadIdx.x; 
    unsigned int iy = blockDim.y * blk_y + threadIdx.y; 

    if(ix < nx && iy < ny){
        out[ix*ny + iy] = in[iy*nx + ix];
    }
}
```
同理换一下坐标就得到了基于列的对角坐标核函数

性能提升的原因与**DRAM的并行访问有关**，当使用笛卡尔坐标将线程块映射到数据块时，全局内存访问可能无法均匀到整个DRAM的 从分区 中，这时可能会出现**分区冲突**的现象。
而使用对角坐标后，由于是非线性映射，所以交叉访问不太可能会落入到一个独立的分区中。

### 4.4.2.5 使用瘦块来增加并行性
基于列的方法，可以使用`瘦块`来存储在线程块中连续元素的数量，提高存储操作的效率

# 4.5 使用统一内存的矩阵加法
我们可以使用统一内存来简化矩阵加法的代码

```cpp
float *A, *B, *gpuRef; 
cudaMallocManaged((void**) &A, nBytes);
cudaMallocManaged((void**) &B, nBytes);
cudaMallocManaged((void**) &gpuRef, nBytes);
initialData(A, nxy); 
initialData(B, nxy); 
sumMatrixGPU<<<grid, block>>>(A, B, gpuRef, nx, ny);
cudaDeviceSynchronize();
```

这里笔者在2080Ti测试与书中结果相反，个人猜测统一内存的效率与架构强相关（书中使用的是kepler架构）

