在前面我们可以了解到，由于GPU硬件存在一级缓存，所以非对齐的内存访问对于性能影响较小。但是CUDA实现的算法和程序中，我们无法避免非合并访问，我们可以通过使用共享内存来提高全局内存合并访问的可能。

# 5.1 CUDA共享内存概述
GPU有两种类型的内存
- 板载内存（如全局内存，有较高的延时）
- 片上内存（如共享内存，延迟低，带宽高）

共享内存的用途一般有：
1. 块内线程通信的通道
2. 用于全局内存数据的可编程管理的缓存
3. 高速暂存存储器，用于转换数据以优化全局内存访问模式

## 5.1.1 共享内存
物理上，每个SM都有一个小的低延迟内存池，这个内存池被当前正在该SM上执行的线程块的线程所共享。

每个线程块开始执行时，会分配其一定数量的共享内存。共享内存的地址空间被线程块的所有线程共享，与所在线程块具有相同的**生命周期**，理想情况下，每个被线程束共享内存访问的请求在一个事务中完成，最差情况就是拆成32个事务顺序执行。

共享内存被SM中的所有常驻线程块划分，因此，共享内存是限制设备并行性的关键资源。**一个核函数使用的共享内存越多，处于并发活跃状态的线程块越少**

## 5.1.2 共享内存分配
共享内存变量用`__shared__`修饰符声明

共享内存数组可以动态声明，注意只能**动态声明一维数组**
```cpp
extern __shared__ int tile[];
```
启动核函数时，需要将所需共享内存大小作为括号的第三个参数
```cpp
kernel<<<grid, block, isize*sizeof(int)>>>;
```

## 5.1.3 共享内存存储体和访问模式
### 5.1.3.1 内存存储体
共享内存被分为32（一个线程束所含的线程数量）个同样大小的内存模型，被称为**存储体**，可被同时访问。
如果通过线程束发布共享内存加载或存储操作，且在每个存储体上只访问不多于1个的内存地址，那么该操作可由一个内存事务完成。
### 5.1.3.2 存储体冲突
在共享内存中，**多个地址请求落在相同的内存存储体中**，就会发生存储体冲突

线程束发出共享内存请求时，有3种典型的模式：
1. 并行访问：多个地址访问多个存储体
2. 串行访问：多个地址访问单个存储体
3. 广播访问：单一地址读取单一存储体

并行访问较为常见，最佳情况是，**每个地址都位于一个单独的存储体，执行无冲突的共享内存访问**

串行访问时最坏的情况，**线程束中32个线程访问同一存储体中不同的内存地址**，需要32个内存事务

广播访问下，线程束所有线程读取同一存储体下相同的地址。**虽然只需要一个内存事务，但是每次只有一小部分字节被读取，带宽利用率很差**

### 5.1.3.3 访问模式
存储体宽度规定了共享内存地址与共享内存存储体的关系。宽度随设备计算能力的不同而变化。

存储体索引计算方式（以存储体宽度为4为例子）
```
存储体索引 = (字节地址 / 4字节每存储体) % 32存储体
```

### 5.1.3.4 内存填充
我们可以通过内存填充，改变**字到存储体的映射**，进而避免存储体的冲突

用于填充的内存不能用于数据存储，其作用是**移动数据元素**。因此带来的缺点是线程块可用的总共享内存数量减少，还需要重新计算数组索引，以访问正确的元素。

### 5.1.3.5 访问模式配置
具体可参考书181页

更改共享内存存储体的大小不会增加共享内存的使用量。一个大的存储体可能为共享内存访问产生更高的带宽，但是可能会导致更多的存储体冲突。

### 5.1.4 配置共享内存量
每个SM都有64KB的片上内存，**共享内存和一级缓存共享该硬件资源**
我们可以
- 按设备进行配置
- 按核函数进行配置

我们可以用下面的运行时函数拟合对核函数配置一级缓存和共享内存大小
```cpp
cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache cacheConfig);
```
cacheConfig指明划分策略，包括如下： 
```cpp
cudaFuncCachePreferNone 表示默认配置
cudaFuncCachePreferShared 倾向分配48KB的shared memory和16KB的L1缓存
cudaFuncCachePreferL1 倾向分配48KB的L1缓存和16KB的shared memory
cudaFuncCachePreferEqual 倾向分配32KB的L1缓存和32KB的shared memory
```
CUDA运行时会尽可能使用请求设备的片上内存，如果需要执行一个核函数，**可自由地选择不同的配置**，参考以下CUDA运行时函数
```cpp
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig);
```

#### Tips 一级缓存与共享内存的区别
1. 共享内存通过32个存储体访问，一级缓存是通过缓存行进行访问
2. 共享内存对存储内容和存放位置有绝对的控制权，一级缓存的数据删除工作是由硬件完成的

## 5.1.5 同步
同步的两个基本方法
- 障碍：所有调用的线程等待其余调用的线程到达障碍点
- 内存栅栏：所有调用的线程必须等到全部内存修改对其余调用线程可见时才能继续执行

### 5.1.5.1 弱排序内存模型
GPU线程在不同内存中，写入数据的顺序，不一定和这些数据在源代码中访问的顺序相同。一个线程的写入顺序对其他线程可见时，可能和写操作被执行的实际顺序不一致。

为了强制程序以一个确切的顺序执行，我们需要插入内存栅栏和障碍

### 5.1.5.2 显示障碍
通过在代码中增加 `__syncthreads();`

需要注意的是在条件代码中使用`__syncthreads()`。如果只在某条件下执行，很可能让线程执行一直挂起。你需要对整个线程块的所有线程生效。一个反例是
```cpp
if(threadId % 2 == 0){
    __syncthreads();
}else{
    __syncthreads();
}
```

### 5.1.5.3 内存栅栏
内存栅栏确保栅栏前的任何内存写操作对栅栏后的其他线程都是可见的。可分为块，网格，系统的栅栏。

- `void __threadfence_block();`是线程块的栅栏，保证了栅栏前被调用和线程产生的对共享内存和全局内存的所有写操作对栅栏后同一块中的其他线程都是可见的。
- `void __threadfence();`创建网格级的栅栏，挂起调用的线程，直到全局内存中的所有写操作对相同网格内的所有线程是可见的。
- `void __threadfence_system();`创建跨系统的栅栏，挂起调用的线程，确保该线程对全局内存，锁页主机内存和其他设备内存中的所有写操作对全部设备中的线程和主机线程是可见的。

### 5.1.5.4 Volatile修饰符
修饰的变量可以防止编译器优化，就不会放置到缓存到寄存器或本地内存中。

# 5.2 共享内存的数据布局
## 5.2.1 方形共享内存
假设共享内存是一个方形数组，即
```cpp
__shared__ int tile[N][N];
```
现在我们以二维线程块去访问，方式有两种
```cpp
tile[threadIdx.y][threadIdx.x]; // 方式1
tile[threadIdx.x][threadIdx.y]; // 方式2
```
由于线程束中线程可由连续的threadIdx.x来确定。所以我们以threadIdx.x作为列索引，以访问不同的独立存储体，此时效率最高（即方式1）

### 5.2.1.1 行主序访问和列主序访问
考虑网格由1个二维线程块，每个行为都包含32个可用线程

我们执行两个操作
1. 将全局线程索引按行主序写入到一个二维共享内存数组中
2. 从共享内存中按行主序读取这些值，并存储到全局内存中

使用线程同步，保证所有线程都完成写入到共享内存数组任务后，再赋值

上面操作的核函数如下
```cpp
__global__ void setRowReadRow(int *out){
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    tile[threadIdx.y][threadIdx.x] = idx; 

    __syncthreads();

    out[idx] = tile[threadIdx.y][threadIdx.x];
}
```
当我们把threadIdx.y和threadIdx.x对换，就得到列主序写和读取
用
```shell
nvcc checkSmemSquare.cu -o checkSmemSquare
nvprof ./checkSmemSquare
```
可以看到行主序的写和读取，速度最快（参考上面的分析）

### 5.2.1.2 行主序写和列主序读
我们只需在给输出数组赋值，交换索引即可

### 5.2.1.3 动态共享内存
动态共享内存可以声明在核函数外/内，使其作用域分别是全局/局部

动态共享内存必须被声明为一个**未定大小的一维数组**

```cpp
__global__ void setRowReadColDyn(int *out){
    extern __shared__ int tile[];

    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;

    tile[row_idx] = row_idx;
    __syncthreads();
    out[row_idx] = tile[col_idx];
}
```

### 5.2.1.4 填充静态声明的共享内存
前面有提到过，增加一列，可以让列元素分布在不同的存储体内，避免冲突（针对Fermi设备，不同设备所需填充不同）
```cpp
__shared__ int tile[BDIMY][BDIMX+1];
```

### 5.2.1.5 填充动态声明的共享内存
我们计算索引的时候，每一行要多跳过一个填充的内存空间
```cpp
unsigned int row_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
unsigned int col_idx = threadIdx.x * (blockDim.x + 1) + threadIdx.y;
```

## 5.2.2 矩形共享内存
### 5.2.2.1 行主序访问与列主序访问
与前面的主要区别是，行主序的共享内存还是
```cpp
__shared__ int tile[BDIMY][BDIMX];
```
而列主序是
```cpp
__shared__ int tile[BDIMX][BDIMY];
```
这里我们是以一个16x32的矩阵为示例，而针对列操作，处理的是16个元素，在k40里，存储体宽度是8个字，因此，16x4/8，需要8个存储体，从而有8路冲突

### 5.2.2.2 行主序写操作和列主序读操作
我们实现一个矩阵转置操作
首先计算全局ID
```cpp
unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
```
此时我们的矩阵是矩形而不是前面的方形，转置过后我们需要重新计算坐标
```cpp
unsigned int irow = idx / blockDim.y;
unsigned int icol = idx % blockDim.y;
```
完整代码如下
```cpp
__global__ void setRowReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow];
}
```
### 5.2.2.3 动态声明的共享内存
因为动态声明的共享内存只能是一维的，所以我们在前面的基础上，还要再计算出一个一维索引
```cpp
unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
unsigned int irow = idx / blockDim.y;
unsigned int icol = idx % blockDim.y;
unsigned int col_idx = icol * blockDim.x + irow;
```
### 5.2.2.4 填充静态声明的共享内存
跟前面类似，这里不展开

### 5.2.2.5 填充动态声明的共享内存
跟前面类似，这里不展开

# 5.3 减少全局内存访问
## 5.3.1 使用共享内存的并行归约
我们以`reduceGmem`核函数作为基准性能，具体可参考`reduceInteger.cu`

接下来测试基于共享内存的核函数`reduceSmem`，主要区别是规约时候不用全局内存，而是将数据拷贝到共享内存

## 5.3.2 使用展开的并行归约
这里采取Unroll4，一次处理4个数据块

## 5.3.3 使用动态共享内存的并行归约
在Unroll函数中，用动态共享内存进行替代
```cpp
extern __shared__ int smem[];
```
运行速度上与静态声明的共享内存没什么差别

## 5.3.4 有效带宽
```
有效带宽 = （读字节+写字节） / （运行时间*10^9） GB/s
```

# 5.4 合并的全局内存访问
使用共享内存可以帮助避免对未合兵的全局内存访问

我们以矩阵转置为例，读操作是可以合并的，但是写操作是交叉访问。为了提高性能，我们可以在共享内存完成转置操作，然后再写到全局内存

## 5.4.1 基准转置内核
我们以最朴素的方式实现基准转置内核
```cpp
#define INDEX(ROW, COL, INNER) ((ROW) * (INNER) + (COL))
__global__ void copyGmem(float *out, float *in, const int nrows, const int ncols)
{
    // matrix coordinate (ix,iy)
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // transpose with boundary test
    if (row < nrows && col < ncols)
    {
		// NOTE this is a transpose, not a copy
        out[INDEX(col, row, nrows)] = in[INDEX(row, col, ncols)];
    }
}
```
为了测试性能上界，我们设置了一个基于行顺序的拷贝的核函数 `copyGmem`

## 5.4.2 使用共享内存的矩阵转置
首先读取全局内存的一行，写入到共享内存的一行，然后读取一列，写入到全局内存的一行。虽然会导致共享内存存储体冲突，但还是会比我们以全局内存的朴素实现要好的。

我这里的和书上提供的略微不一样
```cpp
__global__ void transposeSmem(float *out, float *in, int nrows, int ncols)
{
    __shared__ float tile[BDIMY][BDIMX];

    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;  // 计算全局的行，列坐标
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x; 

    unsigned int offset = INDEX(row, col, ncols); // 计算全局的一维行方向索引

    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x; // 计算一个线程块内的索引
    unsigned int irow = bidx / blockDim.y; // 转置，以3x4矩阵，编号为7的元素（从0开始），应在(1, 3)。这里转置后，变为4x3矩阵，第七个元素应该是(2, 1)。这里对应irow=2， icol=1
    unsigned int icol = bidx % blockDim.y;
    unsigned int transposed_offset = INDEX(col, row, nrows); // 转置后矩阵的偏移值
    if(row < nrows && col < ncols){
        tile[threadIdx.y][threadIdx.x] = in[offset];
    }
    __syncthreads();
    
    if(row < nrows && col < ncols){
        out[transposed_offset] = tile[irow][icol];
    }
}
```

## 5.4.3 使用填充共享内存的矩阵转置
```cpp
__shared__ float tile[BDIMY][BDIMX+2];
```

## 5.4.4 使用展开的矩阵转置
我们同时对两个数据块进行处理
```cpp
__global__ void transposeSmemUnroll(float *out, float *in, const int nrows, 
                                            const int ncols) 
{
    __shared__ float tile[BDIMY][BDIMX * 2];

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = (2 * blockIdx.x * blockDim.x) + threadIdx.x; // 一次性在行方向上操作两个数据块，这个是第一个数据块的列index

    unsigned int row2 = row;
    unsigned int col2 = col + blockDim.x; // 第二个数据块的列index

    // linear global memory index for original matrix
    unsigned int offset = INDEX(row, col, ncols); // 计算全局的index
    unsigned int offset2 = INDEX(row2, col2, ncols);

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    unsigned int transposed_offset = INDEX(col, row, nrows); // 计算转置之后的索引
    unsigned int transposed_offset2 = INDEX(col2, row2, nrows);

    if (row < nrows && col < ncols)
    {
        tile[threadIdx.y][threadIdx.x] = in[offset];
    }
    if (row2 < nrows && col2 < ncols)
    {
        tile[threadIdx.y][blockDim.x + threadIdx.x] = in[offset2];
    }

    __syncthreads();

    if (row < nrows && col < ncols)
    {
        out[transposed_offset] = tile[irow][icol];
    }
    if (row2 < nrows && col2 < ncols)
    {
        out[transposed_offset2] = tile[irow][blockDim.x + icol];
    }
}
```
# 5.5 常量内存
常量内存是一种用于只读数据和统一访问线程束中线程的数据。**常量内存对内核代码而言是只读的，但它对主机是即可读又可写的。**

常量内存在DRAM上，有一个专用的片上缓存，大小为64KB，读取延迟更低

**常量内存的最优访问模式是：线程束中的所有线程都访问相同的位置**

使用修饰符 `__constant__`

因为设备只能读取常量内存，所以常量内存的值需通过 `cudaMemcpyToSymbol` 进行初始化

## 5.5.1 使用常量内存实现一维模板
我们尝试实现一个
```
f(x) = c0(f(x+4h)-f(x-4h)) + c1(f(x+3h)-f(x-3h)) + c2(f(x+2h)-f(x-2h)) + c3(f(x+h)-f(x-h)) + 
```
考虑到每个线程都需要读取系数c0-c3，所以我们考虑把其放到常量内存

```cpp
__constant__ float coef[RADIUS+1];

// 定义系数
#define a0 0.0000f
...

// 初始化常量内存的数据
void setup_coef_constant(void){
    const float h_coef[] = {a0, a1, a2, a3, a4};
    CHECK(cudaMemcpyToSymbol(coef, h_coef, (RADIUS+1)*sizeof(float)));
}

__global__ void stencil_1d(float *in, float *out, int N){
    __shared__ float smem[BDIM+2*RADIUS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x; 

    while(idx < N){
        int sidx = threadIdx.x + RADIUS; 
        smem[sidx] = in[idx]; 
        if (threadIdx.x < RADIUS){
            // 前四个线程将RADIUS的数据读取到共享内存中
            smem[sidx - RADIUS] = in[idx - RADIUS]; // 这里传入in的地址时候，是d_in + RADIUS，因此不会越界
            smem[sidx + BDIM] = in[idx + BDIM];
        }

        __syncthreads();

        float tmp = 0.0f; 
        #pragma unroll // 提示编译器展开循环
        for(int i = 1; i <= RADIUS; i++){
            tmp += coef[i] * (smem[sidx + i] - smem[sidx - i]);
        }
        out[idx] = tmp; 
        idx += gridDim.x * blockDim.x; 
    }
}
```

## 5.5.2 与只读缓存的比较
kepler GPU使用GPU纹理流水线作为只读缓存，用于存储全局内存中的数据
每个kepler SM有48KB的只读缓存，它在分散读取下，比一级缓存更好，当线程束中的线程都读取相同地址时，不应使用只读缓存。（因为只读缓存的粒度是**32字节**）

有两种方法向编译器指出内核里，某些数据是只读的
1. 使用函数 `__ldg`(这是一个更好的选择)
它的作用相当于**标准的指针解引用**
```cpp
__global__ void kernel(float *output, float *in){
    output[idx] += __ldg(&in[idx]);
}
```
2. 使用全局内存的限定指针 `const __restrict__`
```cpp
void kernel(float* output, const float* __restrict__ input){
    ...
}
```

> 常量缓存加载的数据必须是相对较小的，而且访问必须一致以获得更好的性能
> 只读缓存加载的数据可以是比较大的，而且能够在一个非统一的模式下进行访问

tips: 
- 常量缓存和只读缓存都是**只读的**
- 每个SM上资源有限，常量缓存是64KB，只读缓存是48KB
- 常量缓存在统一读取中更好（统一读取是指线程束中每一个线程访问相同地址），只读缓存更适合分散读取

# 5.6 线程束洗牌指令
洗牌指令机制是指：**只要两个线程在相同的线程束中，那么就允许这两个线程直接读取另一个线程的寄存器**

这使得线程束中线程彼此可以直接交换数据，而不需要通过共享/全局内存，并且比共享内存有**更低的延迟**

我们可以通过公式
```cpp
laneID = threadIdx.x % 32 // 束内线程索引（即线程束内第几个线程）
warpID = threadIdx.x / 32 // 线程束索引（即第几个线程束）
```

## 5.6.1 线程束洗牌指令的不同形式
线程束洗牌指令基于整型和浮点数有两组，每组有四种指令

1. 在线程束内交换整型变量
```cpp
int __shfl(int var, int srcLane, int width=warpSize);
```
我们先看默认情况，即width=线程束大小32。
**此时srcLane表示束内线程索引**，会让各个线程接收到srcLane这个线程的var变量
```cpp
__shfl(var, 2); // 此时所有线程接收到2号线程的var变量
```
我们再看更复杂的情况，其中width大小必须是2的指数（从2到32）
**因为width可以小于32，所以会把线程束分成更小的单位————段**
此时srcLane是基于每个段的偏移索引
```cpp
__shfl(var, 3, 16);
```
以16的宽度，我们线程束可以分成两段。
其中第一段0-15线程，接受第3个线程的var变量（因为第一段起始为0，偏移3，0+3=3）

而第二段16-31线程，接受第19个线程的var变量（因为第二段起始为16，偏移3，16+3=19）

2. 在线程束内右移变量
```cpp
int __shfl_up(int var, unsigned int delta, int width=warpSize);
```
这里width和上面的是一样
以
```cpp
__shfl_up(val, 2);
```
为例子
它将0-29号线程的val变量右移2位(因为30，31号线程没地方移了)
即 0~29 -> 2~31
而 0，1号线程的变量保持不变

3. 在线程束内左移变量
```cpp
int __shfl_down(int var, unsigned int delta, int width=warpSize);
```
与上面类似，不展开，只是改变了移动的方向

4. 线程束内按位异或运算
```cpp
int __shfl_xnor(int var, int lanemask, int width=warpSize);
```
每个线程都与1异或
```cpp
000 ^ 001 = 001 // 0号线程去1号 
001 ^ 001 = 000 // 1号线程去0号 
010 ^ 001 = 011 // 2号线程去3号 
011 ^ 001 = 010 // 3号线程去2号 
```
```cpp
__shfl_xnor(val, 1); // 实现蝴蝶寻址模式
```

## 5.6.2 线程束内的共享数据
### 5.6.2.1 跨线程束值的广播
```cpp
// 具体可参考simpleShuffle.cu中的test_shfl_broadcast
```
我们传入`srcLane=2`，它将2号线程的变量广播到所有线程中
```
initialData             :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shfl bcast              :  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2  2 
```

### 5.6.2.2 线程束内上移
```cpp
// 具体可参考simpleShuffle.cu中的test_shfl_up

initialData             :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shfl up                 :  0  1  0  1  2  3  4  5  6  7  8  9 10 11 12 13 
```

### 5.6.2.3 线程束内下移
类似，这里不展开

### 5.6.2.4 线程束内环绕移动
我们通过设置`__shfl`指令中的srcLane为**当前线程ID+offset**进而实现环绕移动
```cpp
__global__ void test_shfl_wrap(int *d_out, int *d_in, int const offset){
    int value = d_in[threadIdx.x];
    value = __shfl(value, threadIdx.x+offset, BDIMX)
    d_out[threadIdx.x] = value;
}

initialData             :  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 
shfl wrap left          :  2  3  4  5  6  7  8  9 10 11 12 13 14 15  0  1 
```
### 5.6.2.5 跨线程束的蝴蝶交换
省略

### 5.6.2.6 跨线程束交换数组值
我们让每个线程管理一个寄存器数组，然后交换各个数组的值
```cpp
__global__ void test_shfl_xor_array(int *d_out, int *d_in, int const mask){
    int idx = threadIdx.x * SEGM; 
    int value[SEGM];

    for(int i = 0; i < SEGM; i++){
        value[i] = d_in[idx+i];
    }

    value[0] = __shfl_xor(value[0], mask, BDIMX);
    value[1] = __shfl_xor(value[1], mask, BDIMX);
    value[2] = __shfl_xor(value[2], mask, BDIMX);
    value[3] = __shfl_xor(value[3], mask, BDIMX);

    for(int i = 0; i < SEGM; i++){
        d_out[idx+i] = value[i];
    }
}

shfl array :  4  5  6  7  0  1  2  3 12 13 14 15  8  9 10 11 
```

### 5.6.2.7 跨线程束使用数组索引交换数值
我们尝试对每两个数组交换其首尾元素
```cpp
__global__ void test_shfl_xor_array_swap (int *d_out, int *d_in, int const mask,
    int srcIdx, int dstIdx)
{
    int idx = threadIdx.x * SEGM;
    int value[SEGM];

    for (int i = 0; i < SEGM; i++) value[i] = d_in[idx + i];

    bool pred = ((threadIdx.x & 1) != mask);

    if (pred)
    {
        int tmp = value[srcIdx];
        value[srcIdx] = value[dstIdx];
        value[dstIdx] = tmp;
    }

    value[dstIdx] = __shfl_xor (value[dstIdx], mask, BDIMX);

    if (pred)
    {
        int tmp = value[srcIdx];
        value[srcIdx] = value[dstIdx];
        value[dstIdx] = tmp;
    }

    for (int i = 0; i < SEGM; i++) d_out[idx + i] = value[i];
}
```
首先我们一共有4个线程，每个线程管理自己的一个大小为4的数组

然后分成两组，pred意思是，每一组的第一个线程为True

第一组
```
原型： 0 1 2 3  4 5 6 7 
针对第一个线程的数组，交换firstIdx和secondIdx(0, 3)：3 1 2 0  4 5 6 7
针对secondIdx做蝴蝶交换：3 1 2 7  4 5 6 0
再针对第一个线程的数组，交换firstIdx和secondIdx(0, 3) 7 1 2 3  4 5 6 0
```
对于第二组 `8 9 10 11 12 13 14 15`也是类似的流程

## 5.6.3 使用线程束洗牌指令的并行归约
归约分为三个层面
1. 线程束级
2. 线程块级
3. 网格级

针对线程束级别很简单，用到前面的蝴蝶交换
```cpp
__inline__ __device__ int warpReduce(int localSum)
{
    localSum += __shfl_xor(localSum, 16);
    localSum += __shfl_xor(localSum, 8);
    localSum += __shfl_xor(localSum, 4);
    localSum += __shfl_xor(localSum, 2);
    localSum += __shfl_xor(localSum, 1);

    return localSum;
}
```
然后保存到共享内存中
```cpp
int laneIdx = threadIdx.x % warpSize; 
int warpIdx = threadIdx.x / warpSize; 
mySum = warpReduce(mySum);
if(laneIdx == 0) smem[warpIdx] = mySum; // 让每个线程束的第一个线程保存值
```
对于线程块级归约，先取出每个线程束的总和，然后将总和相加
```cpp
__syncthreads(); // 同步块
mySum = (threadIdx.x < SMEMDIM) ? smem[laneIdx] : 0;
if (warpIdx == 0) localSum = warpReduce(localSum); 
```
对于网格级归约，将输出复制回主机
```cpp
if (threadIdx.x == 0) g_odata[blockIdx.x] = localSum;
```
具体可参考`reduceIntegerShfl.cu`
