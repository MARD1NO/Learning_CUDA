# 7.1 CUDA指令概述
## 7.1.1 浮点指令
浮点数的标准规定，数据编码成三部分： 
- 符号段， 一个比特位
- 指数段， 多个比特位
- 尾数段， 多个比特位

对于一个`float32`变量，其存储分配分别是 `1, 8, 23`
而对于`double`变量，其存储分配分别是 `1, 11, 52`

浮点型数值不能精确存储，只能在四舍五入后再存储

随着浮点数值离零越来越远（在正负两个方向上），表示数值的区间也会随之增大

## 7.1.2 内部函数和标准函数
CUDA将所有算数函数分成内部函数和标准函数，标准函数**用于支持可对主机和设备进行访问并标准化主机和设备的操作**

CUDA内置函数**只能对设备代码进行访问**，编译时，会对其产生更积极的优化

CUDA中，许多内部函数与标准函数是有关联的，如标注拿回那束中，双精度浮点平方根函数为`sqrt`。对应相同功能的内部函数为`__dsqrt_rn`。还有执行单精度浮点除法运算的内部函数：`__fdividef`

内部函数将等价的标准函数**分解成更少的指令**，优点是内部函数运算更快，缺点是**数值精确度更低**

## 7.1.3 原子操作指令
原子操作执行一个数学运算，**阻止了多个线程之间互相干扰**，可以对跨线程共享数据进行`读-改-写`
我们以一个核函数（作用是递增）来作为例子
```cpp
__global__ void ince(int *ptr){
    int temp = *ptr; 
    temp = temp + 1; 
    *ptr = temp; 
}
```
此时我们没有指定具体的线程ID，所以所有线程会对同一个位置加1。这种多个线程对统一内存位置进行写操作，我们称为`数据竞争`，这是一种不安全的行为，我们不能得知确定的结果

我们可以使用原子操作指令来避免不安全行为的发生

原子操作分为三种：
- 算数运算函数
- 按位运算函数
- 替换函数。用一个新的值来替换内存位置上原有的值，可以是有条件的，也可以是无条件的。`atomicExch`可以无条件地替换已有的值。如果当前存储的值与由GPU线程调用指定的值相同，那么`atomicCAS`可以有条件地替换已有的值。**原子替换函数总是会返回最初存储在目标位置上的值**

使用原子函数改写的自增核函数如下
```cpp
__global__ void incr(__global__ int *ptr){
    int temp = atomicAdd(ptr, 1)
}
```
如果此时核函数使用的是32个线程，则会增加32

考虑另外一个核函数
```cpp
__global__ void check_threshold(int *arr, int threshold, int *flag){
    if (arr[blockIdx.x * blockDim.x + threadIdx.x] > threshold){
        *flag = 1;
    }
}
```
这里对flag的访问也是不安全的，我们可以使用`atomicExch`来无条件的将1替换
```cpp
__global__ void check_threshold(int *arr, int threshold, int *flag){
    if (arr[blockIdx.x*blockDim.x + threadIdx.x] > threshold){
        atomicExch(flag, 1);
    }
}
```
但其实不安全的访问（即线程访问前后顺序不定），不会改变这个核函数的目的（即设为1）。频繁使用原子操作还会造成性能下降
# 7.2 程序优化指令
## 7.2.1 单精度与双精度的比较
可参考`floating-point-accuracy.cu`和`floating-point-perf.cu`

- 单精度和双精度浮点计算在通信和计算上的性能差异是不可忽略的。双精度的计算和通信耗时几乎是单精度的两倍
- 单，双精度的结果有较大的数值差异

此外如果在核函数声明了双精度数值，则线程块总的共享寄存器区域会比使用浮点数小的多。 

在声明单精度浮点数时，如果错误地忽略了尾数f的声明(正确的应如`3.1415926f`)，nvcc编译器会自动转换成双精度数

## 7.2.2 标准函数与内部函数的比较
### 7.2.2.1 标准函数和内部函数优化
我们可以在编译的时候加入`--ptx`标志，让编译器在并行线程执行(PTX)和指令集架构(ISA)中生成程序的中间表达式。

在`foo.cu`写两个核函数
```cpp
__global__ void intrinsic(float *ptr){
    *ptr = __powf(*ptr, 2.0f);
}

__global__ void standard(float *ptr){
    *ptr = powf(*ptr, 2.0f);
}
```
然后生成一个ptx文件
```shell
nvcc --ptx foo.cu -o foo.ptx
```
其中`.entry`指令**代表一个函数的开始**
可以从ptx文件看到内部函数`__powf`生成的代码要比标准函数`powf`少很多

我们也可以利用`intrinsic-standard-comp.cu`代码来看到内部函数和标准函数在速度和精度上的差别

### 7.2.2.2 操纵指令生成
CUDA编译器中有两种方法可以控制指令级优化类型：编译器标志，内部或标准函数调用

比如`__fdividef`与运算符`/`相比，都是执行浮点数除法，但是速度更快，精确度较低

此外可以在编译的时候设置标志来达到优化的结果
如果想让乘和加合并为一个指令（Multiply add)，可以在编译的时候`--fmad=true`，但是是以数值精度作为代价

其他的编译器标志可以在`nvcc --help`找到，下面列举下相关标志
- `--ftz`将所有单精度非正规浮点数置为零（超出使用范围，不细究）
- `--prec-div`提高所有单精度除法和倒数数值的精度
- `--prec-sqrt`强制执行一个精度更高的平方根函数
- `--fmad`控制是否融合乘-加操作为一个指令
- `--use_fast_math`用等价的**内部函数**替换标准函数，并让`ftz=true`，`prec-div`和`prec-sqrt`为False

CUDA还包含了一对用于控制FMAD指令生成的内部函数：`__fmul`和`__dmul`，这两个分别对应单精度/双精度乘法。这些函数**不会影响**性能，在有`*`运算符地方调用它们的时候，会阻止nvcc将乘法作为乘加优化的一部分来使用。

在`--fmad=true/false`下都会阻止相关地方MAD指令的生成。你可以在某些地方提升数值精度，同时也能让MAD优化全局

一些浮点型内部函数(__fadd, __fsubm, __fmul)还可以使用两个后缀字符，表示`浮点四舍五入的模式`
- rn 当不能精确表示的浮点数值，会采用可表示的最近似值（默认）
- rz 向零取整
- ru 向上取整
- rd 向下取整

## 7.2.3 了解原子指令
### 7.2.3.1 从头开始
我们以`原子级比较并交换运算符CAS(compare and swap)`作为示例

CAS输入有：`内存地址，存储在此地址中的期望值，想要存储在该位置的新值`，然后执行：
1. 读取目标地址，并将该处地址的存储值与预期值进行比较
   - 如果存储值与预期值相等，那么新值将存入目标位置
   - 如果存储值与预期值不等，那么目标位置不会发生变化
2. 不论发生什么情况，一个CAS操作总是返回目标地址的值

CUDA提供的`atomicCAS`函数为`int atomicCAS(int *address, int compare, int val)`, compare是预期值，val是实际想写入的新值

我们利用该函数实现一个原子加法，自定义原子操作时，`定义目标的起始和结束状态是很有帮助的`

对于原子加法来说，`起始状态是基值，结束状态为起始状态+增量的总和`

对于原子加法，我们需要传入一个目的地址，和增量
```cpp
__device__ int myAtomicAdd(int *address, int incr){
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + incr);
    ...
}
```
但是由于目标位置是多线程共享访问，所以另外一个线程有可能修改address的值。我们加入一个while循环来确保最终修改成功
```cpp
__device__ int myAtomicAdd(int *address, int incr)
{
    // Create an initial guess for the value stored at *address.
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + incr);
    // Loop while the guess is incorrect.
    while (oldValue != guess)
    {
        guess = oldValue;
        oldValue = atomicCAS(address, guess, guess + incr);
    }
    return oldValue;
}
```
### 7.2.3.2 内置的CUDA原子函数
忽略，查官网文档
### 7.2.3.3 原子操作的成本
原子函数在一些应用很有帮助，但可能**付出很高的性能代价**，原因如下：
1. 当在全局或共享内存中执行原子操作时，能保证所有的数值变化对所有线程都是立即可见的。一个原子操作指令将通过任何方式进入到全局或共享内存中读取当前存储的数值，而不需要缓存
2. 前面atomicAdd示例也提到。共享地址冲突的访问，需要让线程不断尝试，直到是正确的值（即atomicAdd的while循环）。如果应用程序反复循环致使IO开销较大，性能会降低
3. 原子操作类似线程冲突的问题，一个线程束中只有一个线程的原子操作可以成功，其他线程必须重试。线程束中剩下的线程会等待所有原子操作完成，并且一个原子操作意味着一个全局的读写

**只有在保证正确性前提下，才可以使用不安全访问**
### 7.2.3.4 限制原子操作的性能成本
让各线程块产生一个中间结果，然后使用原子操作将局部结果结合乘最终的全局结果
### 7.2.3.5 原子级浮点支持
原子函数**大多被声明在整型数值上操作**。只有atomicExch和atomicAdd支持单精度浮点数。
如果涉及到浮点变量需要自己实现
以一个单精度浮点数实现的`my-AtomicAdd`核函数为例
```cpp
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
```
1. 使用强制转换，让address指针转换成uint类型
2. 使用`__float2uint_rn`将各浮点数转换为包含相同比特位的unsigned int类型
3. 返回结果转换为`float`类型，但是地址存的仍是int类型值 具体可参考`my-atomic-add-float.cu`（可能实现有误）