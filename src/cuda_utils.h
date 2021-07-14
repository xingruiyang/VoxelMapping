#ifndef VMAPPING_INCLUDE_CUDA_UTILS_H
#define VMAPPING_INCLUDE_CUDA_UTILS_H

#include <cuda_runtime_api.h>
#include <iostream>

namespace vmap
{

#if defined(__GNUC__)
#define SafeCall(expr) ___SafeCall(expr, __FILE__, __LINE__, __func__)
#else
#define SafeCall(expr) ___SafeCall(expr, __FILE__, __LINE__)
#endif

static inline void error(const char* error_string, const char* file, const int line, const char* func)
{
    std::cout << "Error: " << error_string << "\t" << file << ":" << line << std::endl;
    exit(0);
}

static inline void ___SafeCall(cudaError_t err, const char* file, const int line, const char* func = "")
{
    if (cudaSuccess != err)
        error(cudaGetErrorString(err), file, line, func);
}

template <class T>
__global__ void callDeviceFunctor(const T functor)
{
    functor();
}

#ifdef __CUDACC__

template <int TB>
__device__ __forceinline__ int
PrefixSum(unsigned int element, unsigned int* sum)
{
    __shared__ unsigned int buffer[TB];
    __shared__ unsigned int blockOffset;

    if (threadIdx.x == 0)
        memset(buffer, 0, sizeof(unsigned int) * 16 * 16);
    __syncthreads();

    buffer[threadIdx.x] = element;
    __syncthreads();

    int s1, s2;

    for (s1 = 1, s2 = 1; s1 < TB; s1 <<= 1)
    {
        s2 |= s1;
        if ((threadIdx.x & s2) == s2)
            buffer[threadIdx.x] += buffer[threadIdx.x - s1];

        __syncthreads();
    }

    for (s1 >>= 2, s2 >>= 1; s1 >= 1; s1 >>= 1, s2 >>= 1)
    {
        if (threadIdx.x != TB - 1 && (threadIdx.x & s2) == s2)
            buffer[threadIdx.x + s1] += buffer[threadIdx.x];

        __syncthreads();
    }

    if (threadIdx.x == 0 && buffer[TB - 1] > 0)
        blockOffset = atomicAdd(sum, buffer[TB - 1]);

    __syncthreads();

    int offset;
    if (threadIdx.x == 0)
    {
        if (buffer[threadIdx.x] == 0)
            offset = -1;
        else
            offset = blockOffset;
    }
    else
    {
        if (buffer[threadIdx.x] == buffer[threadIdx.x - 1])
            offset = -1;
        else
            offset = blockOffset + buffer[threadIdx.x - 1];
    }

    return offset;
}

#endif

} // namespace vmap

#endif