#pragma once
#include <cuda_runtime.h>

namespace vmap
{

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

} // namespace vmap