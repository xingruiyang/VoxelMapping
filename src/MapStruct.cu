#include "GlobalFuncs.h"
#include "MapStruct.h"
#include "MapStructFuncs.h"
#include <opencv2/opencv.hpp>

namespace voxelization
{
__global__ void resetHashKernel(HashEntry* hashTable, int numEntry)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numEntry)
        return;

    hashTable[index].ptr = -1;
    hashTable[index].offset = -1;
}

__global__ void resetHeapKernel(int* heap, int* heapPtr, int numBlock)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numBlock)
        return;

    if (index == 0)
        heapPtr[0] = numBlock - 1;

    heap[index] = numBlock - index - 1;
}

void MapStruct::reset()
{
    dim3 block(1024);
    dim3 grid(cv::divUp(nEntry, block.x));
    resetHashKernel<<<grid, block>>>(hashTable, nEntry);

    grid = dim3(cv::divUp(nVBlock, block.x));
    resetHeapKernel<<<grid, block>>>(heap, heapPtr, nVBlock);

    SafeCall(cudaMemset(excessPtr, 0, sizeof(int)));
    SafeCall(cudaMemset(bucketMutex, 0, sizeof(int) * nBucket));
    SafeCall(cudaMemset(voxelBlock, 0, sizeof(Voxel) * BLOCK_SIZE_3 * nVBlock));
}

void MapStruct::create(
    int nEntry,
    int nBucket,
    int nVBlock,
    float voxelSize,
    float truncationDist)
{
    SafeCall(cudaMalloc((void**)&excessPtr, sizeof(int)));
    SafeCall(cudaMalloc((void**)&heapPtr, sizeof(int)));
    SafeCall(cudaMalloc((void**)&visibleBlockNum, sizeof(uint)));
    SafeCall(cudaMalloc((void**)&bucketMutex, sizeof(int) * nBucket));
    SafeCall(cudaMalloc((void**)&heap, sizeof(int) * nVBlock));
    SafeCall(cudaMalloc((void**)&hashTable, sizeof(HashEntry) * nEntry));
    SafeCall(cudaMalloc((void**)&visibleTable, sizeof(HashEntry) * nEntry));
    SafeCall(cudaMalloc((void**)&voxelBlock, sizeof(Voxel) * nVBlock * BLOCK_SIZE_3));

    this->nEntry = nEntry;
    this->nBucket = nBucket;
    this->nVBlock = nVBlock;
    this->voxelSize = voxelSize;
    this->truncationDist = truncationDist;
}

void MapStruct::release()
{
    SafeCall(cudaFree((void*)heap));
    SafeCall(cudaFree((void*)heapPtr));
    SafeCall(cudaFree((void*)hashTable));
    SafeCall(cudaFree((void*)bucketMutex));
    SafeCall(cudaFree((void*)excessPtr));
    SafeCall(cudaFree((void*)voxelBlock));
    SafeCall(cudaFree((void*)visibleBlockNum));
    SafeCall(cudaFree((void*)visibleTable));
}

void MapStruct::getVisibleBlockCount(uint& hostData)
{
    SafeCall(cudaMemcpy(&hostData, visibleBlockNum, sizeof(uint), cudaMemcpyDeviceToHost));
}

void MapStruct::resetVisibleBlockCount()
{
    SafeCall(cudaMemset(visibleBlockNum, 0, sizeof(uint)));
}

bool MapStruct::empty()
{
    return nBucket == 0;
}

} // namespace voxelization