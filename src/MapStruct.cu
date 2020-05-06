#include "MapStruct.h"
#include "MapStructFuncs.h"
#include <opencv2/opencv.hpp>

namespace voxelization
{

__global__ void resetHashKernel(HashEntry *hashTable, int numEntry)
{
    int index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index >= numEntry)
        return;

    hashTable[index].ptr = -1;
    hashTable[index].offset = -1;
}

__global__ void resetHeapKernel(int *heap, int *heapPtr, int numBlock)
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
    dim3 grid(cv::divUp(hashTableSize, block.x));
    resetHashKernel<<<grid, block>>>(hashTable, hashTableSize);

    grid = dim3(cv::divUp(voxelBlockSize, block.x));
    resetHeapKernel<<<grid, block>>>(heap, heapPtr, voxelBlockSize);

    cudaMemset(excessPtr, 0, sizeof(int));
    cudaMemset(bucketMutex, 0, sizeof(int) * bucketSize);
    cudaMemset(voxelBlock, 0, sizeof(Voxel) * BLOCK_SIZE_3 * voxelBlockSize);
}

void MapStruct::create(
    int hashTableSize,
    int bucketSize,
    int voxelBlockSize,
    float voxelSize,
    float truncationDist)
{
    cudaMalloc((void **)&excessPtr, sizeof(int));
    cudaMalloc((void **)&heapPtr, sizeof(int));
    cudaMalloc((void **)&visibleBlockNum, sizeof(uint));
    cudaMalloc((void **)&bucketMutex, sizeof(int) * bucketSize);
    cudaMalloc((void **)&heap, sizeof(int) * voxelBlockSize);
    cudaMalloc((void **)&hashTable, sizeof(HashEntry) * hashTableSize);
    cudaMalloc((void **)&visibleTable, sizeof(HashEntry) * hashTableSize);
    cudaMalloc((void **)&voxelBlock, sizeof(Voxel) * voxelBlockSize * BLOCK_SIZE_3);

    this->hashTableSize = hashTableSize;
    this->bucketSize = bucketSize;
    this->voxelBlockSize = voxelBlockSize;
    this->voxelSize = voxelSize;
    this->truncationDist = truncationDist;
}

void MapStruct::release()
{
    cudaFree((void *)heap);
    cudaFree((void *)heapPtr);
    cudaFree((void *)hashTable);
    cudaFree((void *)bucketMutex);
    cudaFree((void *)excessPtr);
    cudaFree((void *)voxelBlock);
    cudaFree((void *)visibleBlockNum);
    cudaFree((void *)visibleTable);
}

void MapStruct::getVisibleBlockCount(uint &hostData)
{
    cudaMemcpy(&hostData, visibleBlockNum, sizeof(uint), cudaMemcpyDeviceToHost);
}

void MapStruct::resetVisibleBlockCount()
{
    cudaMemset(visibleBlockNum, 0, sizeof(uint));
}

bool MapStruct::empty()
{
    return bucketSize == 0;
}

} // namespace voxelization