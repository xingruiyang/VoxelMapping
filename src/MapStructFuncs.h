#pragma once
#include <sophus/se3.hpp>
#include <cuda_runtime_api.h>

namespace voxelization
{

    __device__ __forceinline__ float unpackFloat(short val)
    {
        return val / (float)SHRT_MAX;
    }

    __device__ __forceinline__ short packFloat(float val)
    {
        return (short)(val * SHRT_MAX);
    }

    __device__ __forceinline__ void atomicMax(float *add, float val)
    {
        int *address_as_i = (int *)add;
        int old = *address_as_i, assumed;
        do
        {
            assumed = old;
            old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }

    __device__ __forceinline__ void atomicMin(float *add, float val)
    {
        int *address_as_i = (int *)add;
        int old = *address_as_i, assumed;
        do
        {
            assumed = old;
            old = atomicCAS(address_as_i, assumed, __float_as_int(fminf(val, __int_as_float(assumed))));
        } while (assumed != old);
    }

    __device__ __forceinline__ int hash(const Eigen::Vector3i &pos, const int &noBuckets)
    {
        int res = (pos[0] * 73856093) ^ (pos[1] * 19349669) ^ (pos[2] * 83492791);
        res %= noBuckets;
        return res < 0 ? res + noBuckets : res;
    }

    __device__ __forceinline__ Eigen::Vector3i floor(const Eigen::Vector3f &pt)
    {
        return Eigen::Vector3i((int)::floor(pt[0]), (int)::floor(pt[1]), (int)::floor(pt[2]));
    }

    __device__ __forceinline__ Eigen::Vector3i worldPtToVoxelPos(Eigen::Vector3f pt, const float &voxelSize)
    {
        pt = pt / voxelSize;
        return floor(pt);
    }

    __device__ __forceinline__ Eigen::Vector3f voxelPosToWorldPt(const Eigen::Vector3i &voxelPos, const float &voxelSize)
    {
        Eigen::Vector3f worldPt;
        worldPt[0] = voxelPos[0] * voxelSize;
        worldPt[1] = voxelPos[1] * voxelSize;
        worldPt[2] = voxelPos[2] * voxelSize;
        return worldPt;
    }

    __device__ __forceinline__ Eigen::Vector3i voxelPosToBlockPos(Eigen::Vector3i voxelPos)
    {
        if (voxelPos(0) < 0)
            voxelPos(0) -= BLOCK_SIZE_M1;
        if (voxelPos(1) < 0)
            voxelPos(1) -= BLOCK_SIZE_M1;
        if (voxelPos(2) < 0)
            voxelPos(2) -= BLOCK_SIZE_M1;

        return voxelPos / BLOCK_SIZE;
    }

    __device__ __forceinline__ Eigen::Vector3i blockPosToVoxelPos(const Eigen::Vector3i &blockPos)
    {
        return blockPos * BLOCK_SIZE;
    }

    __device__ __forceinline__ Eigen::Vector3i voxelPosToLocalPos(Eigen::Vector3i voxelPos)
    {
        int x = voxelPos(0) % BLOCK_SIZE;
        int y = voxelPos(1) % BLOCK_SIZE;
        int z = voxelPos(2) % BLOCK_SIZE;

        if (x < 0)
            x += BLOCK_SIZE;
        if (y < 0)
            y += BLOCK_SIZE;
        if (z < 0)
            z += BLOCK_SIZE;

        return Eigen::Vector3i(x, y, z);
    }

    __device__ __forceinline__ int localPosToLocalIdx(const Eigen::Vector3i &localPos)
    {
        return localPos(2) * BLOCK_SIZE * BLOCK_SIZE + localPos(1) * BLOCK_SIZE + localPos(0);
    }

    __device__ __forceinline__ Eigen::Vector3i localIdxToLocalPos(const int &localIdx)
    {
        uint x = localIdx % BLOCK_SIZE;
        uint y = localIdx % (BLOCK_SIZE * BLOCK_SIZE) / BLOCK_SIZE;
        uint z = localIdx / (BLOCK_SIZE * BLOCK_SIZE);
        return Eigen::Vector3i(x, y, z);
    }

    __device__ __forceinline__ int voxelPosToLocalIdx(const Eigen::Vector3i &voxelPos)
    {
        return localPosToLocalIdx(voxelPosToLocalPos(voxelPos));
    }

    __device__ __forceinline__ Eigen::Vector2f project(const Eigen::Vector3f &pt,
                                                       const float &fx, const float &fy,
                                                       const float &cx, const float &cy)
    {
        return Eigen::Vector2f(fx * pt(0) / pt(2) + cx, fy * pt(1) / pt(2) + cy);
    }

    __device__ __forceinline__ Eigen::Vector3f unproject(const int &x, const int &y, const float &z,
                                                         const float &ifx, const float &ify,
                                                         const float &cx, const float &cy)
    {
        return Eigen::Vector3f(ifx * (x - cx) * z, ify * (y - cy) * z, z);
    }

    __device__ __forceinline__ Eigen::Vector3f unprojectWorld(const int &x, const int &y, const float &z,
                                                              const float &invfx, const float &invfy,
                                                              const float &cx, const float &cy,
                                                              const Eigen::Transform<float, 3, Eigen::Affine> &camToWorld)
    {
        return camToWorld * unproject(x, y, z, invfx, invfy, cx, cy);
    }

    __device__ __forceinline__ bool checkVertexVisible(const Eigen::Vector3f &worldPt,
                                                       const Eigen::Transform<float, 3, Eigen::Affine> &worldToCam,
                                                       const int &cols, const int &rows,
                                                       const float &fx, const float &fy,
                                                       const float &cx, const float &cy,
                                                       const float &depthMin,
                                                       const float &depthMax)
    {
        Eigen::Vector3f pt = worldToCam * worldPt;
        Eigen::Vector2f pixel = project(pt, fx, fy, cx, cy);
        return !(pixel[0] < 0 || pixel[1] < 0 ||
                 pixel[0] > cols - 1 || pixel[1] > rows - 1 ||
                 pt[2] < depthMin || pt[2] > depthMax);
    }

    __device__ __forceinline__ bool checkBlockVisible(const Eigen::Vector3i &blockPos,
                                                      const Eigen::Transform<float, 3, Eigen::Affine> &worldToCam,
                                                      const float &voxelSize,
                                                      const int &cols, const int &rows,
                                                      const float &fx, const float &fy,
                                                      const float &cx, const float &cy,
                                                      const float &depthMin,
                                                      const float &depthMax)
    {
        float scale = voxelSize * BLOCK_SIZE;
#pragma unroll
        for (int idx = 0; idx < 8; ++idx)
        {
            Eigen::Vector3i tmp = blockPos;
            tmp(0) += (idx & 1) ? 1 : 0;
            tmp(1) += (idx & 2) ? 1 : 0;
            tmp(2) += (idx & 4) ? 1 : 0;

            if (checkVertexVisible(tmp.cast<float>() * scale, worldToCam, cols, rows,
                                   fx, fy, cx, cy,
                                   depthMin, depthMax))
                return true;
        }

        return false;
    }

    __device__ __forceinline__ bool lockBucket(int *mutex)
    {
        if (atomicExch(mutex, 1) != 1)
            return true;
        else
            return false;
    }

    __device__ __forceinline__ void unlockBucket(int *mutex)
    {
        atomicExch(mutex, 0);
    }

    // __device__ __forceinline__ bool deleteHashEntry(int *heapPtr, int *heap, int nVBlock, HashEntry *entry)
    // {
    //     int val_old = atomicAdd(heapPtr, 1);
    //     if (val_old < nVBlock)
    //     {
    //         heap[val_old + 1] = entry->ptr / BLOCK_SIZE_3;
    //         entry->ptr = -1;
    //         return true;
    //     }
    //     else
    //     {
    //         atomicSub(heapPtr, 1);
    //         return false;
    //     }
    // }

    __device__ __forceinline__ bool AllocateHashEntry(int *heap, int *heapPtr, const Eigen::Vector3i &pos,
                                                      const int &offset, HashEntry *newEntry)
    {
        if (!newEntry)
            return false;

        int currPtr = atomicSub(heapPtr, 1);
        if (currPtr >= 0)
        {
            int ptr = heap[currPtr];
            if (ptr != -1)
            {
                newEntry->pos = pos;
                newEntry->ptr = ptr * BLOCK_SIZE_3;
                newEntry->offset = offset;
                return true;
            }
        }

        atomicAdd(heapPtr, 1);
        return false;
    }

    __device__ __forceinline__ void createBlock(const Eigen::Vector3i &blockPos, int *heap, int *heapPtr,
                                                HashEntry *hashTable, int *bucketMutex, int *linedListPtr,
                                                int numHashEntry, int numBucket)
    {
        auto volatileIdx = hash(blockPos, numBucket);
        int *mutex = &bucketMutex[volatileIdx];
        HashEntry *current = &hashTable[volatileIdx];
        HashEntry *emptyEntry = nullptr;
        if (current->pos == blockPos && current->ptr != -1)
            return;

        if (current->ptr == -1)
            emptyEntry = current;

        while (current->offset >= 0)
        {
            volatileIdx = numBucket + current->offset - 1;
            current = &hashTable[volatileIdx];
            if (current->pos == blockPos && current->ptr != -1)
                return;

            if (current->ptr == -1 && !emptyEntry)
                emptyEntry = current;
        }

        if (emptyEntry != nullptr)
        {
            if (lockBucket(mutex))
            {
                AllocateHashEntry(heap, heapPtr, blockPos, current->offset, emptyEntry);
                unlockBucket(mutex);
            }
        }
        else
        {
            if (lockBucket(mutex))
            {
                int offset = atomicAdd(linedListPtr, 1);
                if ((offset + numBucket) < numHashEntry)
                {
                    emptyEntry = &hashTable[numBucket + offset - 1];
                    if (AllocateHashEntry(heap, heapPtr, blockPos, -1, emptyEntry))
                        current->offset = offset;
                }
                else
                    atomicSub(linedListPtr, 1);

                unlockBucket(mutex);
            }
        }
    }

    __device__ __forceinline__ bool findEntry(const Eigen::Vector3i &blockPos, HashEntry *&out,
                                              HashEntry *hashTable, int nBucket)
    {
        uint volatileIdx = hash(blockPos, nBucket);
        out = &hashTable[volatileIdx];
        if (out->ptr != -1 && out->pos == blockPos)
            return true;

        while (out->offset >= 0)
        {
            volatileIdx = nBucket + out->offset - 1;
            out = &hashTable[volatileIdx];
            if (out->ptr != -1 && out->pos == blockPos)
                return true;
        }

        out = nullptr;
        return false;
    }

    __device__ __forceinline__ void findVoxel(const Eigen::Vector3i &voxelPos,
                                              Voxel *&out, HashEntry *hashTable,
                                              Voxel *listBlocks, int nBucket)
    {
        HashEntry *current;
        if (findEntry(voxelPosToBlockPos(voxelPos), current, hashTable, nBucket))
            out = &listBlocks[current->ptr + voxelPosToLocalIdx(voxelPos)];
    }

} // namespace voxelization