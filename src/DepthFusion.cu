#include "GlobalFuncs.h"
#include "GlobalMapFuncs.h"
#include "MapStruct.h"
#include "MapStructFuncs.h"
#include "PrefixSum.h"

namespace voxelization
{

struct CreateBlockLineTracingFunctor
{
    int* heap;
    int* heapPtr;
    HashEntry* hashTable;
    int* bucketMutex;
    int* excessPtr;
    int nEntry;
    int nBucket;

    float voxelSize;
    float truncDistHalf;
    cv::cuda::PtrStepSz<float> depth;

    float invfx, invfy, cx, cy;
    float depthMin, depthMax;

    Eigen::Transform<float, 3, Eigen::Affine> T;

    __device__ __forceinline__ void allocateBlock(const Eigen::Vector3i& blockPos) const
    {
        createBlock(blockPos, heap, heapPtr, hashTable, bucketMutex, excessPtr, nEntry, nBucket);
    }

    __device__ __forceinline__ void operator()() const
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= depth.cols || y >= depth.rows)
            return;

        float dist = depth.ptr(y)[x];
        if (isnan(dist) || dist < depthMin || dist > depthMax)
            return;

        float distNear = max(depthMin, dist - truncDistHalf);
        float distFar = min(depthMax, dist + truncDistHalf);
        if (distNear >= distFar)
            return;

        Eigen::Vector3i blockStart = voxelPosToBlockPos(worldPtToVoxelPos(unprojectWorld(x, y, distNear, invfx, invfy, cx, cy, T), voxelSize));
        Eigen::Vector3i blockEnd = voxelPosToBlockPos(worldPtToVoxelPos(unprojectWorld(x, y, distFar, invfx, invfy, cx, cy, T), voxelSize));

        Eigen::Vector3i dir = blockEnd - blockStart;
        Eigen::Vector3i increment = Eigen::Vector3i(dir(0) < 0 ? -1 : 1, dir(1) < 0 ? -1 : 1, dir(2) < 0 ? -1 : 1);
        Eigen::Vector3i absIncrement = Eigen::Vector3i(abs(dir(0)), abs(dir(1)), abs(dir(2)));
        Eigen::Vector3i incrementErr = Eigen::Vector3i(absIncrement(0) << 1, absIncrement(1) << 1, absIncrement(2) << 1);

        int err1;
        int err2;

        // Bresenham's line algorithm
        // details see : https://en.m.wikipedia.org/wiki/Bresenham%27s_line_algorithm
        if ((absIncrement(0) >= absIncrement(1)) && (absIncrement(0) >= absIncrement(2)))
        {
            err1 = incrementErr(1) - 1;
            err2 = incrementErr(2) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < absIncrement(0); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(1) += increment(1);
                    err1 -= incrementErr(0);
                }

                if (err2 > 0)
                {
                    blockStart(2) += increment(2);
                    err2 -= incrementErr(0);
                }

                err1 += incrementErr(1);
                err2 += incrementErr(2);
                blockStart(0) += increment(0);
                allocateBlock(blockStart);
            }
        }
        else if ((absIncrement(1) >= absIncrement(0)) && (absIncrement(1) >= absIncrement(2)))
        {
            err1 = incrementErr(0) - 1;
            err2 = incrementErr(2) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < absIncrement(1); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(0) += increment(0);
                    err1 -= incrementErr(1);
                }

                if (err2 > 0)
                {
                    blockStart(2) += increment(2);
                    err2 -= incrementErr(1);
                }

                err1 += incrementErr(0);
                err2 += incrementErr(2);
                blockStart(1) += increment(1);
                allocateBlock(blockStart);
            }
        }
        else
        {
            err1 = incrementErr(1) - 1;
            err2 = incrementErr(0) - 1;
            allocateBlock(blockStart);
            for (int i = 0; i < absIncrement(2); ++i)
            {
                if (err1 > 0)
                {
                    blockStart(1) += increment(1);
                    err1 -= incrementErr(2);
                }

                if (err2 > 0)
                {
                    blockStart(0) += increment(0);
                    err2 -= incrementErr(2);
                }

                err1 += incrementErr(1);
                err2 += incrementErr(0);
                blockStart(2) += increment(2);
                allocateBlock(blockStart);
            }
        }
    }
};

struct CheckEntryVisibilityFunctor
{
    HashEntry* hashTable;
    HashEntry* visibleEntry;
    uint* visibleEntryCount;
    Eigen::Transform<float, 3, Eigen::Affine> Tinv;

    int* heap;
    int* heapPtr;
    Voxel* voxelBlock;
    int cols, rows;
    float fx, fy;
    float cx, cy;
    float depthMin;
    float depthMax;
    float voxelSize;
    int nEntry;
    int nVBlock;

    __device__ __forceinline__ void operator()() const
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;

        __shared__ bool needScan;

        if (threadIdx.x == 0)
            needScan = false;

        __syncthreads();

        uint increment = 0;
        if (idx < nEntry)
        {
            HashEntry* current = &hashTable[idx];
            if (current->ptr >= 0)
            {

                bool rval = checkBlockVisible(
                    current->pos,
                    Tinv,
                    voxelSize,
                    cols, rows,
                    fx, fy,
                    cx, cy,
                    depthMin,
                    depthMax);

                if (rval)
                {
                    needScan = true;
                    increment = 1;
                }
            }
        }

        __syncthreads();

        if (needScan)
        {
            auto offset = PrefixSum<1024>(increment, visibleEntryCount);
            if (offset >= 0 && offset < nEntry && idx < nEntry)
                visibleEntry[offset] = hashTable[idx];
        }
    }
};

struct DepthFusionFunctor
{
    Voxel* listBlock;
    HashEntry* visible_blocks;

    Eigen::Transform<float, 3, Eigen::Affine> Tinv;
    float fx, fy;
    float cx, cy;
    float depthMin;
    float depthMax;

    float truncationDist;
    int nEntry;
    float voxelSize;
    uint count_visible_block;

    cv::cuda::PtrStepSz<float> depth;

    __device__ __forceinline__ void operator()() const
    {
        if (blockIdx.x >= nEntry || blockIdx.x >= count_visible_block)
            return;

        HashEntry& current = visible_blocks[blockIdx.x];
        if (current.ptr == -1)
            return;

        Eigen::Vector3i voxelPos = blockPosToVoxelPos(current.pos);

#pragma unroll
        for (int blockIdxZ = 0; blockIdxZ < 8; ++blockIdxZ)
        {
            Eigen::Vector3i localPos = Eigen::Vector3i(threadIdx.x, threadIdx.y, blockIdxZ);
            Eigen::Vector3f pt = Tinv * voxelPosToWorldPt(voxelPos + localPos, voxelSize);

            int u = __float2int_rd(fx * pt(0) / pt(2) + cx + 0.5);
            int v = __float2int_rd(fy * pt(1) / pt(2) + cy + 0.5);
            if (u < 0 || v < 0 || u > depth.cols - 1 || v > depth.rows - 1)
                continue;

            float dist = depth.ptr(v)[u];
            if (isnan(dist) || dist > depthMax || dist < depthMin)
                continue;

            float sdf = dist - pt(2);
            if (sdf < -truncationDist)
                continue;

            sdf = fmin(1.0f, sdf / truncationDist);
            const int localIdx = localPosToLocalIdx(localPos);
            Voxel& voxel = listBlock[current.ptr + localIdx];

            auto oldSDF = unpackFloat(voxel.sdf);
            auto oldWT = voxel.wt;

            if (oldWT == 0)
            {
                voxel.sdf = packFloat(sdf);
                voxel.wt = 1;
                continue;
            }

            voxel.sdf = packFloat((oldSDF * oldWT + sdf * 1) / (oldWT + 1));
            voxel.wt = min(255, oldWT + 1);
        }
    }
};

int FuseImage(MapStruct map,
              const cv::cuda::GpuMat depth,
              const Eigen::Matrix4f& camToWorld,
              const Eigen::Matrix3f& K)
{
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float invfx = 1.0 / K(0, 0);
    float invfy = 1.0 / K(1, 1);
    const int cols = depth.cols;
    const int rows = depth.rows;

    dim3 thread(8, 8);
    dim3 block(cv::divUp(cols, thread.x), cv::divUp(rows, thread.y));

    CreateBlockLineTracingFunctor bfunctor;
    bfunctor.heap = map.heap;
    bfunctor.heapPtr = map.heapPtr;
    bfunctor.hashTable = map.hashTable;
    bfunctor.bucketMutex = map.bucketMutex;
    bfunctor.excessPtr = map.excessPtr;
    bfunctor.nEntry = map.nEntry;
    bfunctor.nBucket = map.nBucket;
    bfunctor.voxelSize = map.voxelSize;
    bfunctor.truncDistHalf = map.truncationDist * 0.5;
    bfunctor.depth = depth;
    bfunctor.invfx = invfx;
    bfunctor.invfy = invfy;
    bfunctor.cx = cx;
    bfunctor.cy = cy;
    bfunctor.depthMin = 0.3f;
    bfunctor.depthMax = 3.0f;
    bfunctor.T = camToWorld;

    callDeviceFunctor<<<block, thread>>>(bfunctor);
    map.resetVisibleBlockCount();

    CheckEntryVisibilityFunctor cfunctor;
    cfunctor.hashTable = map.hashTable;
    cfunctor.voxelBlock = map.voxelBlock;
    cfunctor.visibleEntry = map.visibleTable;
    cfunctor.visibleEntryCount = map.visibleBlockNum;
    cfunctor.heap = map.heap;
    cfunctor.heapPtr = map.heapPtr;
    cfunctor.nVBlock = map.nVBlock;
    cfunctor.Tinv = camToWorld.inverse().cast<float>();
    cfunctor.cols = cols;
    cfunctor.rows = rows;
    cfunctor.fx = fx;
    cfunctor.fy = fy;
    cfunctor.cx = cx;
    cfunctor.cy = cy;
    cfunctor.depthMin = 0.3f;
    cfunctor.depthMax = 3.0f;
    cfunctor.voxelSize = map.voxelSize;
    cfunctor.nEntry = map.nEntry;

    thread = dim3(1024);
    block = dim3(cv::divUp(map.nEntry, thread.x));

    callDeviceFunctor<<<block, thread>>>(cfunctor);

    uint visible_block_count = 0;
    map.getVisibleBlockCount(visible_block_count);
    if (visible_block_count == 0)
        return 0;

    DepthFusionFunctor functor;
    functor.listBlock = map.voxelBlock;
    functor.visible_blocks = map.visibleTable;
    functor.Tinv = camToWorld.inverse().cast<float>();
    functor.fx = fx;
    functor.fy = fy;
    functor.cx = cx;
    functor.cy = cy;
    functor.depthMin = 0.3f;
    functor.depthMax = 3.0f;
    functor.truncationDist = map.truncationDist;
    functor.nEntry = map.nEntry;
    functor.voxelSize = map.voxelSize;
    functor.count_visible_block = visible_block_count;
    functor.depth = depth;

    thread = dim3(8, 8);
    block = dim3(visible_block_count);

    callDeviceFunctor<<<block, thread>>>(functor);
    return visible_block_count;
}

} // namespace voxelization