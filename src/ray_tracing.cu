#include "cuda_utils.h"
#include "device_functions.h"
#include "map_struct.h"

#define RenderingBlockSizeX 16
#define RenderingBlockSizeY 16
#define RenderingBlockSubSample 8
#define MaxNumRenderingBlock 1000000

namespace vmap
{

struct RenderingBlockDelegate
{
    Eigen::Transform<float, 3, Eigen::Affine> Tinv;
    int width, height;
    float fx, fy, cx, cy;
    float depthMin, depthMax;
    float voxelSize;

    uint* rendering_block_count;
    uint visible_block_count;

    HashEntry* visibleEntry;
    RenderingBlock* listRenderingBlock;

    mutable cv::cuda::PtrStepSz<float> zRangeX;
    mutable cv::cuda::PtrStep<float> zRangeY;

    __device__ __forceinline__ bool createRenderingBlock(const Eigen::Vector3i& blockPos, RenderingBlock& block) const
    {
        block.upper_left = Eigen::Matrix<short, 2, 1>(zRangeX.cols, zRangeX.rows);
        block.lower_right = Eigen::Matrix<short, 2, 1>(-1, -1);
        block.zrange = Eigen::Vector2f(depthMax, depthMin);

        float scale = voxelSize * BLOCK_SIZE;
#pragma unroll
        for (int corner = 0; corner < 8; ++corner)
        {
            Eigen::Vector3i tmp = blockPos;
            tmp(0) += (corner & 1) ? 1 : 0;
            tmp(1) += (corner & 2) ? 1 : 0;
            tmp(2) += (corner & 4) ? 1 : 0;

            auto pt3d = Tinv * (tmp.cast<float>() * scale);
            Eigen::Vector2f pt2d = project(pt3d, fx, fy, cx, cy) / RenderingBlockSubSample;

            if (block.upper_left(0) > std::floor(pt2d(0)))
                block.upper_left(0) = (int)std::floor(pt2d(0));
            if (block.lower_right(0) < ceil(pt2d(0)))
                block.lower_right(0) = (int)ceil(pt2d(0));
            if (block.upper_left(1) > std::floor(pt2d(1)))
                block.upper_left(1) = (int)std::floor(pt2d(1));
            if (block.lower_right(1) < ceil(pt2d(1)))
                block.lower_right(1) = (int)ceil(pt2d(1));
            if (block.zrange(0) > pt3d(2))
                block.zrange(0) = pt3d(2);
            if (block.zrange(1) < pt3d(2))
                block.zrange(1) = pt3d(2);
        }

        if (block.upper_left(0) < 0)
            block.upper_left(0) = 0;
        if (block.upper_left(1) < 0)
            block.upper_left(1) = 0;
        if (block.lower_right(0) >= zRangeX.cols)
            block.lower_right(0) = zRangeX.cols - 1;
        if (block.lower_right(1) >= zRangeX.rows)
            block.lower_right(1) = zRangeX.rows - 1;
        if (block.upper_left(0) > block.lower_right(0))
            return false;
        if (block.upper_left(1) > block.lower_right(1))
            return false;
        if (block.zrange(0) < depthMin)
            block.zrange(0) = depthMin;
        if (block.zrange(1) < depthMin)
            return false;

        return true;
    }

    __device__ __forceinline__ void splitRenderingBlock(int offset, const RenderingBlock& block, int& nx, int& ny) const
    {
        for (int y = 0; y < ny; ++y)
        {
            for (int x = 0; x < nx; ++x)
            {
                if (offset < MaxNumRenderingBlock)
                {
                    RenderingBlock& b(listRenderingBlock[offset++]);
                    b.upper_left(0) = block.upper_left(0) + x * RenderingBlockSizeX;
                    b.upper_left(1) = block.upper_left(1) + y * RenderingBlockSizeY;
                    b.lower_right(0) = block.upper_left(0) + (x + 1) * RenderingBlockSizeX;
                    b.lower_right(1) = block.upper_left(1) + (y + 1) * RenderingBlockSizeY;
                    if (b.lower_right(0) > block.lower_right(0))
                        b.lower_right(0) = block.lower_right(0);
                    if (b.lower_right(1) > block.lower_right(1))
                        b.lower_right(1) = block.lower_right(1);
                    b.zrange = block.zrange;
                }
            }
        }
    }

    __device__ __forceinline__ void operator()() const
    {
        int x = threadIdx.x + blockDim.x * blockIdx.x;

        bool valid = false;
        uint requiredNoBlocks = 0;
        RenderingBlock block;
        int nx, ny;

        if (x < visible_block_count && visibleEntry[x].ptr != -1)
        {
            valid = createRenderingBlock(visibleEntry[x].pos, block);
            float dx = (float)block.lower_right(0) - block.upper_left(0) + 1;
            float dy = (float)block.lower_right(1) - block.upper_left(1) + 1;
            nx = __float2int_ru(dx / RenderingBlockSizeX);
            ny = __float2int_ru(dy / RenderingBlockSizeY);

            if (valid)
            {
                requiredNoBlocks = nx * ny;
                uint totalNoBlocks = *rendering_block_count + requiredNoBlocks;
                if (totalNoBlocks >= MaxNumRenderingBlock)
                {
                    requiredNoBlocks = 0;
                }
            }
        }

        int offset = PrefixSum<1024>(requiredNoBlocks, rendering_block_count);
        if (valid && offset != -1 && (offset + requiredNoBlocks) < MaxNumRenderingBlock)
            splitRenderingBlock(offset, block, nx, ny);
    }
};

struct FillRenderingBlockFunctor
{
    mutable cv::cuda::PtrStepSz<float> zRangeX;
    mutable cv::cuda::PtrStep<float> zRangeY;
    RenderingBlock* listRenderingBlock;

    __device__ __forceinline__ void operator()() const
    {
        int x = threadIdx.x;
        int y = threadIdx.y;

        int block = blockIdx.x * 4 + blockIdx.y;
        if (block >= MaxNumRenderingBlock)
            return;

        RenderingBlock& b(listRenderingBlock[block]);

        int xpos = b.upper_left(0) + x;
        if (xpos > b.lower_right(0) || xpos >= zRangeX.cols)
            return;

        int ypos = b.upper_left(1) + y;
        if (ypos > b.lower_right(1) || ypos >= zRangeX.rows)
            return;

        atomicMin(&zRangeX.ptr(ypos)[xpos], b.zrange(0));
        atomicMax(&zRangeY.ptr(ypos)[xpos], b.zrange(1));

        return;
    }
};

template <class TVoxel>
void ProjectRenderingBlocks(
    MapStruct<TVoxel> map,
    uint count_visible_block,
    uint& count_rendering_block,
    cv::cuda::GpuMat& zRangeX,
    cv::cuda::GpuMat& zRangeY,
    RenderingBlock* listRenderingBlock,
    const Eigen::Matrix4f& worldToCam,
    const Eigen::Matrix3f& K)
{
    if (count_visible_block == 0)
        return;

    const int cols = zRangeX.cols;
    const int rows = zRangeY.rows;

    zRangeX.setTo(cv::Scalar(100.f));
    zRangeY.setTo(cv::Scalar(0));

    uint* count_device;
    count_rendering_block = 0;
    cudaMalloc((void**)&count_device, sizeof(uint));
    cudaMemset((void*)count_device, 0, sizeof(uint));

    RenderingBlockDelegate delegate;

    delegate.width = cols;
    delegate.height = rows;
    delegate.Tinv = worldToCam;
    delegate.zRangeX = zRangeX;
    delegate.zRangeY = zRangeY;
    delegate.fx = K(0, 0);
    delegate.fy = K(1, 1);
    delegate.cx = K(0, 2);
    delegate.cy = K(1, 2);
    delegate.visibleEntry = map.visibleTable;
    delegate.visible_block_count = count_visible_block;
    delegate.rendering_block_count = count_device;
    delegate.listRenderingBlock = listRenderingBlock;
    delegate.depthMax = MAX_DEPTH;
    delegate.depthMin = MIN_DEPTH;
    delegate.voxelSize = map.voxelSize;

    dim3 thread = dim3(1024);
    dim3 block = dim3(cv::divUp((size_t)count_visible_block, thread.x));

    callDeviceFunctor<<<block, thread>>>(delegate);

    (cudaMemcpy(&count_rendering_block, count_device, sizeof(uint), cudaMemcpyDeviceToHost));
    if (count_rendering_block == 0)
        return;

    thread = dim3(RenderingBlockSizeX, RenderingBlockSizeY);
    block = dim3((uint)ceil((float)count_rendering_block / 4), 4);

    FillRenderingBlockFunctor functor;
    functor.listRenderingBlock = listRenderingBlock;
    functor.zRangeX = zRangeX;
    functor.zRangeY = zRangeY;

    callDeviceFunctor<<<block, thread>>>(functor);
    (cudaFree((void*)count_device));
}

template void ProjectRenderingBlocks<Voxel>(
    MapStruct<Voxel> map,
    uint count_visible_block,
    uint& count_rendering_block,
    cv::cuda::GpuMat& zRangeX,
    cv::cuda::GpuMat& zRangeY,
    RenderingBlock* listRenderingBlock,
    const Eigen::Matrix4f& worldToCam,
    const Eigen::Matrix3f& K);
template void ProjectRenderingBlocks<VoxelRGB>(
    MapStruct<VoxelRGB> map,
    uint count_visible_block,
    uint& count_rendering_block,
    cv::cuda::GpuMat& zRangeX,
    cv::cuda::GpuMat& zRangeY,
    RenderingBlock* listRenderingBlock,
    const Eigen::Matrix4f& worldToCam,
    const Eigen::Matrix3f& K);

template <class TVoxel>
struct MapRenderingDelegate
{
    int width, height;
    // MapStruct map;
    mutable cv::cuda::PtrStep<Eigen::Vector4f> vert_map;
    cv::cuda::PtrStepSz<float> zRangeX;
    cv::cuda::PtrStepSz<float> zRangeY;
    float invfx, invfy, cx, cy;
    Eigen::Transform<float, 3, Eigen::Affine> pose, Tinv;

    HashEntry* hashTable;
    TVoxel* listBlock;
    int nBucket;
    float voxelSizeInv;
    float raytraceStep;

    __device__ __forceinline__ float read_sdf(const Eigen::Vector3f& pt3d, bool& valid) const
    {
        TVoxel* voxel = NULL;
        findVoxel(floor(pt3d), voxel, hashTable, listBlock, nBucket);
        if (voxel && voxel->wt != 0)
        {
            valid = true;
            return unpackFloat(voxel->sdf);
        }
        else
        {
            valid = false;
            return 1.0;
        }
    }

    __device__ __forceinline__ float read_sdf_interped(const Eigen::Vector3f& pt, bool& valid) const
    {
        // Eigen::Vector3f xyz = pt - floor(pt);
        Eigen::Vector3f xyz = Eigen::Vector3f(pt(0) - ::floor(pt(0)), pt(1) - ::floor(pt(1)), pt(2) - ::floor(pt(2)));
        float sdf[2], result[4];
        bool valid_pt;

        sdf[0] = read_sdf(pt, valid_pt);
        sdf[1] = read_sdf(pt + Eigen::Vector3f(1, 0, 0), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];

        sdf[0] = read_sdf(pt + Eigen::Vector3f(0, 1, 0), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Eigen::Vector3f(1, 1, 0), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];
        result[2] = (1.0f - xyz(1)) * result[0] + xyz(1) * result[1];

        sdf[0] = read_sdf(pt + Eigen::Vector3f(0, 0, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Eigen::Vector3f(1, 0, 1), valid);
        valid_pt &= valid;
        result[0] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];

        sdf[0] = read_sdf(pt + Eigen::Vector3f(0, 1, 1), valid);
        valid_pt &= valid;
        sdf[1] = read_sdf(pt + Eigen::Vector3f(1, 1, 1), valid);
        valid_pt &= valid;
        result[1] = (1.0f - xyz(0)) * sdf[0] + xyz(0) * sdf[1];
        result[3] = (1.0f - xyz(1)) * result[0] + xyz(1) * result[1];
        valid = valid_pt;
        return (1.0f - xyz(2)) * result[2] + xyz(2) * result[3];
    }

    __device__ __forceinline__ void operator()() const
    {
        const int x = threadIdx.x + blockDim.x * blockIdx.x;
        const int y = threadIdx.y + blockDim.y * blockIdx.y;
        if (x >= width || y >= height)
            return;

        vert_map.ptr(y)[x](0) = __int_as_float(0x7fffffff);

        int u = __float2int_rd((float)x / 8);
        int v = __float2int_rd((float)y / 8);

        auto zNear = zRangeX.ptr(v)[u];
        auto zFar = zRangeY.ptr(v)[u];
        if (zNear < FLT_EPSILON || zFar < FLT_EPSILON ||
            isnan(zNear) || isnan(zFar))
            return;

        float sdf = 1.0f;
        float lastReadSDF;

        Eigen::Vector3f pt = unproject(x, y, zNear, invfx, invfy, cx, cy);
        float dist_s = pt.norm() * voxelSizeInv;
        Eigen::Vector3f blockStart = pose * (pt)*voxelSizeInv;

        pt = unproject(x, y, zFar, invfx, invfy, cx, cy);
        float distEnd = pt.norm() * voxelSizeInv;
        Eigen::Vector3f blockEnd = pose * (pt)*voxelSizeInv;

        Eigen::Vector3f dir = (blockEnd - blockStart).normalized();
        Eigen::Vector3f result = blockStart;

        bool sdfValid = false;
        bool ptFound = false;
        float step;

        while (dist_s < distEnd)
        {
            lastReadSDF = sdf;
            sdf = read_sdf(result, sdfValid);

            if (sdf <= 0.5f && sdf >= -0.5f)
                sdf = read_sdf_interped(result, sdfValid);
            if (sdf <= 0.0f)
                break;
            if (sdf >= 0.f && lastReadSDF < 0.f)
                return;
            if (sdfValid)
                step = max(sdf * raytraceStep, 1.0f);
            else
                step = 1;

            result += step * dir;
            dist_s += step;
        }

        if (sdf <= 0.0f)
        {
            step = sdf * raytraceStep;
            result += step * dir;

            sdf = read_sdf_interped(result, sdfValid);

            step = sdf * raytraceStep;
            result += step * dir;

            if (sdfValid)
                ptFound = true;
        }

        if (ptFound)
        {
            result = Tinv * (result / voxelSizeInv);
            vert_map.ptr(y)[x] = Eigen::Vector4f(result(0), result(1), result(2), 1.0f);
        }
    }
};

template <class TVoxel>
void RenderScene(
    MapStruct<TVoxel> map, cv::cuda::GpuMat vert_map,
    cv::cuda::GpuMat zRangeX, cv::cuda::GpuMat zRangeY,
    const Eigen::Matrix4f& camToWorld, const Eigen::Matrix3f& K)
{
    const int cols = vert_map.cols;
    const int rows = vert_map.rows;

    MapRenderingDelegate<TVoxel> delegate;

    Eigen::Matrix4f worldToCam = camToWorld.inverse();

    delegate.width = cols;
    delegate.height = rows;
    delegate.vert_map = vert_map;
    delegate.zRangeX = zRangeX;
    delegate.zRangeY = zRangeY;
    delegate.invfx = 1.0 / K(0, 0);
    delegate.invfy = 1.0 / K(1, 1);
    delegate.cx = K(0, 2);
    delegate.cy = K(1, 2);
    delegate.pose = camToWorld;
    delegate.Tinv = worldToCam;
    delegate.hashTable = map.hashTable;
    delegate.listBlock = map.voxelBlock;
    delegate.nBucket = map.nBucket;
    delegate.voxelSizeInv = 1.0 / map.voxelSize;
    delegate.raytraceStep = map.truncationDist / map.voxelSize;

    dim3 thread(4, 8);
    dim3 block(cv::divUp(cols, thread.x), cv::divUp(rows, thread.y));

    callDeviceFunctor<<<block, thread>>>(delegate);
}

template void RenderScene<Voxel>(
    MapStruct<Voxel> map, cv::cuda::GpuMat vert_map,
    cv::cuda::GpuMat zRangeX, cv::cuda::GpuMat zRangeY,
    const Eigen::Matrix4f& camToWorld, const Eigen::Matrix3f& K);
template void RenderScene<VoxelRGB>(
    MapStruct<VoxelRGB> map, cv::cuda::GpuMat vert_map,
    cv::cuda::GpuMat zRangeX, cv::cuda::GpuMat zRangeY,
    const Eigen::Matrix4f& camToWorld, const Eigen::Matrix3f& K);

} // namespace vmap