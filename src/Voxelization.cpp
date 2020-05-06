#include "MapStruct.h"
#include "GlobalMapFuncs.h"
#include "Voxelization.h"
#include <cuda_runtime.h>

namespace voxelization
{
namespace internal
{
struct VoxelizationImpl
{
  int width;
  int height;
  Eigen::Matrix3d mK;
  MapStruct deviceMap;

  // for map udate
  uint numVisBlock;

  // for raycast
  cv::cuda::GpuMat zRangeX;
  cv::cuda::GpuMat zRangeY;
  uint numRdBlocks;
  RenderingBlock *renderingBlocks;

  inline VoxelizationImpl(int w, int h, const Eigen::Matrix3d &K) : width(w), height(h), mK(K)
  {
    deviceMap.create(800000, 600000, 650000, 0.005f, 0.03f);
    deviceMap.reset();
    zRangeX.create(h / 8, w / 8, CV_32FC1);
    zRangeY.create(h / 8, w / 8, CV_32FC1);
    cudaMalloc((void **)&renderingBlocks, sizeof(RenderingBlock) * 100000);
  }

  inline ~VoxelizationImpl()
  {
    deviceMap.release();
    cudaFree((void *)renderingBlocks);
  }

  inline void FuseImage(cv::cuda::GpuMat depth, cv::cuda::GpuMat color, const Sophus::SE3d &camToWorld)
  {
    numVisBlock = voxelization::FuseImage(deviceMap, depth, color, camToWorld, mK);
  }

  inline void RenderScene(cv::cuda::GpuMat &vmap, const Sophus::SE3d &camToWorld)
  {
    if (numVisBlock == 0)
      return;

    if (vmap.empty())
      vmap.create(height, width, CV_32FC4);

    Sophus::SE3f camToWorldF = camToWorld.cast<float>();
    Sophus::SE3f worldToCamF = camToWorld.inverse().cast<float>();
    ProjectRenderingBlocks(deviceMap, numVisBlock, numRdBlocks,
                           zRangeX, zRangeY, renderingBlocks, worldToCamF, mK);

    if (numRdBlocks != 0)
      voxelization::RenderScene(deviceMap, vmap, zRangeX, zRangeY, camToWorldF, mK);
  }

  inline void reset()
  {
    deviceMap.reset();
  }

  inline int Polygonize(void *vertex, void *normal)
  {
    uint numTri = 0;
    voxelization::Polygonize(deviceMap, numVisBlock,
                             numTri, vertex, normal);
    return numTri;
  }
};
} // namespace internal

Voxelization::Voxelization(int w, int h, const Eigen::Matrix3d &K) : impl(new internal::VoxelizationImpl(w, h, K))
{
}

void Voxelization::FuseImage(cv::cuda::GpuMat depth, cv::cuda::GpuMat color, const Sophus::SE3d &camToWorld)
{
  impl->FuseImage(depth, color, camToWorld);
}

void Voxelization::RenderScene(cv::cuda::GpuMat &vmap, const Sophus::SE3d &camToWorld)
{
  impl->RenderScene(vmap, camToWorld);
}

void Voxelization::reset()
{
  impl->reset();
}

int Voxelization::Polygonize(void *vertex_out, void *normal_out)
{
  return impl->Polygonize(vertex_out, normal_out);
}

} // namespace voxelization