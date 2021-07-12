#include "MapStruct.h"
#include "GlobalMapFuncs.h"
#include "Voxelization.h"
#include <cuda_runtime.h>

#define MAX_VERTS_BUFFER 10000000

namespace voxelization
{
  struct Voxelization::VoxelizationImpl
  {
    int width;
    int height;
    Eigen::Matrix3f mK;
    MapStruct deviceMap;

    // for map udate
    uint numVisBlock;

    // for raycast
    cv::cuda::GpuMat zRangeX;
    cv::cuda::GpuMat zRangeY;
    uint numRdBlocks;
    RenderingBlock *renderingBlocks;

    // for generating mesh
    float *verts_gpu, *verts_cpu;
    float *norms_gpu, *norms_cpu;

    inline VoxelizationImpl(int w, int h, const Eigen::Matrix3f &K) : width(w), height(h), mK(K)
    {
      deviceMap.create(250000, 180000, 200000, 0.01f, 0.03f);
      deviceMap.reset();
      zRangeX.create(h / 8, w / 8, CV_32FC1);
      zRangeY.create(h / 8, w / 8, CV_32FC1);

      cudaMalloc((void **)&renderingBlocks, sizeof(RenderingBlock) * 100000);
      cudaMalloc((void **)&verts_gpu, sizeof(float) * MAX_VERTS_BUFFER * 3);
      cudaMalloc((void **)&norms_gpu, sizeof(float) * MAX_VERTS_BUFFER * 3);

      verts_cpu = new float[MAX_VERTS_BUFFER * 3];
      norms_cpu = new float[MAX_VERTS_BUFFER * 3];
    }

    inline ~VoxelizationImpl()
    {
      deviceMap.release();

      if (renderingBlocks)
        cudaFree((void *)renderingBlocks);

      if (norms_gpu)
        cudaFree((void *)norms_gpu);
      if (verts_gpu)
        cudaFree((void *)verts_gpu);
      if (verts_gpu)
        cudaFree((void *)verts_gpu);
      if (verts_cpu)
        free((void *)verts_cpu);
      if (norms_cpu)
        free((void *)norms_cpu);
    }

    inline void fuse_depth(
        cv::cuda::GpuMat depth, const Eigen::Matrix4f &camToWorld)
    {
      numVisBlock = voxelization::FuseImage(deviceMap, depth, camToWorld, mK);
    }

    inline void render_scene(
        cv::cuda::GpuMat &vmap, const Eigen::Matrix4f &camToWorld)
    {
      if (numVisBlock == 0)
        return;

      if (vmap.empty())
        vmap.create(height, width, CV_32FC4);

      Eigen::Matrix4f camToWorldF = camToWorld.cast<float>();
      Eigen::Matrix4f worldToCamF = camToWorld.inverse().cast<float>();
      ProjectRenderingBlocks(
          deviceMap, numVisBlock, numRdBlocks,
          zRangeX, zRangeY, renderingBlocks,
          worldToCamF, mK);

      if (numRdBlocks != 0)
        voxelization::RenderScene(
            deviceMap, vmap, zRangeX, zRangeY, camToWorldF, mK);
    }

    inline void reset()
    {
      deviceMap.reset();
    }

    inline int Polygonize(float *&verts, float *&norms)
    {
      uint numTri = 0;
      voxelization::Polygonize(
          deviceMap, numVisBlock, numTri, verts_gpu, norms_gpu, MAX_VERTS_BUFFER);
      cudaMemcpy(verts_cpu, verts_gpu, sizeof(float) * numTri * 9, cudaMemcpyDeviceToHost);
      cudaMemcpy(norms_cpu, norms_gpu, sizeof(float) * numTri * 9, cudaMemcpyDeviceToHost);
      verts = verts_cpu;
      norms = norms_cpu;
      return numTri;
    }
  };

  Voxelization::Voxelization(int w, int h, const Eigen::Matrix3f &K)
      : impl(new VoxelizationImpl(w, h, K))
  {
  }

  Voxelization::~Voxelization() = default;

  void Voxelization::FuseDepth(cv::cuda::GpuMat depth, const Eigen::Matrix4f &camToWorld)
  {
    impl->fuse_depth(depth, camToWorld);
  }

  void Voxelization::RenderScene(cv::cuda::GpuMat &vmap, const Eigen::Matrix4f &camToWorld)
  {
    impl->render_scene(vmap, camToWorld);
  }

  void Voxelization::reset()
  {
    impl->reset();
  }

  int Voxelization::Polygonize(float *&vertex_out, float *&normal_out)
  {
    return impl->Polygonize(vertex_out, normal_out);
  }

} // namespace voxelization