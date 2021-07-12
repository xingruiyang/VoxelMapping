#include "MapStruct.h"
#include "GlobalFuncs.h"
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

    inline VoxelizationImpl(int w, int h, const Eigen::Matrix3f &K)
        : width(w), height(h), mK(K)
    {
    }

    inline ~VoxelizationImpl()
    {
      deviceMap.release();

      if (renderingBlocks)
        SafeCall(cudaFree((void *)renderingBlocks));

      if (norms_gpu)
        SafeCall(cudaFree((void *)norms_gpu));
      if (verts_gpu)
        SafeCall(cudaFree((void *)verts_gpu));
      if (verts_cpu)
        free((void *)verts_cpu);
      if (norms_cpu)
        free((void *)norms_cpu);
    }

    void create_map(int num_entry, int num_voxel, float voxel_size = 0.01f)
    {
      deviceMap.create(num_entry, int(num_voxel * 0.9), num_voxel, voxel_size, voxel_size * 4);
      deviceMap.reset();

      zRangeX.create(height / 8, width / 8, CV_32FC1);
      zRangeY.create(height / 8, width / 8, CV_32FC1);

      SafeCall(cudaMalloc((void **)&renderingBlocks, sizeof(RenderingBlock) * 100000));
      SafeCall(cudaMalloc((void **)&verts_gpu, sizeof(float) * MAX_VERTS_BUFFER * 3));
      SafeCall(cudaMalloc((void **)&norms_gpu, sizeof(float) * MAX_VERTS_BUFFER * 3));

      verts_cpu = new float[MAX_VERTS_BUFFER * 3];
      norms_cpu = new float[MAX_VERTS_BUFFER * 3];

      fprintf(stdout, "voxel map created\n");
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
      SafeCall(cudaMemcpy(verts_cpu, verts_gpu, sizeof(float) * numTri * 9, cudaMemcpyDeviceToHost));
      SafeCall(cudaMemcpy(norms_cpu, norms_gpu, sizeof(float) * numTri * 9, cudaMemcpyDeviceToHost));
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

  void Voxelization::CreateMap(int numEntries, int numVoxels, float voxelSize)
  {
    impl->create_map(numEntries, numVoxels, voxelSize);
  }

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