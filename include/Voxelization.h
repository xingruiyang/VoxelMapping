#pragma once
#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "../Thirdparty/Sophus/sophus/se3.hpp"

namespace voxelization
{
namespace internal
{
struct VoxelizationImpl;
}

class Voxelization
{
public:
  Voxelization(int w, int h, const Eigen::Matrix3d &K);

  void reset();
  void FuseImage(cv::cuda::GpuMat depth, cv::cuda::GpuMat color, const Sophus::SE3d &camToWorld);
  void RenderScene(cv::cuda::GpuMat &vmap, const Sophus::SE3d &camToWorld);
  int Polygonize(void *vertex_out, void *normal_out);

protected:
  std::shared_ptr<internal::VoxelizationImpl> impl;
};

} // namespace voxelization