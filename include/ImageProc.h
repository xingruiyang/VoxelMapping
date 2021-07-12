#pragma once

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

namespace voxelization
{

    void renderScene(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, cv::cuda::GpuMat &image);
    void computeNormal(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat &nmap);

} // namespace voxelization
