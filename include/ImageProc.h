#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

namespace vmap
{

void renderScene(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, cv::cuda::GpuMat& image);
void computeNormal(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat& nmap);

} // namespace vmap
