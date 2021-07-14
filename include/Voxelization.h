#pragma once
#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

namespace vmap
{
class Voxelization
{
public:
    Voxelization(int w, int h, const Eigen::Matrix3f& K);
    ~Voxelization();

    void reset();
    void CreateMap(int numEntries, int numVoxels, float voxelSize);
    void FuseDepth(cv::cuda::GpuMat depth, const Eigen::Matrix4f& camToWorld);
    void FuseDepthAndImage(cv::Mat rgb, cv::Mat depth, const Eigen::Matrix4f& camToWorld);
    void RenderScene(cv::cuda::GpuMat& vmap, const Eigen::Matrix4f& camToWorld);
    int Polygonize(float*& verts_out, float*& norms_out);
    std::vector<Eigen::Vector3f> GetSurfacePoints();

protected:
    struct VoxelizationImpl;
    std::unique_ptr<VoxelizationImpl> impl;
};

} // namespace vmap