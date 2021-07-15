#ifndef VMAPPING_INCLUDE_DEVICE_FUNCTIONS_H
#define VMAPPING_INCLUDE_DEVICE_FUNCTIONS_H

#include "map_struct.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

namespace vmap
{

template <class TVoxel>
int FuseImage(
    MapStruct<TVoxel> map,
    const cv::cuda::GpuMat depth,
    const Eigen::Matrix4f& camToWorld,
    const Eigen::Matrix3f& K);

template <class TVoxel>
int FuseDepthAndImage(
    MapStruct<TVoxel> map,
    const cv::cuda::GpuMat image,
    const cv::cuda::GpuMat depth,
    const Eigen::Matrix4f& camToWorld,
    const Eigen::Matrix3f& K);

template <class TVoxel>
void ProjectRenderingBlocks(
    MapStruct<TVoxel> map,
    uint count_visible_block,
    uint& count_rendering_block,
    cv::cuda::GpuMat& zrange_x,
    cv::cuda::GpuMat& zrange_y,
    RenderingBlock* listRenderingBlock,
    const Eigen::Matrix4f& worldToCam,
    const Eigen::Matrix3f& K);

template <class TVoxel>
void RenderScene(
    MapStruct<TVoxel> map,
    cv::cuda::GpuMat vmap,
    cv::cuda::GpuMat zRangeX,
    cv::cuda::GpuMat zRangeY,
    const Eigen::Matrix4f& camToWorld,
    const Eigen::Matrix3f& K);

template <class TVoxel>
void Polygonize(
    MapStruct<TVoxel> map,
    uint& block_count,
    uint& triangle_count,
    void* vertex_out,
    void* normal_out,
    size_t bufferSize);

void GetSurfacePoints(
    void* vertex_in,  // GPU
    void* vertex_out, // GPU
    uint triangle_count);

template <class TVoxel>
void ReadSDFAndNeighbour(
    MapStruct<TVoxel> map,
    void* query_points_in,
    void* neighbours_out,
    void* sdf_out,
    void* validity_out);

} // namespace vmap

#endif