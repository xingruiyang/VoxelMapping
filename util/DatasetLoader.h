#pragma once

#include <memory>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include "../Thirdparty/Sophus/sophus/se3.hpp"

namespace voxelization
{
namespace internal
{
struct DatasetLoaderImpl;
} // namespace internal

namespace util
{

class DatasetLoader
{
public:
    DatasetLoader(const std::string &filepath);
    void reset();
    bool loadImages(bool pathOnly);
    bool loadGroundTruth();
    Eigen::Matrix3f loadCalibration();
    Sophus::SE3d getFirstFramePose();
    bool GetNext(cv::Mat &depth, cv::Mat &color, double &time, Sophus::SE3d &camToWorld);

protected:
    std::shared_ptr<internal::DatasetLoaderImpl> impl;
};

} // namespace util
} // namespace voxelization
