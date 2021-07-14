#ifndef EXAMPLE_DATASET_LOADER_H
#define EXAMPLE_DATASET_LOADER_H

#include <Eigen/Core>
#include <memory>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

namespace vmap
{

class DatasetLoader
{
public:
    DatasetLoader(const std::string& filepath);
    ~DatasetLoader();

    void reset();
    bool loadImages(bool pathOnly);
    bool loadGroundTruth();
    Eigen::Matrix3f loadCalibration();
    Eigen::Matrix4d getFirstFramePose();
    bool GetNext(cv::Mat& depth, cv::Mat& color, double& time, Eigen::Matrix4d& camToWorld);

protected:
    struct DatasetLoaderImpl;
    std::unique_ptr<DatasetLoaderImpl> impl;
};

} // namespace vmap

#endif