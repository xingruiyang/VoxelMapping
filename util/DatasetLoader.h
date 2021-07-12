#pragma once

#include <memory>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/opencv.hpp>

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
            Eigen::Matrix4d getFirstFramePose();
            bool GetNext(cv::Mat &depth, cv::Mat &color, double &time, Eigen::Matrix4d &camToWorld);

        protected:
            std::shared_ptr<internal::DatasetLoaderImpl> impl;
        };

    } // namespace util
} // namespace voxelization
