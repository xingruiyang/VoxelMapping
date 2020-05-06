#include <assert.h>
#include <pangolin/pangolin.h>
#include "../util/DatasetLoader.h"
#include "../include/Voxelization.h"
#include "../include/ImageProc.h"

int main(int argc, char **argv)
{
    assert(argc == 2);
    voxelization::util::DatasetLoader loader(argv[1]);
    loader.loadImages(true);
    Eigen::Matrix3f K = loader.loadCalibration();
    loader.loadGroundTruth();

    int w = 640;
    int h = 480;
    voxelization::Voxelization map(w, h, K.cast<double>());

    cv::Mat depth, color;
    double time;
    Sophus::SE3d gt_pose;
    int idx = 0;
    while (loader.GetNext(depth, color, time, gt_pose))
    {
        printf("Processing frame %d\n", idx++);
        cv::Mat depth_float;
        depth.convertTo(depth_float, CV_32FC1, 1 / 5000.0);
        map.FuseImage(cv::cuda::GpuMat(depth_float), cv::cuda::GpuMat(color), gt_pose);
    }

    cv::cuda::GpuMat vmap, nmap, image;
    map.RenderScene(vmap, gt_pose);
    voxelization::computeNormal(vmap, nmap);
    voxelization::renderScene(vmap, nmap, image);
    cv::Mat image_cpu(image);
    cv::imshow("image", image_cpu);
    cv::waitKey(0);

    while (!pangolin::ShouldQuit())
    {
        pangolin::CreateWindowAndBind("Preview");
    }
}