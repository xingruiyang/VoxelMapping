#include "DatasetLoader.h"

namespace voxelization
{

struct DatasetLoader::DatasetLoaderImpl
{
    std::string rootPath;
    std::string calibPath;
    std::string groundTruthPath;
    std::string asscociationPath;

    std::vector<std::string> colorFilepath;
    std::vector<std::string> depthFilepath;
    std::vector<double> timeStamps;
    std::vector<Eigen::Matrix4d> groundTruth;

    int startPos, endPos, currPos;

    inline void clear()
    {
        colorFilepath.clear();
        depthFilepath.clear();
        timeStamps.clear();
        groundTruth.clear();
    }

    inline DatasetLoaderImpl(const std::string& filepath) : rootPath(filepath), startPos(0), endPos(0), currPos(0)
    {
        if (rootPath[rootPath.size() - 1] != '/')
            rootPath += '/';

        calibPath = rootPath + "calibration.txt";
        groundTruthPath = rootPath + "groundtruth.txt";
        asscociationPath = rootPath + "associated.txt";
    }

    inline void reset()
    {
    }

    inline bool loadImages(bool pathOnly)
    {
        std::ifstream asscociationFile(asscociationPath);
        assert(asscociationFile.is_open());

        while (!asscociationFile.eof())
        {
            std::string line;
            getline(asscociationFile, line);
            if (!line.empty() && line[0] != '#')
            {
                std::stringstream ss;
                ss << line;
                std::string colorPath, depthPath;
                double time;
                ss >> time;
                timeStamps.push_back(time);
                ss >> colorPath >> time >> depthPath;
                colorFilepath.push_back(rootPath + colorPath);
                depthFilepath.push_back(rootPath + depthPath);
            }
        }

        printf("%d Images loaded.\n", (int)timeStamps.size());
        currPos = 0;
        startPos = 0;
        endPos = (int)timeStamps.size();
        return true;
    }

    inline bool loadGroundTruth()
    {
        std::vector<std::pair<double, Eigen::Matrix4d>> allGroundTruth;
        std::ifstream groundTruthFile(groundTruthPath);
        assert(groundTruthFile.is_open());

        while (!groundTruthFile.eof())
        {
            std::string line;
            getline(groundTruthFile, line);
            if (!line.empty() && line[0] != '#')
            {
                double time, tx, ty, tz, qx, qy, qz, qw;
                std::stringstream ss;
                ss << line;
                ss >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
                Eigen::Vector3d transolation(tx, ty, tz);
                Eigen::Quaterniond rotation(qw, qx, qy, qz);
                rotation.normalize();
                Eigen::Matrix4d camToWorld;
                camToWorld.setIdentity();
                camToWorld.topLeftCorner(3, 3) = rotation.toRotationMatrix();
                camToWorld.topRightCorner(3, 1) = transolation;
                allGroundTruth.push_back(std::make_pair(time, camToWorld));
            }
        }

        if (allGroundTruth.size() == 0)
            return false;

        groundTruth.clear();
        for (double time : timeStamps)
        {
            int bestIdx = -1;
            double bestDiff = 1000;
            int idx = 0;

            for (auto& timePosePair : allGroundTruth)
            {
                double timeDiff = fabs(timePosePair.first - time);
                if (timeDiff < bestDiff)
                {
                    bestDiff = timeDiff;
                    bestIdx = idx;
                }
                idx++;
            }

            groundTruth.push_back(allGroundTruth[bestIdx].second);
        }

        return (groundTruth.size() == timeStamps.size());
    }

    inline bool GetNext(cv::Mat& depth, cv::Mat& color, double& time, Eigen::Matrix4d& camToWorld)
    {
        if (currPos == endPos)
            return false;

        depth = cv::imread(depthFilepath[currPos], -1);
        color = cv::imread(colorFilepath[currPos], -1);
        time = timeStamps[currPos];

        if (groundTruth.size() != 0)
            camToWorld = groundTruth[currPos];

        currPos++;
        return true;
    }

    inline Eigen::Matrix3f loadCalibration()
    {
        std::ifstream calibFile(calibPath);
        Eigen::Matrix3f K;

        std::string line;
        getline(calibFile, line);
        std::stringstream ss;
        ss << line;
        double fx, fy, cx, cy;
        ss >> fx >> fy >> cx >> cy;
        K.setIdentity();
        K(0, 0) = fx;
        K(1, 1) = fy;
        K(0, 2) = cx;
        K(1, 2) = cy;
        return K;
    }

    inline Eigen::Matrix4d getFirstFramePose()
    {
        return groundTruth[0];
    }
};

DatasetLoader::DatasetLoader(const std::string& filepath)
    : impl(new DatasetLoaderImpl(filepath))
{
}

DatasetLoader::~DatasetLoader() = default;

bool DatasetLoader::loadImages(bool pathOnly)
{
    return impl->loadImages(pathOnly);
}

Eigen::Matrix3f DatasetLoader::loadCalibration()
{
    return impl->loadCalibration();
}

Eigen::Matrix4d DatasetLoader::getFirstFramePose()
{
    return impl->getFirstFramePose();
}

bool DatasetLoader::loadGroundTruth()
{
    return impl->loadGroundTruth();
}

void DatasetLoader::reset()
{
    impl->reset();
}

bool DatasetLoader::GetNext(cv::Mat& depth, cv::Mat& color, double& time, Eigen::Matrix4d& camToWorld)
{
    return impl->GetNext(depth, color, time, camToWorld);
}

} // namespace voxelization