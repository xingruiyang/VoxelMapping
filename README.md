# Online Voxel Map Generation And Rendering

![Screenshot1](example/1.png)
![Screenshot2](example/2.png)

## Dependencies

+ [OpenCV](https://github.com/opencv/opencv) >= 3.4
+ [Eigen3](https://github.com/eigenteam/eigen-git-mirror) >= 3.3
+ [CUDA](https://developer.nvidia.com/cuda-downloads) >= 10.0 (Although older versions might work)
+ Sophus (Included)
+ [Pangolin](https://github.com/stevenlovegrove/Pangolin) (Optional, only for visualization)

## Examples

To run the example, make sure you have Pangolin installed. The example should be automatically built during compilation. 

```DatasetLoader``` looks for three files underneath the root path of the dataset:
+ ```calibration.txt``` contains all neccessary intrinsic parameters needed for backprojecting depth. It shoud be a one liner with the format of ```fx fy cx cy```, separated by space.
+ ```associated.txt``` contains the list of color and depth images including their time stamps.
+ ```groundtruth.txt``` have the ground truth poses for all frames. This usually comes with some datasets but you could also provide your own (by running a SLAM system for example).

Run the example with the following format:

```bash
./PangolinDisplay3D <path-to-dataset>/
```