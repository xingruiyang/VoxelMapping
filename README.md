# Online Voxel Map Generation And Rendering

## Dependencies

+ OpenCV >= 3.4
+ Eigen3 >= 3.3
+ CUDA >= 10.0 (Although older versions might work)
+ Sophus (Included)
+ Pangolin (Optional, only for visualization)

## Examples

To run the example, make sure you have Pangolin installed. The example should be automatically built during compilation. Create a ```calibration.txt``` file in your dataset's root directory if not already have one. The calibration file should only have one line containing all intrinsic parameters, e.g.:

```
fx fy cx cy
```

Run the example with the following format:

```bash
./PangolinDisplay3D <path-to-dataset>/
```