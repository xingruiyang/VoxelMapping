#include "cuda_utils.h"
#include "device_functions.h"
#include "map_struct.h"

namespace vmap
{

template <class TVoxel>
struct ReadSDFFunctor
{
    TVoxel* voxels;
    Eigen::Vector3f* query_points;
    Eigen::Vector3f* neighbours_pts;
    float* sdf_sample;
    bool* validity;
    uint num_points;

    __device__ __forceline__ void interp8(
        Eigen::Vector3f pt, Eigen::Vector3f* neighbours)
    {
        bool valid = false;
        sdf[0] = read_sdf(pos, valid);
        if (!valid) //|| fabs(sdf[0]) > 0.6)
            return false;

        sdf[1] = read_sdf(pos + Eigen::Vector3f(1, 0, 0), valid);
        if (!valid)
            return false;

        sdf[2] = read_sdf(pos + Eigen::Vector3f(1, 1, 0), valid);
        if (!valid)
            return false;

        sdf[3] = read_sdf(pos + Eigen::Vector3f(0, 1, 0), valid);
        if (!valid)
            return false;

        sdf[4] = read_sdf(pos + Eigen::Vector3f(0, 0, 1), valid);
        if (!valid)
            return false;

        sdf[5] = read_sdf(pos + Eigen::Vector3f(1, 0, 1), valid);
        if (!valid)
            return false;

        sdf[6] = read_sdf(pos + Eigen::Vector3f(1, 1, 1), valid);
        if (!valid)
            return false;

        sdf[7] = read_sdf(pos + Eigen::Vector3f(0, 1, 1), valid);
        if (!valid)
            return false;

        return true;
    }

    __device__ __forceinline__ void operator()()
    {
        uint idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx > num_points)
            return
    }
};

template <class TVoxel>
void ReadSDFAndNeighbour(
    MapStruct<TVoxel> map,
    void* query_points_in,
    void* neighbours_out,
    void* sdf_out,
    void* validity_out)
{
}

template void ReadSDFAndNeighbour<Voxel>(
    MapStruct<Voxel> map, void* query_points_in,
    void* neighbours_out, void* sdf_out, void* validity_out);
template void ReadSDFAndNeighbour<VoxelRGB>(
    MapStruct<VoxelRGB> map, void* query_points_in,
    void* neighbours_out, void* sdf_out, void* validity_out);

} // namespace vmap