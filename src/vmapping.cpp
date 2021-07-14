#include "vmapping.h"
#include "cuda_utils.h"
#include "device_functions.h"
#include "map_struct.h"

#define MAX_VERTS_BUFFER 10000000

namespace vmap
{
struct VoxelMapping::VoxelizationImpl
{
    int width;
    int height;
    Eigen::Matrix3f mK;
    MapStruct<Voxel> deviceMap;

    // for map udate
    uint numVisBlock;

    // for raycast
    cv::cuda::GpuMat zRangeX;
    cv::cuda::GpuMat zRangeY;
    uint numRdBlocks;
    RenderingBlock* renderingBlocks;

    // for generating mesh
    float *verts_gpu, *verts_cpu;
    float *norms_gpu, *norms_cpu;
    float *points_gpu, *points_cpu;

    inline VoxelizationImpl(int w, int h, const Eigen::Matrix3f& K)
        : width(w), height(h), mK(K)
    {
    }

    inline ~VoxelizationImpl()
    {
        deviceMap.release();

        if (renderingBlocks)
            SafeCall(cudaFree((void*)renderingBlocks));

        if (norms_gpu)
            SafeCall(cudaFree((void*)norms_gpu));
        if (verts_gpu)
            SafeCall(cudaFree((void*)verts_gpu));
        if (points_gpu)
            SafeCall(cudaFree((void*)points_gpu));
        if (verts_cpu)
            free((void*)verts_cpu);
        if (norms_cpu)
            free((void*)norms_cpu);
        if (points_cpu)
            free((void*)points_cpu);
    }

    void create_map(int num_entry, int num_voxel, float voxel_size = 0.01f)
    {
        deviceMap.create(num_entry, int(num_voxel * 0.9), num_voxel, voxel_size, voxel_size * 4);
        deviceMap.reset();

        zRangeX.create(height / 8, width / 8, CV_32FC1);
        zRangeY.create(height / 8, width / 8, CV_32FC1);

        SafeCall(cudaMalloc((void**)&renderingBlocks, sizeof(RenderingBlock) * 100000));
        SafeCall(cudaMalloc((void**)&verts_gpu, sizeof(float) * MAX_VERTS_BUFFER * 3));
        SafeCall(cudaMalloc((void**)&norms_gpu, sizeof(float) * MAX_VERTS_BUFFER * 3));
        SafeCall(cudaMalloc((void**)&points_gpu, sizeof(float) * MAX_VERTS_BUFFER));

        verts_cpu = new float[MAX_VERTS_BUFFER * 3];
        norms_cpu = new float[MAX_VERTS_BUFFER * 3];
        points_cpu = new float[MAX_VERTS_BUFFER];

        fprintf(stdout, "voxel map created\n");
    }

    inline void fuse_depth(
        cv::cuda::GpuMat depth, const Eigen::Matrix4f& camToWorld)
    {
        numVisBlock = vmap::FuseImage(deviceMap, depth, camToWorld, mK);
    }

    inline void render_scene(
        cv::cuda::GpuMat& vmap, const Eigen::Matrix4f& camToWorld)
    {
        if (numVisBlock == 0)
            return;

        if (vmap.empty())
            vmap.create(height, width, CV_32FC4);

        Eigen::Matrix4f camToWorldF = camToWorld.cast<float>();
        Eigen::Matrix4f worldToCamF = camToWorld.inverse().cast<float>();

        ProjectRenderingBlocks(
            deviceMap, numVisBlock, numRdBlocks,
            zRangeX, zRangeY, renderingBlocks,
            worldToCamF, mK);

        if (numRdBlocks != 0)
            vmap::RenderScene(
                deviceMap, vmap, zRangeX, zRangeY, camToWorldF, mK);
    }

    inline void reset()
    {
        deviceMap.reset();
    }

    inline uint create_mesh()
    {
        uint num_triangles = 0;
        vmap::Polygonize(
            deviceMap, numVisBlock, num_triangles, verts_gpu, norms_gpu, MAX_VERTS_BUFFER);
        return num_triangles;
    }

    inline int get_verts_cpu(float*& verts, float*& norms)
    {
        uint num_triangles = create_mesh();
        if (num_triangles == 0)
        {
            return 0;
        }

        SafeCall(cudaMemcpy(verts_cpu, verts_gpu, sizeof(float) * num_triangles * 9, cudaMemcpyDeviceToHost));
        SafeCall(cudaMemcpy(norms_cpu, norms_gpu, sizeof(float) * num_triangles * 9, cudaMemcpyDeviceToHost));
        verts = verts_cpu;
        norms = norms_cpu;
        return num_triangles;
    }

    std::vector<Eigen::Vector3f> GetSurfacePoints()
    {
        uint num_triangles = create_mesh();
        if (num_triangles == 0)
        {
            return std::vector<Eigen::Vector3f>();
        }

        vmap::GetSurfacePoints(verts_gpu, points_gpu, num_triangles);
        SafeCall(cudaMemcpy(points_cpu, points_gpu, sizeof(float) * num_triangles * 3, cudaMemcpyDeviceToHost));
        return std::vector<Eigen::Vector3f>(
            static_cast<Eigen::Vector3f*>((void*)points_cpu),
            static_cast<Eigen::Vector3f*>((void*)points_cpu) + num_triangles);
    }
};

VoxelMapping::VoxelMapping(int w, int h, const Eigen::Matrix3f& K)
    : impl(new VoxelizationImpl(w, h, K))
{
}

VoxelMapping::~VoxelMapping() = default;

void VoxelMapping::CreateMap(int numEntries, int numVoxels, float voxelSize)
{
    impl->create_map(numEntries, numVoxels, voxelSize);
}

void VoxelMapping::FuseDepth(cv::cuda::GpuMat depth, const Eigen::Matrix4f& camToWorld)
{
    impl->fuse_depth(depth, camToWorld);
}

void VoxelMapping::FuseDepthAndImage(cv::Mat rgb, cv::Mat depth, const Eigen::Matrix4f& camToWorld)
{
    fprintf(stderr, "%s(%d) Not implemented yet\n", __FILE__, __LINE__);
}

void VoxelMapping::RenderScene(cv::cuda::GpuMat& vmap, const Eigen::Matrix4f& camToWorld)
{
    impl->render_scene(vmap, camToWorld);
}

void VoxelMapping::reset()
{
    impl->reset();
}

int VoxelMapping::Polygonize(float*& vertex_out, float*& normal_out)
{
    return impl->get_verts_cpu(vertex_out, normal_out);
}

std::vector<Eigen::Vector3f> VoxelMapping::GetSurfacePoints()
{
    return impl->GetSurfacePoints();
}

} // namespace vmap