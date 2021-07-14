#include "GlobalFuncs.h"
#include "ImageProc.h"

namespace vmap
{

__device__ __forceinline__ Eigen::Matrix<uchar, 4, 1> renderPoint(const Eigen::Vector3f& point, const Eigen::Vector3f& normal,
                                                                  const Eigen::Vector3f& image, const Eigen::Vector3f& lightPos)
{
    Eigen::Vector3f colour(4.f / 255.f, 2.f / 255.f, 2.f / 255.f);
    if (!isnan(point(0)))
    {
        // ambient coeff
        const float Ka = 0.3f;
        // diffuse coeff
        const float Kd = 0.5f;
        // specular coeff
        const float Ks = 0.2f;
        // specular power
        const float n = 20.f;

        // ambient color
        const float Ax = image(0);
        // diffuse color
        const float Dx = image(1);
        // specular color
        const float Sx = image(2);
        // light color
        const float Lx = 1.f;

        Eigen::Vector3f L = (lightPos - point).normalized();
        Eigen::Vector3f V = (Eigen::Vector3f(0.f, 0.f, 0.f) - point).normalized();
        Eigen::Vector3f R = (2 * normal * (normal.dot(L)) - L).normalized();

        float Ix = Ax * Ka * Dx + Lx * Kd * Dx * fmax(0.f, (normal.dot(L))) + Lx * Ks * Sx * pow(fmax(0.f, (R.dot(V))), n);
        colour = Eigen::Vector3f(Ix, Ix, Ix);
    }

    return Eigen::Matrix<uchar, 4, 1>(static_cast<uchar>(__saturatef(colour(0)) * 255.f),
                                      static_cast<uchar>(__saturatef(colour(1)) * 255.f),
                                      static_cast<uchar>(__saturatef(colour(2)) * 255.f),
                                      255);
}

__global__ void renderSceneKernel(const cv::cuda::PtrStep<Eigen::Vector4f> vmap,
                                  const cv::cuda::PtrStep<Eigen::Vector4f> nmap,
                                  const Eigen::Vector3f lightPos,
                                  cv::cuda::PtrStepSz<Eigen::Matrix<uchar, 4, 1>> dst)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= dst.cols || y >= dst.rows)
        return;

    Eigen::Vector3f point = vmap.ptr(y)[x].head<3>();
    Eigen::Vector3f normal = nmap.ptr(y)[x].head<3>();
    Eigen::Vector3f pixel(1.f, 1.f, 1.f);

    dst.ptr(y)[x] = renderPoint(point, normal, pixel, lightPos);
}

void renderScene(const cv::cuda::GpuMat vmap, const cv::cuda::GpuMat nmap, cv::cuda::GpuMat& image)
{
    if (image.empty())
        image.create(vmap.size(), CV_8UC4);

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

    renderSceneKernel<<<grid, block>>>(vmap, nmap, Eigen::Vector3f(5, 5, 5), image);
}

__global__ void computeNormalKernel(cv::cuda::PtrStepSz<Eigen::Vector4f> vmap, cv::cuda::PtrStep<Eigen::Vector4f> nmap)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= vmap.cols - 1 || y >= vmap.rows - 1)
        return;

    int x10 = max(x - 1, 0);
    int x01 = min(x + 1, vmap.cols);
    int y10 = max(y - 1, 0);
    int y01 = min(y + 1, vmap.rows);

    Eigen::Vector3f v00 = vmap.ptr(y)[x10].head<3>();
    Eigen::Vector3f v01 = vmap.ptr(y)[x01].head<3>();
    Eigen::Vector3f v10 = vmap.ptr(y10)[x].head<3>();
    Eigen::Vector3f v11 = vmap.ptr(y01)[x].head<3>();

    nmap.ptr(y)[x].head<3>() = ((v01 - v00).cross(v11 - v10)).normalized();
    nmap.ptr(y)[x](3) = 1.f;
}

void computeNormal(const cv::cuda::GpuMat vmap, cv::cuda::GpuMat& nmap)
{
    if (nmap.empty())
        nmap.create(vmap.size(), vmap.type());

    dim3 block(8, 8);
    dim3 grid(cv::divUp(vmap.cols, block.x), cv::divUp(vmap.rows, block.y));

    computeNormalKernel<<<grid, block>>>(vmap, nmap);
}

} // namespace vmap