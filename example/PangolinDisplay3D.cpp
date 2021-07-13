#include <assert.h>
#include <pangolin/pangolin.h>
// #include <pangolin/gl/glcuda.h>
// #include <pangolin/gl/glvbo.h>
#include "DatasetLoader.h"
#include "ImageProc.h"
#include "Voxelization.h"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        printf("Usage: %s <path-to-dataset>\n", argv[0]);
        exit(0);
    }

    voxelization::DatasetLoader loader(argv[1]);
    loader.loadImages(true);
    loader.loadGroundTruth();
    Eigen::Matrix3f K = loader.loadCalibration();
    Eigen::Matrix4d firstFramePose = loader.getFirstFramePose().inverse();

    int w = 640;
    int h = 480;
    voxelization::Voxelization map(w, h, K);

    cv::Mat depth, color;
    double time;
    Eigen::Matrix4d gt_pose;
    int idx = 0;
    while (loader.GetNext(depth, color, time, gt_pose))
    {
        printf("Processing frame %d\n", idx++);
        cv::Mat depth_float;
        depth.convertTo(depth_float, CV_32FC1, 1 / 5000.0);
        map.FuseDepth(cv::cuda::GpuMat(depth_float), gt_pose.cast<float>());
        break;
    };

    pangolin::CreateWindowAndBind("Preview");
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAt(0, 0, -2, 0, 0, 0, pangolin::AxisNegY));

    pangolin::View& d_cam = pangolin::Display("cam")
                                .SetBounds(0.0, 1.0, 0, 1.0, -640.0f / 480.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);

    const char vertShader[] =
        "#version 330\n"
        "\n"
        "layout(location = 0) in vec3 position;\n"
        "layout(location = 1) in vec3 a_normal;\n"
        "uniform mat4 modelMat;\n"
        "uniform mat4 mvpMat;\n"
        "out vec3 shaded_colour;\n"
        "\n"
        "void main(void) {\n"
        "    gl_Position = mvpMat * modelMat * vec4(position, 1.0);\n"
        "    vec3 lightpos = vec3(mvpMat[0][2], mvpMat[1][2], mvpMat[2][2]-3);\n"
        "    const float ka = 0.3;\n"
        "    const float kd = 0.5;\n"
        "    const float ks = 0.2;\n"
        "    const float n = 20.0;\n"
        "    const float ax = 1.0;\n"
        "    const float dx = 1.0;\n"
        "    const float sx = 1.0;\n"
        "    const float lx = 1.0;\n"
        "    vec3 L = normalize(lightpos - position);\n"
        "    vec3 V = normalize(vec3(0.0) - position);\n"
        "    vec3 R = normalize(2 * a_normal * dot(a_normal, L) - L);\n"
        "    float i1 = ax * ka * dx;\n"
        "    float i2 = lx * kd * dx * max(0.0, dot(a_normal, L));\n"
        "    float i3 = lx * ks * sx * pow(max(0.0, dot(R, V)), n);\n"
        "    float Ix = max(0.0, min(255.0, i1 + i2 + i3));\n"
        "    shaded_colour = vec3(Ix, Ix, Ix);\n"
        "}\n";

    const char fragShader[] =
        "#version 330\n"
        "\n"
        "in vec3 shaded_colour;\n"
        "out vec4 colour_out;\n"
        "void main(void) {\n"
        "    colour_out = vec4(shaded_colour, 1);\n"
        "}\n";

    pangolin::GlSlProgram program;
    program.AddShader(pangolin::GlSlShaderType::GlSlVertexShader, vertShader);
    program.AddShader(pangolin::GlSlShaderType::GlSlFragmentShader, fragShader);
    program.Link();

    // const size_t bufferSize = sizeof(float) * 3 * 20000000;

    pangolin::GlBuffer vertBuffer, normBuffer;
    // glGenBuffers(2, buffer);

    // pangolin::GlBufferCudaPtr vertexBuffer(
    //     pangolin::GlArrayBuffer, bufferSize, GL_FLOAT, 3,
    //     cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);
    // pangolin::GlBufferCudaPtr normalBuffer(
    //     pangolin::GlArrayBuffer, bufferSize, GL_UNSIGNED_BYTE, 3,
    //     cudaGraphicsMapFlagsWriteDiscard, GL_STREAM_DRAW);

    // pangolin::CudaScopedMappedPtr vertex_out(vertexBuffer);
    // pangolin::CudaScopedMappedPtr normal_out(normalBuffer);

    float *verts, *norms;
    int num_tri = map.Polygonize(verts, norms);

    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    vertBuffer.Bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    normBuffer.Bind();
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    normBuffer.Unbind();
    glBindVertexArray(0);

    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glColor4f(0.5f, 0.5f, 0.6f, 1.0f);

        d_cam.Activate(s_cam);
        program.Bind();
        glBindVertexArray(vao);
        pangolin::OpenGlMatrix modelMat(firstFramePose);
        program.SetUniform("modelMat", modelMat);
        program.SetUniform("mvpMat", s_cam.GetProjectionModelViewMatrix());
        glDrawArrays(GL_TRIANGLES, 0, num_tri * 3);
        glBindVertexArray(0);
        program.Unbind();

        pangolin::FinishFrame();
    }
}