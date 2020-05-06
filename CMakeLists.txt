cmake_minimum_required(VERSION 3.4)
project(Voxelization LANGUAGES CXX CUDA)

Set(CMAKE_BUILD_TYPE Release)
# Set(CMAKE_BUILD_TYPE RelWithDebugInfo)

Set(CMAKE_CXX_STANDARD 11)
Set(CMAKE_CXX_STANDARD_REQUIRED ON)

Find_Package(Eigen3 3.3 REQUIRED)
Find_Package(OpenCV 3.4 REQUIRED)

Add_Library(${PROJECT_NAME} SHARED
    src/MapStruct.cu
    src/DepthFusion.cu
    src/RayTracing.cu
    src/ImageProc.cu
    src/Meshification.cu
    src/Voxelization.cpp
)

Target_Include_Directories(${PROJECT_NAME}
PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    $<INSTALL_INTERFACE:include/>
)

Target_Link_Libraries(${PROJECT_NAME}
PUBLIC
    Eigen3::Eigen
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)

Set_Target_Properties(${PROJECT_NAME}
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
)

Find_Package(Pangolin QUIET)
Message(CHECK_START "Looking for Pangolin")
if(${Pangolin_FOUND})
    Message(CHECK_PASS "Found, will build examples")
    add_executable(PangolinDisplay3D
        example/PangolinDisplay3D.cpp
        util/DatasetLoader.cpp
    )

    target_link_libraries(PangolinDisplay3D
    PUBLIC
        ${PROJECT_NAME}
        ${Pangolin_LIBRARIES}
    )
else()
    Message(CHECK_FAILED "Not Found")
endif()