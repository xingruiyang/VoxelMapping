cmake_minimum_required(VERSION 3.0)
project(vmapping LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED)
find_package(OpenCV 3.4 REQUIRED)

add_library(${PROJECT_NAME} SHARED
    src/MapStruct.cu
    src/DepthFusion.cu
    src/RayTracing.cu
    src/ImageProc.cu
    src/Meshification.cu
    src/Voxelization.cpp
)

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/util>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
)

target_link_libraries(${PROJECT_NAME}
PUBLIC
    Eigen3::Eigen
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)

set_target_properties(${PROJECT_NAME}
PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 11
    CUDA_ARCHITECTURES 75
)

# add_subdirectory(example)

install(
TARGETS ${PROJECT_NAME}
EXPORT ${PROJECT_NAME}-export
LIBRARY DESTINATION lib
ARCHIVE DESTINATION lib
CONFIGURATIONS Release)

install(
EXPORT ${PROJECT_NAME}-export
FILE ${PROJECT_NAME}Config.cmake
NAMESPACE vmapping::
DESTINATION lib/cmake/${PROJECT_NAME})

install(
DIRECTORY "${CMAKE_SOURCE_DIR}/include/"
DESTINATION "include/${PROJECT_NAME}"
FILES_MATCHING
PATTERN "*.h")