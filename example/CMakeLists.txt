cmake_minimum_required(VERSION 3.0)
project(pangolin_example)

Find_Package(Pangolin QUIET)
message(CHECK_START "Looking for Pangolin")
if(${Pangolin_FOUND})
    message(CHECK_PASS "Found, will build examples")
    add_executable(pangolin_example
        pangolin_display_3d.cpp
        dataset_loader.cpp
    )

    target_link_libraries(pangolin_example
    PUBLIC
        vmapping
        ${Pangolin_LIBRARIES}
    )
else()
    message(CHECK_FAILED "Not Found")
endif()