cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
project (differentiable-wdfs)

add_executable(wdf-clipper
    wdf_cpp/wdf-clipper.cpp
)

# include WDF library
target_include_directories(wdf-clipper
    PUBLIC
        modules/chowdsp_utils/DSP/WDF
)