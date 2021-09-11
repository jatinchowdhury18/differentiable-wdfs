cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
project (differentiable-wdfs)

function(create_wdf_exe name)
    message(STATUS "Creating WDF simulation executable: ${name}...")
    add_executable(${name} wdf_cpp/${name}.cpp)
    target_include_directories(${name} PUBLIC modules/chowdsp_utils/DSP/WDF)
endfunction()

create_wdf_exe(wdf-clipper)
create_wdf_exe(wdf-clipper-pot)
