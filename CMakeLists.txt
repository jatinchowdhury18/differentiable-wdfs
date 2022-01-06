cmake_minimum_required(VERSION 3.15)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "Minimum OS X deployment target")
project(DifferentiableWDFs VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(COMPANY_NAME "chowdsp")
set(PRODUCT_NAME "DifferentiableWDFs")

add_subdirectory(modules)
add_subdirectory(plugin)
