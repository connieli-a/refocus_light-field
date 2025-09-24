cmake_minimum_required (VERSION 3.18)
project(camera LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED)

add_executable (camera "test.cpp")
target_link_libraries( camera PRIVATE pylon::pylon ${OpenCV_LIBS})
target_include_directories(camera PRIVATE ${OpenCV_INCLUDE_DIRS})