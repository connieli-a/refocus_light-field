# CMakeList.txt: refocus_cmake 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#

# 将源代码添加到此项目的可执行文件。
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

set(OpenCV_DIR "G:\\program_files\\vcpkg\\installed\\x64-windows\\share\\opencv4")
set(quirc_DIR "G:\\program_files\\vcpkg\\installed\\x64-windows\\share\\quirc")

add_executable (refocus_cmake "refocus_cmake.cpp" "refocus_cmake.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET refocus_cmake PROPERTY CXX_STANDARD 20)
endif()

# TODO: 如有需要，请添加测试并安装目标。
#target_include_directories(refocus_cmake PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(refocus_cmake PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)


