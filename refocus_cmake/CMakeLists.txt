# CMakeList.txt: refocus_cmake 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#MSVC + NVCC + Opencv + Pylon + Multi-thread

cmake_minimum_required (VERSION 3.18)

enable_language(CUDA)
#设置项目及使用语言
project(refocus_cmake LANGUAGES CXX CUDA)



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# 指定 host 编译器为 MSVC cl.exe
set(CMAKE_C_COMPILE "cl.exe")
set(CMAKE_CXX_COMPILE "cl.exe")
set(CMAKE_CUDA_HOST_COMPILER "C://Program Files//Microsoft Visual Studio//2022//Community//VC//Tools//MSVC//14.42.34433//bin//Hostx64//x64//cl.exe" CACHE FILEPATH "CUDA host compiler")

set(OpenCV_DIR "C://app//opencv-4.10.0//install//x64//vc17//lib")
# set(quirc_DIR "G:\\program_files\\vcpkg\\installed\\x64-windows\\share\\quirc")
# set(ZLIB_DIR "G:\\program_files\\vcpkg\\installed\\x64-windows\\share\\zlib")
# 将源代码添加到此项目的可执行文件。
# find_package(OpenMP REQUIRED)
# find_package(ZLIB REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
#optimization
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG")



add_executable (refocus_cmake "main.cpp" "refocus.cpp" "kernel.cu" "globals.cpp" "DAControl.cpp" "ADControl.cpp")

# 让 CMake 控制运行库（cpp + cuda 都会生效）
cmake_policy(SET CMP0091 NEW)
#set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")        # /MD
#set(CMAKE_MSVC_RUNTIME_LIBRARY_DEBUG "MultiThreadedDebugDLL") # /MDd
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")            #MT
set(CMAKE_MSVC_RUNTIME_LIBRARY_DEBUG "MultiThreadedDebug")  #MTd

#设置CUDA架构

set_target_properties(refocus_cmake PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES 75)



add_compile_options(/wd611 /wd9025)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=611")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I${OpenCV_INCLUDE_DIRS}")
# TODO: 如有需要，请添加测试并安装目标。
target_include_directories(refocus_cmake PRIVATE ${OpenCV_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
#target_link_directories(refocus_cmake PRIVATE ${CMAKE_SOURCE_DIR}/refocus_cmake/lib/GPC3100/x64 ${CMAKE_SOURCE_DIR}/refocus_cmake/lib/GPC3300)
target_link_libraries(refocus_cmake 
    PRIVATE pylon::pylon 
    ${OpenCV_LIBS} 
    "C:/Program Files (x86)/Interface/GPC3100/lib/x64/FbiAd.lib"
    "C:/Program Files (x86)/Interface/GPC3100/lib/x64/FbiAdDC.lib"
    "C:/Program Files (x86)/Interface/GPC3300/lib/x64/FbiDa.lib"
    "C:/Program Files (x86)/Interface/GPC3300/lib/x64/FbiDaDC.lib"

)


set_target_properties(refocus_cmake PROPERTIES
    COMPILE_PDB_NAME ""
    COMPILE_PDB_OUTPUT_DIRECTORY ""
)




