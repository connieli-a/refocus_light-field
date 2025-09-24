# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 3.18)

# # 如果支持，请为 MSVC 编译器启用热重载。
# if (POLICY CMP0141)
#   cmake_policy(SET CMP0141 NEW)
#   set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
# endif()
# for Windows system. All other OS PYLON_DEV_DIR will be empty

project ("refocus_cmake")


# Pylon SDK 路径
list(APPEND CMAKE_PREFIX_PATH $ENV{PYLON_DEV_DIR})

find_package(pylon 10.0.0 REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 包含子项目。
add_subdirectory ("./refocus_cmake")
# add_subdirectory ("./camera")
# target_compile_options(refocus_cuda PRIVATE "/source-charset:utf-8" "/execution-charset:utf-8")
