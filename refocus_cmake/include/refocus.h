// refocus_cmake.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。


#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <pylon/PylonIncludes.h>
#include <pylon/BaslerUniversalInstantCamera.h> 
// #include <opencv2/cudaarithm.hpp>
// TODO: 在此处引用程序需要的其他标头。
using namespace std;
using namespace Pylon;
using namespace GenApi; // NodeMap / CFloatPtr / CEnumerationPtr


struct CircleInf
{
    /* data */
    float x;
    float y;
    float radius;
    int valid;
    CircleInf(): x(0.0f), y(0.0f), radius(0.0f), valid(0){}
    CircleInf(float _x, float _y, float r): x(_x), y(_y), radius(r), valid(1) {}
    
};
struct Result
{
    float brenner;
    float z;
};

class ImageProcessor {
    public:
    virtual ~ImageProcessor() = default;
    
    static std::shared_ptr<ImageProcessor> create(const vector<CircleInf>& circleList, const float y_tolerance, const int patch_size_cpp, const int image_rows, const int image_cols, const int32_t device = 0, const bool useGraphe = true);

    
    // pure virtual functions, only defining interfaces
    virtual float imageprocess_cuda(
    const cv::Mat& image_mla) = 0;         // CV_8UC3
                        
    virtual vector<cv::Vec3f> currentimage() = 0;
    
    virtual int get_col() const = 0;
    virtual int get_row() const = 0;
    
    protected:
    int n_cols;
    int n_rows;
    
};









