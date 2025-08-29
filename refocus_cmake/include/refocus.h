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
// #include <opencv2/cudaarithm.hpp>
// TODO: 在此处引用程序需要的其他标头。
using namespace std;



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

class ImageProcessor {
    public:
    virtual ~ImageProcessor() = default;
    
    static std::shared_ptr<ImageProcessor> create(const vector<cv::Vec3f> circles, const float y_tolerance, const int patch_size_cpp, const int num_depth_plane, const std::vector<float> disparity_x_flat, const std::vector<float> disparity_y_flat, const int32_t device = 0, const bool useGraphe = true);

    
    // pure virtual functions, only defining interfaces
    virtual std::vector<cv::Vec3f> imageprocess_cuda(
    const cv::Mat& image_mla) = 0;                     // CV_8UC3

    
    virtual int get_col() const = 0;
    virtual int get_row() const = 0;
    
    protected:
    int n_cols;
    int n_rows;
   
};



void generate_disparity_table(const int num_depth_plane, const int start, const int end, const int patch_size, vector<float>& disparity_x_flat, vector<float>& disparity_y_flat);






