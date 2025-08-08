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
// TODO: 在此处引用程序需要的其他标头。
using namespace std;
using namespace cv;

struct CircleInf
{
    /* data */
    Point2f center;
    float radius;
    bool valid;
    CircleInf(): center(0.0f, 0.0f), radius(0.0f), valid(false){}
    CircleInf(Point2d c, float r): center(c), radius(r), valid(true) {}
};



vector<vector<CircleInf>>extract_rows(vector<CircleInf> sortedList, float y_tolerance,  int& n_cols);

// vector<vector<Mat>>extract_microlens_patches(Mat image_mla, int patch_size, vector<vector<CircleInf>>rows);
// Mat extract_patch(const Mat& image, const Point2f& center, int patch_size);
// void transpose_patch_array(const vector<vector<Mat>>& patches, vector<vector<vector<vector<Vec3b>>>> &vp_img_arr, vector<vector<Mat>> &img_arr, int &n_rows, int &n_cols, int &patch_h, int &patch_w);
Vec3b bilinear_rgb(const Mat& image_mla, const Point2f& pt);
void transform_image(const Mat& image_mla, const int patch_size, const vector<vector<CircleInf>>& rows, vector<Vec3f>& images, const int n_rows, const int n_cols);

// vector<Mat> shift_and_sum(const vector<vector<Mat>> &images, vector<vector<vector<float>>> & disparity_x, vector<vector<vector<float>>> & disparity_y, const vector<float>& depth_range );
vector<Mat> shift_and_sum(const vector<Vec3f> &images,  const int n_rows, const int n_cols, const int num_depth_plane,const int patch_size, const vector<vector<vector<float>>>& disparity_x, const vector<vector<vector<float>>>& disparity_y);
void shift_img(const Mat& img, float dx, float dy, Mat& shifted);
// void shift_img_implace(const Mat& img, float dx, float dy, Mat& out);
