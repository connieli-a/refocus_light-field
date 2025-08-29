#include "include/cuda_kernel.cuh"
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>


GPUProcessor::GPUProcessor(const vector<cv::Vec3f>& circles, const float y_tolerance, const int patch_size_cpp, const int num_depth_plane_cpp, const std::vector<float> disparity_x_flat, const std::vector<float> disparity_y_flat):  patch_size(patch_size_cpp), num_depth_plane(num_depth_plane_cpp),disparity_x_flat(disparity_x_flat), disparity_y_flat(disparity_y_flat){
    extract_rows(circles, y_tolerance);
    prepare_data();
    //set the GPU information 
    // 使用GPUの設定

    cudaMalloc(&d_rows_flat, sizeof(CircleInf) * total_circles);
    cudaMalloc(&d_rows_offsets, sizeof(int) * rows_offsets.size());
    cudaMalloc(&d_disp_x, patch_area * num_depth_plane * sizeof(float));
    cudaMalloc(&d_disp_y, patch_area * num_depth_plane * sizeof(float));
    cudaMemcpy(d_rows_flat, rows_flat.data(),
    sizeof(CircleInf) * total_circles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows_offsets, rows_offsets.data(),
    sizeof(int) * rows_offsets.size(), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_disp_x, disparity_x_flat.data(),
    patch_area * num_depth_plane * sizeof(float),
    cudaMemcpyHostToDevice);
    cudaMemcpy(d_disp_y, disparity_y_flat.data(),
    patch_area * num_depth_plane * sizeof(float),
    cudaMemcpyHostToDevice);
    
    
}
GPUProcessor::~GPUProcessor(){
    cudaFree(d_images);
    cudaFree(d_rows_offsets);
    cudaFree(d_rows_flat);
    cudaFree(d_volume);
    cudaFree(d_disp_x);
    cudaFree(d_disp_y);
    cudaFree(d_imagefloat);
    // cudaStreamSynchronize(m_stream);
}

// __device__ __forceinline__ Vec3f get_pixel(const uchar3* img, int idx, int cols, size_t step){
//     int y = idx / cols;
//     int x = idx % cols;
//     const uchar3* row_ptr = (const uchar3*) ((const uchar*)img + y * step);
//     uchar3 c = row_ptr[x];
//     return Vec3f(float(c.x)/255.f, float(c.y)/255.f, float(c.z)/255.f);
// }
// // transform_双线性取样（uchar3 BGR）
// __device__ inline Vec3f bilinear_bgr(
//     const uchar3* img, int rows, int cols, size_t step, float x, float y)
// {
   
//     int ix = floorf(x);
//     int iy = floorf(y);
//     float dx = x - ix;
//     float dy = y - iy;

//     if (ix < 0 || ix + 1 >= cols || iy < 0 || iy + 1 >= rows)
//         return Vec3f(0.f, 0.f, 0.f);

//     Vec3f c00 = get_pixel(img, ix + iy * cols, cols, step);
//     Vec3f c01 = get_pixel(img, (ix + 1)+ iy* cols,cols ,step);
//     Vec3f c10 = get_pixel(img, ix + (iy + 1) * cols, cols,step);
//     Vec3f c11 = get_pixel(img, (ix + 1) + (iy + 1) * cols,cols, step);
//     // Vec3f c00 = img[iy*cols + ix];
//     // Vec3f c01 = img[iy*cols + (ix+1)];
//     // Vec3f c10 = img[(iy+1)*cols + ix];
//     // Vec3f c11 = img[(iy+1)*cols + (ix+1)];

//     Vec3f val;
//     val[0] = (1 - dx) * (1 - dy) * c00[0] + dx * (1 - dy) * c01[0]
//           + (1 - dx) * dy * c10[0] + dx * dy * c11[0];
//     val[1] = (1 - dx) * (1 - dy) * c00[1] + dx * (1 - dy) * c01[1]
//           + (1 - dx) * dy * c10[1] + dx * dy * c11[1];
//     val[2] = (1 - dx) * (1 - dy) * c00[2] + dx * (1 - dy) * c01[2]
//           + (1 - dx) * dy * c10[2] + dx * dy * c11[2];
//     return val;
// }
__global__ void copy_kernel(const uchar3* __restrict__ src, int rows, int cols, size_t step,
                            Vec3f* __restrict__ dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    const uchar3* row_ptr = (const uchar3*)((const char*)src + y * step);
    uchar3 p = row_ptr[x];

    int idx = y * cols + x;
    dst[idx] = Vec3f(float(p.x)/255.f, float(p.y)/255.f, float(p.z)/255.f);
}
//双线性插值bilinear interpolation
__device__ inline Vec3f bilinear_lookup(
    const Vec3f* img, int n_rows, int n_cols, float x, float y)
{
    int ix = floorf(x);
    int iy = floorf(y);
    float dx = x - ix;
    float dy = y - iy;

    if (ix < 0 || ix + 1 >= n_cols || iy < 0 || iy + 1 >= n_rows)
        return Vec3f(0.f, 0.f, 0.f);

    Vec3f c00 = img[iy * n_cols + ix];
    Vec3f c01 = img[iy * n_cols + (ix + 1)];
    Vec3f c10 = img[(iy + 1) * n_cols + ix];
    Vec3f c11 = img[(iy + 1) * n_cols + (ix + 1)];

    Vec3f val;
    val[0] = (1 - dx) * (1 - dy) * c00[0] + dx * (1 - dy) * c01[0]
          + (1 - dx) * dy * c10[0] + dx * dy * c11[0];
    val[1] = (1 - dx) * (1 - dy) * c00[1] + dx * (1 - dy) * c01[1]
          + (1 - dx) * dy * c10[1] + dx * dy * c11[1];
    val[2] = (1 - dx) * (1 - dy) * c00[2] + dx * (1 - dy) * c01[2]
          + (1 - dx) * dy * c10[2] + dx * dy * c11[2];
    return val;
}

__global__ void transform_kernel( Vec3f* d_img, int image_rows, int image_cols, 
    const CircleInf* __restrict__ rows_flat, const int* __restrict__ rows_offsets,
    int total_circles, int n_rows, int n_cols, int patch_size,
    Vec3f* __restrict__ out){
    if(!d_img ||!rows_flat||!rows_offsets ) return; 
    int patch_area = patch_size * patch_size;
    int total = n_rows * n_cols * patch_area;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    
   
    // if(tid == 0){
    //     printf("kernel running!\n");
    // }
    
    float half = (patch_size - 1) * 0.5f;        
    // 还原 (uv, s, t)
    int uv_idx = tid % (n_rows*n_cols); // 每 patch 内 uv_idx
    int s = (tid / (n_rows*n_cols)) / patch_size;
    int t = (tid / (n_rows*n_cols)) % patch_size;

    int u = uv_idx / n_cols;
    int v = uv_idx % n_cols;
    
    
    // 这一行的起止（可处理“每行列数不同”的情况）
    int start = rows_offsets[u];
    // int end   = (u + 1 < n_rows) ? rows_offsets[u + 1] : total_circles;
    // int row_len = end - start;

    // // 若该 (u,v) 在这一行不存在，直接填零
    // if (v >= row_len) {
    //     out[3 * tid + 0] = 0.f;
    //     out[3 * tid + 1] = 0.f;
    //     out[3 * tid + 2] = 0.f;
    //     return;
    // }

    CircleInf ci = rows_flat[start + v];
    int idx_out = (s * patch_size + t) * n_rows * n_cols + u * n_cols + v;
    if (!ci.valid) {
      out[idx_out] = Vec3f(0,0,0);
        return;
    }

    // 计算取样位置（与 CPU 代码一致）
    float x = ci.x + (t - half);
    float y = ci.y + (s - half);

    Vec3f bgr = bilinear_lookup(d_img, image_rows, image_cols,  x, y);
    out[idx_out] = bgr;
}



__global__ void shift_and_sum_kernel(const Vec3f* d_images,  // [patch_area][n_rows][n_cols]
                          Vec3f* d_volume,        // [num_depth_plane][n_rows][n_cols]
                          const float* d_disp_x,  // [patch_area][num_depth_plane]
                          const float* d_disp_y,  // [patch_area][num_depth_plane]
                          int n_rows, int n_cols, int patch_size,
                          int num_depth_plane)
{
    int u = blockIdx.y * blockDim.y + threadIdx.y;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z;  // 当前深度平面

    if (u >= n_rows || v >= n_cols || z >= num_depth_plane) return;

    int patch_area = patch_size * patch_size;
    Vec3f sum = {0, 0, 0};

    for (int idx = 0; idx < patch_area; idx++) {
        // int h = idx / patch_size;
        // int w = idx % patch_size;

        // disparity lookup
        int disp_idx = idx * num_depth_plane + z;
        float dx = d_disp_x[disp_idx];
        float dy = d_disp_y[disp_idx];

        // 原始像素位置（patch 内偏移）
        int img_base_idx = idx * n_rows * n_cols;

        // 浮点位移，使用双线性插值
        float uu = u + dy;
        float vv = v + dx;
        Vec3f val = bilinear_lookup(d_images + img_base_idx, n_rows, n_cols, vv, uu);

        sum[0] += val[0];
        sum[1] += val[1];
        sum[2] += val[2];
    }

    // 平均
    sum[0] /= patch_area;
    sum[1] /= patch_area;
    sum[2] /= patch_area;

    // 输出到 refocused volume
    int out_idx = z * n_rows * n_cols + u * n_cols + v;
    d_volume[out_idx] = sum;
}





std::vector<cv::Vec3f> GPUProcessor::imageprocess_cuda(
    const cv::Mat& image_mla                 // CV_8UC3      
){
      //------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //-------------------------
    CV_Assert(image_mla.type() == CV_8UC3);
    CV_Assert(patch_size > 0);
    
    //  配置 kernel 维度并 launch
    int threads = 256;
    int total_threads = n_rows * n_cols * patch_area;
    int transform_blocks = (total_threads + threads - 1) / threads;
    
    // cudaEventRecord(start);
    cuda_preprocess(image_mla);
   
    
    //     //------------
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);

    // float ms = 0;
    // cudaEventElapsedTime(&ms, start, stop);
    // printf("the rest of part耗时 : %f ms\n", ms);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    // //------------------

    
    // // 拷回 CPU
    // std::vector<cv::Vec3f> h_images(image_rows * image_cols);
    // cudaMemcpy(h_images.data(), d_images, h_images.size() * sizeof(Vec3f), cudaMemcpyDeviceToHost);
    
    cudaMalloc(&d_images, sizeof(Vec3f) * n_rows * n_cols * patch_area);
    cudaMalloc(&d_volume, num_depth_plane * n_rows * n_cols * sizeof(Vec3f));
 
    // 在同一 stream 里执行 kernel
    transform_kernel<<<transform_blocks, threads>>>(
        d_imagefloat, image_rows, image_cols, 
        d_rows_flat, d_rows_offsets,
         total_circles, n_rows, n_cols, patch_size,
         d_images
    );
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) printf("Kernel error: %s\n", cudaGetErrorString(err));
    
    // kernel 配置
    dim3 block(16, 16);
    dim3 grid((n_cols+15)/16, (n_rows+15)/16, num_depth_plane);
    
    shift_and_sum_kernel<<<grid, block>>>(d_images, d_volume,
    d_disp_x, d_disp_y, n_rows, n_cols, patch_size, 
    num_depth_plane);
        
    cudaError_t err1 = cudaGetLastError();
    if(err1 != cudaSuccess) printf("Kernel error: %s\n", cudaGetErrorString(err1));
    
    // 结果拷回 CPU
    // Device → Host 异步拷贝
    std::vector<cv::Vec3f> h_volume(num_depth_plane * n_rows * n_cols);
    static_assert(sizeof(Vec3f) == sizeof(cv::Vec3f), "Vec3f layout mismatch!");
    cudaMemcpy(h_volume.data(), d_volume,
    h_volume.size() * sizeof(Vec3f),
    cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    
    return h_volume;
}
void GPUProcessor::prepare_data(){
    rows_flat.clear();
    rows_offsets.clear();
    //flatten the uploaded data
    int offsets = 0;
    for(auto& row : rows){
        rows_offsets.push_back(offsets);
        rows_flat.insert(rows_flat.end(), row.begin(), row.end());
        offsets += row.size();
    }
    total_circles = static_cast<int>(rows_flat.size());
    // cout<<"rows_flat"<<rows_flat.size()<<endl;
}
void GPUProcessor::extract_rows(const vector<cv::Vec3f>& circles, const float y_tolerance){
    vector<CircleInf> sortedList;
    preprocess(circles, sortedList);
    int estimated_rows = max(1, (int)(sortedList.size()/34));
    rows.clear();
    rows.reserve(estimated_rows);
    while (!sortedList.empty())
    {
        /* code */
        float row_y = sortedList[0].y;
        for (const auto& c : sortedList)
        {
            /* code */
            if (c.y < row_y)
                row_y = c.y;
        }
        vector<CircleInf>current_row;
        current_row.reserve(34);
        vector<CircleInf>remaining_row;
        remaining_row.reserve(sortedList.size());
        for (const auto& c : sortedList) {
            if (abs(row_y - c.y) < y_tolerance)
                current_row.push_back(c);
            else remaining_row.push_back(c);
        }
        //sort by x-axis
        sort(current_row.begin(), current_row.end(), [](const CircleInf& a, const CircleInf& b) {
            return a.x < b.x;
            });
        rows.push_back(current_row);
        // cout<< current_row.size()<<endl;
        sortedList = move(remaining_row);

    }
    size_t max_cols = 0;
    for (const auto& row : rows)
    {
        max_cols = max(max_cols, row.size());
    }
    for(auto& row : rows){
        if(row.size() < max_cols){
            row.resize(max_cols);
        }
    }
    n_cols = static_cast<int>(max_cols);
    n_rows =  static_cast<int>(rows.size());
    // cout<<"rows size"<<rows.size()<<endl;
    
}
void GPUProcessor::preprocess(const vector<cv::Vec3f>& circles, vector<CircleInf>& sortedList){
    vector<CircleInf> circleList;
    //orgnaize the array
    for (int i = 0; i < circles.size(); i++)
    {
        float x = std::round(circles[i][0] * 100) / 100.0f;
        float y = std::round(circles[i][1] * 100) / 100.0f;
        float radius = std::round(circles[i][2] * 100) / 100.0f;
        circleList.push_back({ x, y, radius });
        
    }
    //sort by the y-axis
    vector<int> idx(circleList.size());
    for (int i = 0; i < idx.size(); i++) idx[i] = i;
    sort(idx.begin(), idx.end(), [&](int a, int b) {
        return circleList[a].y < circleList[b].y;
    });
    
    for (int i = 0; i < idx.size(); i++)
    {
        /* code */
        sortedList.push_back(circleList[idx[i]]);
    }
    
    //set the range
    int rangex1 = 900, rangex2 = 3000;
    int rangey1 = 125, rangey2 = 2000;
    
    vector<CircleInf> rangeList;
    for (int i = 0; i < sortedList.size(); i++)
    {
        /* code */
        float x = sortedList[i].x;
        float y = sortedList[i].y;
        
        if (x >= rangex1 && x <= rangex2 && y >= rangey1 && y <= rangey2) {
            rangeList.push_back(sortedList[i]);
        }
    }
    sortedList = move(rangeList);

}
void GPUProcessor::cuda_preprocess(const cv::Mat& image_mla){
    cv::cuda::GpuMat d_img;
    d_img.upload(image_mla);
    uchar3* d_img_ptr = d_img.ptr<uchar3>();
   
    image_rows = d_img.rows;
    image_cols = d_img.cols;
    size_t step = d_img.step; 
   

    dim3 block_copy(16, 16);
    dim3 grid_copy((image_cols + 15)/16, (image_rows + 15)/16);
    // 分配输出 device 内存（用 float*，回传时直接拷回到 Vec3f 数组）
    cudaMalloc(&d_imagefloat, sizeof(Vec3f) * image_rows * image_cols);
    copy_kernel<<<grid_copy, block_copy>>>(d_img_ptr, image_rows, image_cols, step, d_imagefloat);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) printf("Kernel error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}
