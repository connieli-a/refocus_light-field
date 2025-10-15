#include "include/cuda_kernel.cuh"
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

GPUProcessor::GPUProcessor(const vector<CircleInf>& circleList, const float y_tolerance, const int patch_size_cpp,  const int image_rows, const int image_cols):  patch_size(patch_size_cpp),   image_rows(image_rows), image_cols(image_cols){
    extract_rows(circleList, y_tolerance);
    prepare_data();
    //set the GPU information 
    // 使用GPUの設定
    cudaStreamCreate(&m_stream);

    cudaMalloc(&d_rows_flat, sizeof(CircleInf) * total_circles);
    cudaMalloc(&d_rows_offsets, sizeof(int) * rows_offsets.size());
    cudaMalloc(&d_imagefloat, sizeof(Vec3f) * image_rows * image_cols);
    cudaMalloc(&d_images, sizeof(Vec3f) * n_rows * n_cols * patch_area);
    cudaMalloc(&d_volume, num_depth_plane * n_rows * n_cols * sizeof(Vec3f));
    cudaMalloc(&d_depth, sizeof(float) * num_depth_plane);
    cudaMalloc(&d_brenner_scores, sizeof(float) * num_depth_plane);
    cudaMalloc(&d_volume_1, n_rows * n_cols * sizeof(Vec3f));
    cudaMalloc(&d_depth_1, sizeof(float) );
    cudaMalloc(&d_image_1, sizeof(Vec3f) * n_rows * n_cols);
    cudaMalloc(&d_volume_n, num_plane * n_rows * n_cols * sizeof(Vec3f));
    cudaMalloc(&d_depth_n, sizeof(float) * num_plane);
    cudaMalloc(&d_brenner_scores_n, sizeof(float) * num_plane);
    // cudaMemcpy(d_rows_flat, rows_flat.data(),
    // sizeof(CircleInf) * total_circles, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_rows_offsets, rows_offsets.data(),
    // sizeof(int) * rows_offsets.size(), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_rows_flat, rows_flat.data(),
                sizeof(CircleInf) * total_circles,
                cudaMemcpyHostToDevice, m_stream);

    cudaMemcpyAsync(d_rows_offsets, rows_offsets.data(),
                sizeof(int) * rows_offsets.size(),
                cudaMemcpyHostToDevice, m_stream);
    d_img.create(cv::Size(image_cols, image_rows), CV_8UC3);
}
GPUProcessor::~GPUProcessor(){

    cudaStreamSynchronize(m_stream);
    cudaStreamDestroy(m_stream);

    cudaFree(d_rows_flat);
    cudaFree(d_rows_offsets);
    cudaFree(d_imagefloat);
    cudaFree(d_images);
    cudaFree(d_volume);
    cudaFree(d_depth);     
    cudaFree(d_brenner_scores);
    cudaFree(d_volume_1);
    cudaFree(d_depth_1);
    cudaFree(d_image_1);
    cudaFree(d_volume_n);
    cudaFree(d_depth_n);     
    cudaFree(d_brenner_scores_n);
}


__global__ void copy_kernel(const uchar3* __restrict__ src, int rows, int cols, size_t step,
                            Vec3f* __restrict__ dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    const uchar3* row_ptr = (const uchar3*)((const char*)src + y * step);
    uchar3 p = row_ptr[x];

    int idx = y * cols + x;
    // int idx = y * cols + x;
    // uchar3 p = src[idx]; 
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
__global__ void transform_kernel_centerview( Vec3f* d_img, int image_rows, int image_cols, 
    const CircleInf* __restrict__ rows_flat, const int* __restrict__ rows_offsets,
    int total_circles, int n_rows, int n_cols, int patch_size,
    Vec3f* __restrict__ out){
    if(!d_img ||!rows_flat||!rows_offsets ) return; 
    
    
    int uv_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (uv_idx >= n_rows * n_cols) return;
   
    float half = (patch_size - 1) * 0.5f;        
    // 还原 (uv, s, t)
   
    int s = half;
    int t = half;

    int u = uv_idx / n_cols;
    int v = uv_idx % n_cols;
    
    
    // 这一行的起止（可处理“每行列数不同”的情况）
    int start = rows_offsets[u];
   
    CircleInf ci = rows_flat[start + v];
    if (!ci.valid) {
      out[u * n_cols + v] = Vec3f(0,0,0);
        return;
    }

    // 计算取样位置（与 CPU 代码一致）
    float x = ci.x + (t - half);
    float y = ci.y + (s - half);

    Vec3f bgr = bilinear_lookup(d_img, image_rows, image_cols,  x, y);
    out[u * n_cols + v] = bgr;
}


__global__ void shift_and_sum_kernel(const Vec3f* d_images,  // [patch_area][n_rows][n_cols]
                          Vec3f* d_volume,        // [num_depth_plane][n_rows][n_cols]
                          int n_rows, int n_cols, int patch_size,
                          const float* d_depth, const int num_depth_plane, float pixel_size, int s,//the diameter of the lens
                            int f )//the focal length
{
    int u = blockIdx.y * blockDim.y + threadIdx.y;//row
    int v = blockIdx.x * blockDim.x + threadIdx.x;//col
    int z = blockIdx.z;  // 当前深度平面

    if (u >= n_rows || v >= n_cols || z >= num_depth_plane) return;
    int mid_idx = (patch_size - 1) * 0.5f;
    int patch_area = patch_size * patch_size;
    Vec3f sum = {0, 0, 0};
    float depth = d_depth[z];
    float factor = s * (depth / ((float)f + depth)) / pixel_size;
    float radius = mid_idx * 0.75f ;
    int count = 0;
    for (int idx = 0; idx < patch_area; idx++) {
       
        int h = idx / patch_size;
        int w = idx % patch_size;
        float r2 = (h - mid_idx) * (h - mid_idx) + (w - mid_idx) * (w - mid_idx);
        if(r2 <= radius * radius){
            // disparity compute
            count++;
            float dx = (w - mid_idx) * factor;
            float dy = (h - mid_idx) * factor;

            // 原始像素位置（一个一个views检查）
            int img_base_idx = idx * n_rows * n_cols;

            // 浮点位移，使用双线性插值
            float uu = (float)u + dy;
            float vv = (float)v + dx;
            // uu = fminf(fmaxf(uu, 0.f), n_rows - 1.001f);
            // vv = fminf(fmaxf(vv, 0.f), n_cols - 1.001f);
            Vec3f val = bilinear_lookup(d_images + img_base_idx, n_rows, n_cols, vv, uu);

            sum[0] += val[0];
            sum[1] += val[1];
            sum[2] += val[2];
        }
    }

    // 平均
    sum[0] /= count;
    sum[1] /= count;
    sum[2] /= count;

    
    // 输出到 refocused volume
    int out_idx = z * n_rows * n_cols + u * n_cols + v;
    d_volume[out_idx] = sum;

}
__global__ void brenner_kernel(const Vec3f* d_volume, float* d_brenner_scores, int m, int n_rows, int n_cols, const int num_depth_plane){
    int u = blockIdx.y * blockDim.y + threadIdx.y;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.z;  // 当前深度平面

    if (u >= n_rows || v >= n_cols || z >= num_depth_plane) return;
    float brenner_val = 0.0f;
    int idx = z*n_rows*n_cols + u*n_cols + v;
    Vec3f color = d_volume[idx];
    float center = (color.x * 0.299f + color.y * 0.587f + color.z * 0.114f) * 255.0f;
    if(u + m < n_rows){
        Vec3f down_color = d_volume[z * n_rows * n_cols + (u + m) * n_cols + v];
        float down = down_color.x * 0.299f + down_color.y * 0.587f + down_color.z * 0.114f;
        brenner_val += (center - down) * (center - down) ;
    }
    if(v + m < n_cols){
        Vec3f right_color = d_volume[z * n_rows * n_cols + u  * n_cols + (v + m)];
        float right = right_color.x * 0.299f + right_color.y * 0.587f + right_color.z * 0.114f;
        brenner_val += (center - right) * (center - right);
    }
    atomicAdd(&d_brenner_scores[z], brenner_val);
}


float GPUProcessor::imageprocess_cuda(
    const cv::Mat& image_mla, const vector<float>& depth_range                // CV_8UC3      
){
      //------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    //-------------------------
    cudaEventRecord(start);
    CV_Assert(image_mla.type() == CV_8UC3);
    CV_Assert(patch_size > 0);
    
    //  配置 kernel 维度并 launch
    int threads = 256;
    int total_threads = n_rows * n_cols * patch_area;
    int transform_blocks = (total_threads + threads - 1) / threads;
    
   
    cuda_preprocess(image_mla);
   
    // // 拷回 CPU
    // std::vector<cv::Vec3f> h_images(image_rows * image_cols);
    // cudaMemcpy(h_images.data(), d_images, h_images.size() * sizeof(Vec3f), cudaMemcpyDeviceToHost);
    // cudaEventRecord(start);
    

    // 在同一 stream 里执行 kernel
    transform_kernel<<<transform_blocks, threads, 0, m_stream>>>(
        d_imagefloat, image_rows, image_cols, 
        d_rows_flat, d_rows_offsets,
        total_circles, n_rows, n_cols, patch_size,
        d_images
    );
    cudaError_t err_transform = cudaGetLastError();
    if(err_transform != cudaSuccess) printf("transform_Kernel error: %s\n", cudaGetErrorString(err_transform));
    cudaDeviceSynchronize();

    // kernel 配置
    dim3 block(16, 16);
    dim3 grid((n_cols+15)/16, (n_rows+15)/16, num_depth_plane);
    
    //generate depth_range
    // cudaMemcpy(d_depth, depth_range.data(),
    // sizeof(float) * num_depth_plane, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_depth, depth_range.data(),
                sizeof(float) * num_depth_plane,
                cudaMemcpyHostToDevice, m_stream);
    cudaMemsetAsync(d_brenner_scores, 0, sizeof(float) * num_depth_plane, m_stream);

    shift_and_sum_kernel<<<grid, block, 0, m_stream>>>(d_images, d_volume,
    n_rows, n_cols, patch_size, 
    d_depth, num_depth_plane, pixel_size, s, f);
    cudaError_t err_shift = cudaGetLastError();
    if(err_shift != cudaSuccess) printf("shift_Kernel error: %s\n", cudaGetErrorString(err_shift));
    cudaDeviceSynchronize();
    
    
    int m = 4;
    brenner_kernel<<<grid, block, 0, m_stream>>>(d_volume, d_brenner_scores, m, n_rows, n_cols, num_depth_plane);
    cudaError_t err_brenner = cudaGetLastError();
    if(err_brenner != cudaSuccess) printf("brenner_Kernel error: %s\n", cudaGetErrorString(err_brenner));
    cudaDeviceSynchronize();
    
    std::vector<float> h_brenner_scores(num_depth_plane);
    // cudaMemcpy(h_brenner_scores.data(), d_brenner_scores,
    // h_brenner_scores.size() * sizeof(float),
    // cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(h_brenner_scores.data(), d_brenner_scores,
                sizeof(float) * num_depth_plane,
                cudaMemcpyDeviceToHost, m_stream);
    

    for(int i = 0; i<num_depth_plane;i++){
        cout<<h_brenner_scores[i]<<","<<depth_range[i]<<endl;
        
    }
   
   
    auto it = std::max_element(h_brenner_scores.begin(), h_brenner_scores.end());
    int max_idx = std::distance(h_brenner_scores.begin(), it);
    float z_coarse = depth_range[max_idx];
    float max_score = *it;
    float min_score = *std::min_element(h_brenner_scores.begin(), h_brenner_scores.end());
    cout<<"coarse_best:"<<z_coarse<<", score = "<<max_score<<endl;
    float peak_contract = (max_score - min_score) / max_score;
    float z_best = z_coarse;
    cout<<"peak_contract:"<<peak_contract<<endl;
    if(peak_contract < 0.005){
        cout<<"peak too flat"<<endl;

    }else{
       
        float fine_step = step * 0.2f;//fine_step (0.9-1)
        cout<<"fine_step"<<fine_step<<endl;
       
        z_best = find_bestvalue(z_coarse, fine_step);
    }

    generate_best_view(z_best);





    cudaDeviceSynchronize();   
            
                              // ------------
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("imageprocess_cuda耗时 : %f ms\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //------------------
    
   
    // cudaMemcpy(d_depth_1, &z0,
    // sizeof(float) , cudaMemcpyHostToDevice);
    // cudaMemcpyAsync(d_depth_1, &z0,
    // sizeof(float) , cudaMemcpyHostToDevice, m_stream);
    // shift_and_sum_kernel<<<grid, block, 0, m_stream>>>(d_images, d_volume_1,
    //     n_rows, n_cols, patch_size, 
    //     d_depth_1, 1, pixel_size, s, f);
    // cudaDeviceSynchronize();
        
    // std::vector<cv::Vec3f> h_volume( n_rows * n_cols);
    // // // cudaMemcpy(h_volume.data(), d_volume_1,
    // // // h_volume.size() * sizeof(cv::Vec3f),
    // // // cudaMemcpyDeviceToHost);
    // cudaMemcpyAsync(h_volume.data(), d_volume_1,
    // h_volume.size() * sizeof(cv::Vec3f),
    // cudaMemcpyDeviceToHost, m_stream);
    // cudaStreamSynchronize(m_stream);
   
    // 结果拷回 CPU
    // Device → Host 异步拷贝
    // std::vector<cv::Vec3f> h_image(n_rows * n_cols * patch_area);
    // cudaMemcpy(h_image.data(), d_images,h_image.size() * sizeof(Vec3f),cudaMemcpyDeviceToHost);
    // std::vector<cv::Vec3f> h_volume(num_depth_plane * n_rows * n_cols);
    // cudaMemcpy(h_volume.data(), d_volume,
    // h_volume.size() * sizeof(cv::Vec3f),
    // cudaMemcpyDeviceToHost);
    // std::string folder = "results_large";
    // if(!fs::exists(folder)){
    //     fs::create_directory(folder);  
    // }

    // cv::Mat img(n_rows, n_cols, CV_32FC3);
 
    // cv::Mat img8, img8_large;
    // for(int slice_idx = 0; slice_idx < 1; slice_idx++){
    //         img.setTo(cv::Scalar(0,0,0));
    //         img8.setTo(cv::Scalar(0,0,0));
    //         img8_large.setTo(cv::Scalar(0,0,0));
    //         for (int i = 0; i < n_rows; ++i) {
    //                 cv::Vec3f* ptr = img.ptr<cv::Vec3f>(i);
    //                 for (int j = 0; j < n_cols; ++j) {
    //                         int idx = slice_idx * n_rows * n_cols + i * n_cols + j; // slice_idx 可用于多深度平面
    //                         ptr[j] = h_volume[idx];
                
    //                     }
    //                 }
    //         // 转为 8-bit 显示
    //         img.convertTo(img8, CV_8UC3, 255.0);
    //         cv::resize(img8, img8_large, cv::Size(), 10.0 ,10.0,cv::INTER_NEAREST);
    //         // std::string filename = folder + "/output"+ std::to_string(slice_idx) +".png";
    //         // cv::imwrite(filename, img8_large);
    //         // cv::imwrite("shift_and_sum_original.bmp", img8);
    //         // cv::imwrite("shift_and_sum_larger.bmp", img8_large);
    //         // // // 显示
     
    //         imshow("Volume Slice", img8_large);
    //         imshow("Volume Slice_original", img8);
    //         cv::waitKey(10);
            
    //         // cout<<"value:"<<h_brenner_scores[slice_idx]<<" depth:"<<depth_range[slice_idx]<<endl;
    //     }
    
    //  for(int s = 0; s < patch_size ; s++){
    //         for(int t = 0; t < patch_size; t++){
    //             img.setTo(cv::Scalar(0,0,0));

    //             for (int u = 0; u < n_rows; ++u) {
    //                cv::Vec3f* ptr = img.ptr<cv::Vec3f>(u);
    //                for (int v = 0; v < n_cols; ++v) {
    //                    int idx = (s * patch_size + t) * n_rows * n_cols + u * n_cols + v; // slice_idx 可用于多深度平面
    //                    ptr[v] = h_image[idx];
                   
    //                }
    //            }
    //             cv::Mat img8, img8_large;
    //            img.convertTo(img8, CV_8UC3, 255.0);
    //            cv::resize(img8, img8_large, cv::Size(), 10.0 ,10.0, cv::INTER_LINEAR);
    //             // std::string filename = folder + "/output"+ std::to_string(s*patch_size +t) +".png";
    //             // cv::imwrite(filename, img8);
    //         if (s * patch_size + t == patch_size * patch_size /2) 
    //          // 显示
    //         //   
    //             imshow("Volume Slice", img8_large);
    //                cv::waitKey(0);
    //         }
    //     }

    
    return z_best;
}



float GPUProcessor::quadratic_fit_points(const std::vector<float>& scores, const std::vector<float>& depths) {
    int n = scores.size();
    if (depths.size() != n || n < 3) {
        std::cerr << "Expect at least 3 points!" << std::endl;
        return depths[n/2];
    }
    float max_score = *std::max_element(scores.begin(), scores.end());
    float min_score = *std::min_element(scores.begin(), scores.end());
    float range = max_score - min_score;

    if(range < 1e-6 * max_score){
        cout<<"data is too similar"<<endl;
        auto max_it = std::max_element(scores.begin(), scores.end());
        return depths[std::distance(scores.begin(), max_it)];
    }
    // 构造正规方程的矩阵
    double sum_w = 0, sum_wx = 0, sum_wx2 = 0;
    double sum_wy = 0, sum_wxy = 0, sum_wx2y = 0;
    double sum_wx3 = 0, sum_wx4 = 0;

    for (int i = 0; i < n; ++i) {
        double normalized = (scores[i] - min_score) / range;
        double weight = pow(normalized, 4.0);  
        weight = max(weight, 0.005);
        cout<<"weight:"<<weight<<endl;
        double x = static_cast<double>(depths[i]);
        double y = static_cast<double>(scores[i]);
        
        sum_w += weight;
        sum_wx += weight * x;
        sum_wx2 += weight * x * x;
        sum_wx3 += weight * x * x * x;
        sum_wx4 += weight * x * x * x * x;
        sum_wy += weight * y;
        sum_wxy += weight * x * y;
        sum_wx2y += weight * x * x * y;
    }

    // 3x3 矩阵正规方程: 
    // | Sx4 Sx3 Sx2 |   |A|   = |Sx2y|
    // | Sx3 Sx2 Sx1 | * |B| = |Sxy |
    // | Sx2 Sx1 Sx0 |   |C|   |Sy   |

    // 用克拉默法则求解
    double D = sum_wx4 * (sum_wx2 * sum_w - sum_wx * sum_wx) 
             - sum_wx3 * (sum_wx3 * sum_w - sum_wx * sum_wx2) 
             + sum_wx2 * (sum_wx3 * sum_wx - sum_wx2 * sum_wx2);
    if (fabs(D) < 1e-10) {
        std::cerr << "Singular matrix (D=" << D << "), using max point" << std::endl;
        auto max_it = std::max_element(scores.begin(), scores.end());
        return depths[std::distance(scores.begin(), max_it)];
    }
    cout<<"D:"<<D<<endl;
    double Da = sum_wx2y * (sum_wx2 * sum_w - sum_wx * sum_wx) 
              - sum_wxy * (sum_wx3 * sum_w - sum_wx * sum_wx2) 
              + sum_wy * (sum_wx3 * sum_wx - sum_wx2 * sum_wx2);
              
    double Db = sum_wx4 * (sum_wxy * sum_w - sum_wy * sum_wx) 
              - sum_wx2y * (sum_wx3 * sum_w - sum_wx * sum_wx2) 
              + sum_wx2 * (sum_wx3 * sum_wy - sum_wx2 * sum_wxy);
    
    // double Dc = sum_wx4 * (sum_wx2 * sum_wy - sum_wx * sum_wx2y)
    //       - sum_wx3 * (sum_wx3 * sum_wy - sum_wx * sum_wx2y)
    //       + sum_wx2 * (sum_wx3 * sum_wxy - sum_wx2 * sum_wx2y);

    

    double A = Da / D;
    double B = Db / D;
    double C = (sum_wy - A * sum_wx2 - B * sum_wx) / sum_w;
    
    float z_best = -B / (2*A);
    double y_best = A * z_best * z_best + B * z_best + C;
    cout<<"z_best"<<z_best<<"y_best"<<y_best<<endl;
    return z_best;
}

float GPUProcessor::parabolic_peak_refine(const vector<float>& brenner_scores, const vector<float>& depth){
    auto max = std::max_element(brenner_scores.begin(), brenner_scores.end());
    int max_idx = std::distance(brenner_scores.begin(), max);
    if(max_idx == 0 || max_idx == brenner_scores.size() - 1){
        cout<<"center is not the maximun"<<endl;
        return depth[max_idx];
    }
    float x1 = depth[max_idx - 1];
    float x2 = depth[max_idx];
    float x3 = depth[max_idx + 1];
    float y1 = brenner_scores[max_idx - 1];
    float y2 = brenner_scores[max_idx];
    float y3 = brenner_scores[max_idx + 1];
    float numerator = y1 - y3;
    float denominator = 2 * (y1 - 2*y2 + y3);
    float curvature = fabs(y1 - 2 * y2 + y3) / y2;
    
    if(curvature < 0.0001){
        cout<<"curvature is too small"<<endl;
        return depth[max_idx];
    }
    if(denominator == 0){
        cout<<"fitting error"<<endl;
        return depth[max_idx];
    }
    cout<<"curvature"<<curvature<<endl;
    float z_best = x2 + numerator / denominator * (x3 - x2);
    float y_best = y2 - (numerator * numerator) / (8 * (y1 - 2*y2 + y3));
    cout<<"z_best"<<z_best<<"y_best"<<y_best<<endl;

    return z_best;
}

float GPUProcessor::find_bestvalue(float z_coarse, const float& fine_step){
    //second time fitting curve and generate the best depth image
    // kernel 配置
   
    

    dim3 block(16, 16);
    dim3 grid((n_cols+15)/16, (n_rows+15)/16, num_plane);
   

    vector<float> depth_zn;
    depth_zn.assign({z_coarse - 3*fine_step, z_coarse - 2*fine_step,z_coarse - fine_step, z_coarse, z_coarse + fine_step, z_coarse + 2 * fine_step, z_coarse + 3 * fine_step});
    // depth_zn.assign({z_coarse - fine_step, z_coarse, z_coarse + fine_step});
    cudaMemcpyAsync(d_depth_n, depth_zn.data(),
                sizeof(float) * num_plane,
                cudaMemcpyHostToDevice, m_stream);
    cudaMemsetAsync(d_brenner_scores_n, 0, sizeof(float) * num_plane, m_stream);

    shift_and_sum_kernel<<<grid, block, 0, m_stream>>>(d_images, d_volume_n,
    n_rows, n_cols, patch_size, 
    d_depth_n, num_plane, pixel_size, s, f);
    cudaError_t err_shift = cudaGetLastError();
    if(err_shift != cudaSuccess) printf("shift_Kernel error: %s\n", cudaGetErrorString(err_shift));
    cudaDeviceSynchronize();
    
    
    int m = 4;
    brenner_kernel<<<grid, block, 0, m_stream>>>(d_volume_n, d_brenner_scores_n, m, n_rows, n_cols, num_plane);
    cudaError_t err_brenner = cudaGetLastError();
    if(err_brenner != cudaSuccess) printf("brenner_Kernel error: %s\n", cudaGetErrorString(err_brenner));
    cudaDeviceSynchronize();
    
    std::vector<float> h_brenner_scores_n(num_plane);
    cudaMemcpyAsync(h_brenner_scores_n.data(), d_brenner_scores_n,
                sizeof(float) * num_plane,
                cudaMemcpyDeviceToHost, m_stream);
    cudaDeviceSynchronize(); 
    
    
    for(int i= 0; i < num_plane; i++)
        cout<<h_brenner_scores_n[i]<<","<<depth_zn[i]<<endl;
 
    float z_refined  = quadratic_fit_points(h_brenner_scores_n, depth_zn);
    
    
    return z_refined;
    
 
    
}
void GPUProcessor::generate_best_view(const float& z_best){
    dim3 block(16, 16);
    dim3 grid_1((n_cols+15)/16, (n_rows+15)/16, 1);
    cudaMemcpyAsync(d_depth_1, &z_best,
            sizeof(float) ,
            cudaMemcpyHostToDevice, m_stream);
    shift_and_sum_kernel<<<grid_1, block, 0, m_stream>>>(d_images, d_volume_1,
    n_rows, n_cols, patch_size, 
    d_depth_1, 1, pixel_size, s, f);
    cudaError_t err_shift_1 = cudaGetLastError();
    if(err_shift_1 != cudaSuccess) printf("shift_Kernel error: %s\n", cudaGetErrorString(err_shift_1));
    cudaDeviceSynchronize();

    std::vector<cv::Vec3f> h_volume(n_rows * n_cols);
    cudaMemcpyAsync(h_volume.data(), d_volume_1, h_volume.size() * sizeof(cv::Vec3f), cudaMemcpyDeviceToHost, m_stream);
   

  
   
    cv::Mat img(n_rows, n_cols, CV_32FC3, h_volume.data());
    
    cv::Mat img8, img8_large;
    
    // 转为 8-bit 显示
    img.convertTo(img8, CV_8UC3, 255.0);
    cv::resize(img8, img8_large, cv::Size(), 10.0 ,10.0,cv::INTER_NEAREST);
    // std::string filename = folder + "/output"+ std::to_string(i) +".png";
    // cv::imwrite(filename, img8_large);
    // cv::imwrite("shift_and_sum_original.bmp", img8);
    // cv::imwrite("shift_and_sum_larger.bmp", img8_large);
    // // // 显示
    cv::Point org((img8_large.cols - 200), 150);
    cv::putText(img8_large, to_string(z_best), org, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255,0,0), 1, 1);
    imshow("Volume Slice", img8_large);
    imshow("Volume Slice_original", img8);
    cv::waitKey(1);
}
float GPUProcessor::Equation_solving( vector<float>& y, const vector<float>& x){
     // 三元一次方程求 A,B,C
   
    for(auto &y_idx : y){
       if(fabs(y_idx) > 1e-6f)
            y_idx = 1 / y_idx;
        else
            y_idx = 0;
    }

    double A = ((y[2]-y[0]) - ((y[1]-y[0])*(x[2]-x[0])/(x[1]-x[0]))) /
               ((x[2]*x[2]-x[0]*x[0]) - ((x[1]*x[1]-x[0]*x[0])*(x[2]-x[0])/(x[1]-x[0])));
    double B = (y[1]-y[0] - A*(x[1]*x[1]-x[0]*x[0])) / (x[1]-x[0]);
    double C = y[0] - A*x[0]*x[0] - B*x[0];

    float z0 = -B / (2*A);          // 中心点
    float y0 = A*z0*z0 + B*z0 + C;  // 极值
    return z0;
}
vector<cv::Vec3f> GPUProcessor::currentimage(){
    int threads = 256;
    int total_threads = n_rows * n_cols ;
    int transform_blocks = (total_threads + threads - 1) / threads;
    transform_kernel_centerview<<<transform_blocks, threads, 0, m_stream>>>(
        d_imagefloat, image_rows, image_cols, 
        d_rows_flat, d_rows_offsets,
        total_circles, n_rows, n_cols, patch_size,
        d_image_1
    );
    cudaError_t err_transform_center = cudaGetLastError();
    if(err_transform_center != cudaSuccess) printf("transform_Kernel error: %s\n", cudaGetErrorString(err_transform_center));
    cudaDeviceSynchronize();
    std::vector<cv::Vec3f> h_image( n_rows * n_cols);
  
    cudaMemcpyAsync(h_image.data(), d_image_1,
    h_image.size() * sizeof(cv::Vec3f),
    cudaMemcpyDeviceToHost, m_stream);
    cudaDeviceSynchronize();

    return h_image;
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
void GPUProcessor::extract_rows(const vector<CircleInf>& circleList, const float y_tolerance){
    vector<CircleInf> sortedList;
    // preprocess(circles, sortedList);
    //sort by the y-axis
    sortedList = circleList;
    sort(sortedList.begin(), sortedList.end(),
    [](const CircleInf& a, const CircleInf& b) { return a.y < b.y; });
    int estimated_rows = max(1, (int)(sortedList.size()/30));
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
        // current_row.reserve(34);
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
    cout<<"rows size"<<rows.size()<<endl;
    cout<<"cols size"<<n_cols<<endl;
    
}
void GPUProcessor::cuda_preprocess(const cv::Mat& image_mla){
  
   
    size_t step = d_img.step; 
    
    CV_Assert(image_mla.isContinuous() && image_mla.type() == CV_8UC3);
    d_img.upload(image_mla);
    cudaError_t err_upload = cudaGetLastError();
    if(err_upload != cudaSuccess) printf("copy_Kernel error: %s\n", cudaGetErrorString(err_upload));
    uchar3* d_img_ptr = d_img.ptr<uchar3>();
    // size_t size_bytes = image_rows * image_cols * sizeof(uchar3);
    // cudaMemcpyAsync(d_img_ptr, image_mla.ptr<uchar3>(), size_bytes, cudaMemcpyHostToDevice, m_stream);
    
    
    dim3 block_copy(16, 16);
    dim3 grid_copy((image_cols + 15)/16, (image_rows + 15)/16);
    // 分配输出 device 内存（用 float*，回传时直接拷回到 Vec3f 数组）
    
    copy_kernel<<<grid_copy, block_copy, 0, m_stream>>>(d_img_ptr, image_rows, image_cols, step, d_imagefloat);
    cudaDeviceSynchronize();
    cudaError_t err_copy = cudaGetLastError();
    if(err_copy != cudaSuccess) printf("copy_Kernel error: %s\n", cudaGetErrorString(err_copy));
  
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