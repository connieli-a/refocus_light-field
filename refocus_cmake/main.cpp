#include "include/refocus.h"
#include <iostream>


int main(){  
   
    
    //Initalization
    cv::Mat image = cv::imread("data/Image__2025-05-23__16-37-19.bmp");
    cv::Mat image_mla = cv::imread("data/original_20250617_180038.bmp");
    cv::Mat image_rgb;
    cv::Mat image_gray;
    int Rmin = 9, Rmax = 30;
    

    if (image.empty()) {
        //throw runtime_error("No image, please check the source");
        cout << "no image, please check." << endl;
        return -1;
    }
    if (image_mla.empty()) {
        throw runtime_error("No image_mla, please check the source");
        cout << "no image, please check." << endl;
        return -1;
    }
    
    cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    cvtColor(image_mla, image_rgb, cv::COLOR_BGR2RGB);
    
    //hough transform
    vector<cv::Vec3f> circles;
    HoughCircles(image_gray, circles, cv::HOUGH_GRADIENT, 1.2, 28, 50, 30, Rmin, Rmax);
   
    if(!circles.empty()){

        int patch_size = 64;
        float tolerance = 15;
        int num_depth_plane = 1;
        float start = -5, end = 0;
        
        //create the disparity table
        std::vector<float> disparity_x_flat(patch_size * patch_size * num_depth_plane, 0.0f);
        std::vector<float> disparity_y_flat(patch_size * patch_size * num_depth_plane, 0.0f);
        // vector<vector<vector<float>>> disparity_x(patch_size, vector<vector<float>>(patch_size, vector<float>(num_depth_plane, 0.0f)));
        // vector<vector<vector<float>>> disparity_y(patch_size, vector<vector<float>>(patch_size, vector<float>(num_depth_plane, 0.0f)));
        generate_disparity_table(num_depth_plane, start, end, patch_size, disparity_x_flat, disparity_y_flat);
        
        //create an instance
        //default value-->device:0, cuda graph: true
        std:: shared_ptr<ImageProcessor> refocus_pointer = ImageProcessor::create(circles, tolerance, patch_size, num_depth_plane, disparity_x_flat, disparity_y_flat);
        auto start1 = chrono::high_resolution_clock::now();

        
        vector<cv::Vec3f> volume = refocus_pointer->imageprocess_cuda(image_mla);
        auto end1 = chrono::high_resolution_clock::now();
        int col = refocus_pointer->get_col();
        int row = refocus_pointer->get_row();
        cv::Mat img(row, col, CV_32FC3);
        
        for(int slice_idx = 0; slice_idx < num_depth_plane; slice_idx++){
            img.setTo(cv::Scalar(0,0,0));
            for (int i = 0; i < row; ++i) {
                    cv::Vec3f* ptr = img.ptr<cv::Vec3f>(i);
                    for (int j = 0; j < col; ++j) {
                            int idx = slice_idx * row * col + i * col + j; // slice_idx 可用于多深度平面
                            ptr[j] = volume[idx];
                
                        }
                    }
            // 转为 8-bit 显示
            cv::Mat img8;
            img.convertTo(img8, CV_8UC3, 255.0);
        
            // 显示
            imshow("Volume Slice", img8);
            cv::waitKey(0);
        }
        // cout<<volume[0] <<volume[1] <<volume[2] <<volume[3] <<volume[4]<<endl;
        std::cout << "all took " << chrono::duration<double, milli>(end1 - start1).count() << " ms" << endl;
        // for(int s = 0; s < patch_size ; s++){
        //     for(int t = 0; t < patch_size; t++){
        //         img.setTo(cv::Scalar(0,0,0));

        //         for (int u = 0; u < row; ++u) {
        //            cv::Vec3f* ptr = img.ptr<cv::Vec3f>(u);
        //            for (int v = 0; v < col; ++v) {
        //                int idx = (s * patch_size + t) * row * col + u * col + v; // slice_idx 可用于多深度平面
        //                ptr[v] = volume[idx];
                   
        //            }
        //        }
        //         cv::Mat img8;
        //        img.convertTo(img8, CV_8UC3, 255.0);
       
        //        // 显示
        //        imshow("Volume Slice", img8);
        //        cv::waitKey(0);
               
        //     }
        // }
        // for (int i=0;i<row;i++)
        //     for(int j=0;j<col;j++)
        //         img.ptr<cv::Vec3f>(i)[j] = volume[i*col+j];
        // cv::Mat img8;
        // img.convertTo(img8, CV_8UC3, 255.0);
        // 显示
        // imshow("Volume Slice", img8);
        // cv::waitKey(0);
        // cv::Mat img(row, col, CV_32FC3,volume.data());

        
        // cv::Mat img8;
        // img.convertTo(img8, CV_8UC3, 255.0);
        // cout<<"volume size:"<<volume.size()<<endl;
        // // 显示
        // imshow("Volume Slice", img8);
        // cv::waitKey(0);
        // cv::destroyAllWindows();

    }

    // ImageProcessor* proc = new CudaProcessor();

    // std::vector<float> input = {1,2,3,4,5};
    // std::vector<float> cpu_out, gpu_out;

    // proc->processCPU(input, cpu_out);
    // proc->processGPU(input, gpu_out);

    // std::cout << "CPU:";
    // for (auto v : cpu_out) std::cout << " " << v;
    // std::cout << "\n";

    // std::cout << "GPU:";
    // for (auto v : gpu_out) std::cout << " " << v;
    // std::cout << "\n";
    // delete proc;
    return 0;
}