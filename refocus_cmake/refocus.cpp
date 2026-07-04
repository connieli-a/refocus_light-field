#include "include/refocus.h"
#include "include/cuda_kernel.cuh"

#include <iostream>
#include <limits>
#include <cmath>

//  * @param device 使用するGPUのID (デフォルトは0)
//  * @param useGraph CUDA Graphを使用するかどうか (デフォルトはtrue)
//  * @return ImageProcessorのインスタンス
//  default mode(use gpu)->useGraph: true
std::shared_ptr<ImageProcessor> ImageProcessor::create(const vector<CircleInf>& circleList, const float y_tolerance, const int patch_size_cpp,  const int image_rows, const int image_cols, const int32_t device, const bool useGraph){
     //only 1 gpu device
    // int device_count = 0;
//     cudaGetDeviceCount(&device_count);
//     std::cout << "Detected " << device_count << " CUDA devices." << std::endl;
//     if (device < 0 || device >= device_count) {
//     std::cerr << "Error: invalid GPU device id " << device 
//               << ", available range is 0 to " << device_count - 1 << std::endl;
//     exit(EXIT_FAILURE);
// }

    cudaSetDevice(device);
    
    // int current_device;
    // cudaGetDevice(&current_device);
    // std::cout<<"using gpu"<<current_device<<std::endl;
    if(useGraph){
        return std::make_shared<GPUProcessor>(circleList, y_tolerance, patch_size_cpp,  image_rows, image_cols);
    }
   return nullptr;
}


void gpuThread(std::shared_ptr<ImageProcessor> refocus_pointer){
    while(running){
        std::unique_lock<mutex> lock(frameBuffer.mtx);
        frameBuffer.cv.wait(lock, []{return frameBuffer.newFrame || !running;});
        if(!running)
            break;
        frameBuffer.newFrame = false;
        lock.unlock();
        // cout<<"run the image process"<<endl;
        // auto start = chrono::high_resolution_clock::now();
        cv::Mat* image_mla = frameBuffer.readBuf.load();
        refocus_pointer->imageprocess_cuda(*image_mla);
        //  auto end = chrono::high_resolution_clock::now();
        // auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // cout<<"show process"<< elapsed.count()<<"microseconds"<<endl;
       
    }
   
}

void showThread(std::shared_ptr<ImageProcessor> refocus_pointer){
    while(running){
        std::unique_lock<std::mutex> lock(resultBuffer.mtx);
        resultBuffer.cv.wait(lock, []{return resultBuffer.newResult || !running; });
        if(!running) break;
        resultBuffer.newResult = false;
        lock.unlock();
        auto start = chrono::high_resolution_clock::now();
        // cout<<"run the show process"<<endl;
        auto* rBuf = resultBuffer.readBuf.load();
        refocus_pointer->show_image(*rBuf);
        auto end = chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        cout<<"s: "<< elapsed.count()<<"us"<<endl;
    }
}
void GPUProcessor::show_image(const ResultData& result_frame){
    
    std::vector<float> localFrame = result_frame.best_image;
    float z_best = result_frame.best_value;
    std::vector<float> centerFrame = result_frame.center_img;
    cv::Mat img(n_rows, n_cols, CV_32FC1, localFrame.data());
    cv::Mat center_img(n_rows, n_cols, CV_32FC1, centerFrame.data());

    cv::Mat img8, img8_large, center_img8, center_img8_large;
    // cv::Mat img_count = img.clone();
    // 转为 8-bit 显示
    img.convertTo(img8, CV_8UC1, 255.0);
    cv::resize(img8, img8_large, cv::Size(), 10.0 ,10.0,cv::INTER_NEAREST);
    center_img.convertTo(center_img8, CV_8UC1, 255.0);
    cv::resize(center_img8, center_img8_large, cv::Size(), 10.0 ,10.0,cv::INTER_NEAREST);
    auto center_coordinate = get2d_coordinate(center_img8, 220);
    if(std::isnan(center_coordinate.first) || std::isnan(center_coordinate.second))
        std::cout<<"coordinate is NA."<<std::endl;

    // std::string filename = folder + "/output"+ std::to_string(i) +".png";
    // cv::imwrite(filename, img8_large);
    // cv::imwrite("shift_and_sum_original.bmp", img8);
    // cv::imwrite("shift_and_sum_larger.bmp", img8_large);
    // // // 显示
    cv::Point org((img8_large.cols - 180), 50);
    cv::Point org_1((img8_large.cols - 180), 150);
    
    float displacement = 790.13 * (1 - z_best)/400 *100;
    // float displacement = 790.13 * (1 - z_best)/400 *1000;
    cv::putText(img8_large, to_string(displacement), org, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255), 1, 1);
    cv::putText(img8_large, to_string(z_best), org_1, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255), 1, 1);
    cv::putText(center_img8_large, to_string(center_coordinate.first * 10.f), org, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0), 1, 1);
    cv::putText(center_img8_large, to_string(center_coordinate.second * 10.f), org_1, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255), 1, 1);
    cv::Mat center_color;
    cv::cvtColor(center_img8_large, center_color, cv::COLOR_GRAY2BGR);
    cv::drawMarker(center_color,     
        cv::Point(
        cvRound(center_coordinate.first * 10.0f),
        cvRound(center_coordinate.second * 10.0f)
    ), cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 18, 1);

    {
        std::lock_guard<std::mutex> dl(displayBuffer.mtx);
        displayBuffer.img = img8.clone(); // clone 减少共享内存竞态
        displayBuffer.img_large = img8_large.clone();
        displayBuffer.img_center = center_color.clone();
        displayBuffer.hasNew = true;

    }

}
void cameraThread(){
    try{
        //open camera
        Pylon::CInstantCamera camera(Pylon::CTlFactory::GetInstance().CreateFirstDevice());
        camera.Open();
        
        INodeMap& nodemap = camera.GetNodeMap();
        //exposure time
        CFloatPtr exposureTime(nodemap.GetNode("ExposureTime"));
        if(IsWritable(exposureTime))
        exposureTime->SetValue(2000.0);//µs 7000
        CIntegerPtr width(nodemap.GetNode("Width"));
        CIntegerPtr height(nodemap.GetNode("Height"));
        CIntegerPtr offsetx(nodemap.GetNode("OffsetX"));
        CIntegerPtr offsety(nodemap.GetNode("OffsetY"));
        //color camera
        // CEnumerationPtr balancesector(nodemap.GetNode("BalanceWhiteAuto"));
        // CEnumerationPtr balanceRatioSelector(nodemap.GetNode("BalanceRatioSelector"));
        // CFloatPtr balanceRatio(nodemap.GetNode("BalanceRatio"));
        if(IsWritable(width)) width->SetValue(rangex2 - rangex1);
        if(IsWritable(height)) height->SetValue(rangey2 - rangey1);
        if(IsWritable(offsetx)) offsetx->SetValue(rangex1);
        if(IsWritable(offsety)) offsety->SetValue(rangey1);
        // if(IsWritable(balancesector)) balancesector->FromString("Off");
        // if(IsWritable(balanceRatioSelector)) balanceRatioSelector->FromString("Red");
        // if(IsWritable(balanceRatio)) balanceRatio->SetValue(1.0);
        // if(IsWritable(balanceRatioSelector)) balanceRatioSelector->FromString("Green");
        // if(IsWritable(balanceRatio)) balanceRatio->SetValue(1.0);
        // if(IsWritable(balanceRatioSelector)) balanceRatioSelector->FromString("Blue");
        // if(IsWritable(balanceRatio)) balanceRatio->SetValue(1.5);

        camera.StartGrabbing(Pylon::GrabStrategy_OneByOne);
        // camera.StartGrabbing(200, Pylon::GrabStrategy_OneByOne);
        Pylon::CGrabResultPtr ptrGrabResult;
        Pylon::CPylonImage pylonImage;
        Pylon::CImageFormatConverter converter;
        converter.OutputPixelFormat = Pylon::PixelType_Mono8;
        while(running && camera.IsGrabbing()){
            // cout<<"run the camera grabbing process"<<endl;
            auto start = chrono::high_resolution_clock::now();
            camera.RetrieveResult(INFINITE, ptrGrabResult, Pylon::TimeoutHandling_ThrowException);
            if(ptrGrabResult->GrabSucceeded()){
                
                
                converter.Convert(pylonImage, ptrGrabResult);
                cv::Mat frame(static_cast<int>(ptrGrabResult->GetHeight()), static_cast<int>(ptrGrabResult->GetWidth()), CV_8UC1, (uint8_t*)pylonImage.GetBuffer());
                
                // double scaleX = screenWidth / double(frame.cols);
                // double scaleY = screenHeight / double(frame.rows);
                // double scale = min(scaleX, scaleY); // 保证完整显示



                
                // cv::Mat display;
                
                // cv::resize(frame, display, cv::Size(), scale, scale); // 使用统一缩放比例
                // cv::namedWindow("Camera", cv::WINDOW_NORMAL);
                // cv::imshow("Camera", frame);
                // cv::waitKey(1);
                // imwrite("Camera.bmp",display);
                
                // image_mla = cv::imread("data/original_20250617_180038.bmp");
                // image_mla_roi = image_mla(roi).clone();
                
                image_mla_roi = frame.clone();
               
                if (image_mla_roi.empty()) {
                    std::cerr << "ROI is empty!" << std::endl;
                }
                cv::Mat* buf = frameBuffer.writeBuf.load();
                image_mla_roi.copyTo(*buf);
                //exchange the reading and writing buffer
                cv::Mat* oldRead = frameBuffer.readBuf.load();
                frameBuffer.readBuf.store(buf);
                frameBuffer.writeBuf.store(oldRead);

                frameBuffer.newFrame = true;
                frameBuffer.cv.notify_one();

                {
                    std::lock_guard<std::mutex> lk(cameraDisplayBuffer.mtx);
                    cameraDisplayBuffer.fullframe = frame.clone();
                    cameraDisplayBuffer.hasNew = true;
                }
            }
            auto end = chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            cout<<"c: "<< elapsed.count()<<"us"<<endl;
        }
        running = false;
        camera.StopGrabbing();
        camera.Close();
    } catch(const Pylon::GenericException &e){
        std::cerr<<"An exception occurued:"<<e.GetDescription()<<std::endl;
    } 
}

std::pair<float, float> get2d_coordinate(const cv::Mat& center_image, int black_threshold){
    if(center_image.empty())
     return {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()
     };
     cv::Mat black_mask;
     cv::threshold(center_image, black_mask, black_threshold, 255, cv::THRESH_BINARY_INV);

     cv::Mat labels, stats, centroids;
     int n_labels = cv::connectedComponentsWithStats(black_mask, labels, stats, centroids, 8, CV_32S);

     int best_label = -1;
     int best_area = 0;
     for(int label = 1; label< n_labels; label++){
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if(area > best_area){
            best_area = area;
            best_label = label;
        }
     }
     if(best_label < 0){
        return  {
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::quiet_NaN()
        };
    }
    return {
        static_cast<float>(centroids.at<double>(best_label, 0)),
        static_cast<float>(centroids.at<double>(best_label, 1))
    };
}
