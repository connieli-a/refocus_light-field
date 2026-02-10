#include "include/refocus.h"
#include <iostream>
#include "include/ADControl.h"
#include "include/DAControl.h"


int main(){  
   
    // auto start = chrono::high_resolution_clock::now();
    //Initalization
  
    
    int Rmin = 4, Rmax = 10;//Rmin 15 Rmax 35
    
    cv::Mat image_gray;
    // cv::Rect roi(rangex1, rangey1, (rangex2 - rangex1), (rangey2 - rangey1));
    int patch_size = 18;//64
    float tolerance = 4;//15
    int num_depth_plane = 5;
    int screenWidth = 1536; // 屏幕宽
    int screenHeight = 864; // 屏幕高

   
    cv::Mat image = cv::imread("data/Image__2025-11-12__17-27-20.bmp", cv::IMREAD_UNCHANGED);//be cutted
    cv::Mat image_mla;
   
    
    if (image.empty()) {
        throw runtime_error("No image, please check the source");
        // cout << "no image, please check." << endl;
        return -1;
    }
    if(image.type() != CV_8UC1){
        cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    }else{
        image_gray = image;
    }
    
   
    //hough transform
    
    vector<cv::Vec3f> circles;
    HoughCircles(image_gray, circles, cv::HOUGH_GRADIENT, 1.1, 9, 100, 20, Rmin, Rmax);
    // HoughCircles(image_gray, circles, cv::HOUGH_GRADIENT, 1.2, 29, 100, 30, Rmin, Rmax)；
    // da.outputVoltage(5);
    if(!circles.empty()){
      
        //orgnaize the array
        vector<CircleInf> circleList;
        for (int i = 0; i < circles.size(); i++)
        {
            float x = std::round(circles[i][0] * 100) / 100.0f;
            float y = std::round(circles[i][1] * 100) / 100.0f;
            float radius = std::round(circles[i][2] * 100) / 100.0f;

          
            if(x - radius < 0.0f || y - radius < 0.0f || x + radius >= image_gray.cols || y + radius >= image_gray.rows)
                continue;
            circleList.push_back({ x, y, radius });
            
        }

        // for (const auto& c : circleList)
        // {
        //     cv::Point center(cvRound(c.x), cvRound(c.y));
        //     int radius = cvRound(c.radius);

        //     cv::circle(image_gray, center, radius, cv::Scalar(0, 255, 0), 1); // 绿色圆
        //     cv::circle(image_gray, center, 2, cv::Scalar(0, 0, 255), -1);     // 红色圆心
        // }
        // cv::imshow("circle", image_gray);
        // cv::waitKey(0);
        //create an instance
        std:: shared_ptr<ImageProcessor> refocus_pointer = ImageProcessor::create(circleList, tolerance, patch_size, rangey2 - rangey1, rangex2 - rangex1);
        int col = refocus_pointer->get_col();
        int row = refocus_pointer->get_row();
       
   
        Pylon::PylonInitialize();
       
        std::thread t_gpu(gpuThread, refocus_pointer);
        std::thread t_cam(cameraThread);
        std::thread t_show(showThread, refocus_pointer);
        SetThreadPriority(t_gpu.native_handle(), THREAD_PRIORITY_HIGHEST);
        SetThreadPriority(t_cam.native_handle(), THREAD_PRIORITY_TIME_CRITICAL);
        SetThreadPriority(t_show.native_handle(), THREAD_PRIORITY_BELOW_NORMAL);

          
        while(running ){
            bool has = false;
            cv::Mat frameshow;
            cv::Mat frameshow_large;
            cv::Mat camimage;
            {
                std::lock_guard<std::mutex> dl(displayBuffer.mtx);
                if(displayBuffer.hasNew){
                    frameshow_large = std::move(displayBuffer.img_large);
                    frameshow = std::move(displayBuffer.img);
                    displayBuffer.hasNew = false;
                    has = true;
                   
                }
            }
            
            if(has){
                if(!frameshow_large.empty()){
                    cv::imshow("Volume Slice", frameshow_large);
                }
                if(!frameshow.empty()){
                    cv::imshow("Volume Slice_original", frameshow);
                }
            }
            {
                std::lock_guard<std::mutex> cl(cameraDisplayBuffer.mtx);
                if(cameraDisplayBuffer.hasNew){
                    camimage = cameraDisplayBuffer.fullframe.clone();
                    cameraDisplayBuffer.hasNew = false;
                }
            }
            if(!camimage.empty()){
                cv::imshow("Camera show", camimage);
            }

            if (cv::waitKey(1) == 27) // ESC 退出
            {
                running = false;
                break;
            }
            
           
           
        }
     
    
        
                
    
      
        frameBuffer.cv.notify_all();
        resultBuffer.cv.notify_all();
        t_gpu.join();
        t_cam.join();
        t_show.join();
          
        Pylon::PylonTerminate();   
        cv::destroyAllWindows();
        // auto end = chrono::high_resolution_clock::now();
        // auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        // cout<<"10 times the running time of imageprocess_cuda + image_display :"<< elapsed.count()<<"microseconds"<<endl;
            //realtime read the image
            // image_mla = cv::imread("data/original_20250617_180038.bmp");
            
            // int type = image_mla.type();
            // int depth = type & CV_MAT_DEPTH_MASK;      // 0 → CV_8U
            // int channels = 1 + (type >> CV_CN_SHIFT); 
            // cout<<"type: "<<type<<"depth :"<<depth<<" channels :"<<channels<<endl;
            // if (image_mla.empty()) {
            //     throw runtime_error("No image_mla, please check the source");
            //     cout << "no image, please check." << endl;
            //     return -1;
            // }
            

        // for(int idx = 0; idx < num_depth_plane; idx++){
        //     cout<<depth_range[idx]<<","<<brenner[idx]<<endl;
        // }
        // int col = refocus_pointer->get_col();
        // int row = refocus_pointer->get_row();
        // cv::Mat img(row, col, CV_32FC3);
        
        // for(int slice_idx = 0; slice_idx < num_depth_plane; slice_idx++){
        //     img.setTo(cv::Scalar(0,0,0));
        //     for (int i = 0; i < row; ++i) {
        //             cv::Vec3f* ptr = img.ptr<cv::Vec3f>(i);
        //             for (int j = 0; j < col; ++j) {
        //                     int idx = slice_idx * row * col + i * col + j; // slice_idx 可用于多深度平面
        //                     ptr[j] = volume[idx];
                
        //                 }
        //             }
        //     // 转为 8-bit 显示
        //     cv::Mat img8;
        //     img.convertTo(img8, CV_8UC3, 255.0);
        
        //     // 显示
        //     imshow("Volume Slice", img8);
        //     cv::waitKey(0);
        // }
        // // cout<<volume[0] <<volume[1] <<volume[2] <<volume[3] <<volume[4]<<endl;
        // std::cout << "all took " << chrono::duration<double, milli>(end1 - start1).count() << " ms" << endl;
        // std::string folder = "out_images";
        //  if (!std::filesystem::exists(folder)) {
        //     std::filesystem::create_directory(folder);
        // }
        

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
        //         cv::Mat img8, img8_large;
        //        img.convertTo(img8, CV_8UC3, 255.0);
        //        cv::resize(img8, img8_large, cv::Size(), 10.0 ,10.0, cv::INTER_LINEAR);
        //     //    // 显示
        //     //    imshow("Volume Slice", img8_large);
        //     //    cv::waitKey(100);
        //         std::string savePath = folder + "/image_saved_"+std::to_string(s*patch_size+t) +".jpg";
        //         cv::imwrite(savePath, img8_large);
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

    }else{
        cout<<"no circle detected"<<endl;
    }

 
    return 0;
}