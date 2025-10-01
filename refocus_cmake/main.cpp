#include "include/refocus.h"
#include <iostream>




int main(){  
   
    
    //Initalization
    cv::Mat image_rgb;
    cv::Mat image_gray;
    int Rmin = 15, Rmax = 35;
    //set the range
    int rangex1 = 900, rangex2 = 3000;//960 2880
    int rangey1 = 200, rangey2 = 1800;//540 1620
    cv::Rect roi(rangex1, rangey1, (rangex2 - rangex1), (rangey2 - rangey1));
    int patch_size = 64;
    float tolerance = 15;
    int num_depth_plane = 5;//fixed
    float start = -5, end = 5;
    int screenWidth = 1536; // 屏幕宽
    int screenHeight = 864; // 屏幕高


    vector<float> depth_range;
    float step = static_cast<float>(end - start) / static_cast<float>(num_depth_plane - 1);
    if(num_depth_plane == 1)
        depth_range.push_back(start);
    else{
        for (int i = 0; i < num_depth_plane; ++i)
        {
            depth_range.push_back(start + step * i);
        }
    }
    
    cv::Mat image = cv::imread("data/Image_2025-09-16.bmp");//be cutted
    cv::Mat image_mla;
    cv::Mat image_mla_roi;
    if (image.empty()) {
        //throw runtime_error("No image, please check the source");
        cout << "no image, please check." << endl;
        return -1;
    }
    cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);
    // cv::Mat image_gray_roi = image_gray(roi);
    //hough transform
    vector<cv::Vec3f> circles;
    HoughCircles(image_gray, circles, cv::HOUGH_GRADIENT, 1.2, 29, 100, 30, Rmin, Rmax);


    if(!circles.empty()){
        
        
        // //create the disparity table
        // std::vector<float> disparity_x_flat(patch_size * patch_size * num_depth_plane, 0.0f);
        // std::vector<float> disparity_y_flat(patch_size * patch_size * num_depth_plane, 0.0f);
        // // vector<vector<vector<float>>> disparity_x(patch_size, vector<vector<float>>(patch_size, vector<float>(num_depth_plane, 0.0f)));
        // // vector<vector<vector<float>>> disparity_y(patch_size, vector<vector<float>>(patch_size, vector<float>(num_depth_plane, 0.0f)));
        // generate_disparity_table(num_depth_plane, start, end, patch_size, disparity_x_flat, disparity_y_flat);
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

        //     cv::circle(image, center, radius, cv::Scalar(0, 255, 0), 2); // 绿色圆
        //     cv::circle(image, center, 2, cv::Scalar(0, 0, 255), -1);     // 红色圆心
        // }
        // cv::imshow("circle", image);
        //create an instance
        //default value-->device:0, cuda graph: true
        std:: shared_ptr<ImageProcessor> refocus_pointer = ImageProcessor::create(circleList, tolerance, patch_size, rangey2 - rangey1, rangex2 - rangex1);
        int col = refocus_pointer->get_col();
        int row = refocus_pointer->get_row();
        cv::Mat img(row, col, CV_32FC3);
        // step /= 2;
        PylonInitialize();
        try{
             //open camera
            CInstantCamera camera(CTlFactory::GetInstance().CreateFirstDevice());
            camera.Open();
            
            INodeMap& nodemap = camera.GetNodeMap();
            //exposure time
            CFloatPtr exposureTime(nodemap.GetNode("ExposureTime"));
            if(IsWritable(exposureTime))
            exposureTime->SetValue(7000.0);//µs
            CIntegerPtr width(nodemap.GetNode("Width"));
            CIntegerPtr height(nodemap.GetNode("Height"));
            CIntegerPtr offsetx(nodemap.GetNode("OffsetX"));
            CIntegerPtr offsety(nodemap.GetNode("OffsetY"));
            CEnumerationPtr balancesector(nodemap.GetNode("BalanceWhiteAuto"));
            CEnumerationPtr balanceRatioSelector(nodemap.GetNode("BalanceRatioSelector"));
            CFloatPtr balanceRatio(nodemap.GetNode("BalanceRatio"));
            if(IsWritable(width)) width->SetValue(rangex2 - rangex1);
            if(IsWritable(height)) height->SetValue(rangey2 - rangey1);
            if(IsWritable(offsetx)) offsetx->SetValue(rangex1);
            if(IsWritable(offsety)) offsety->SetValue(rangey1);
            if(IsWritable(balancesector)) balancesector->FromString("Off");
            if(IsWritable(balanceRatioSelector)) balanceRatioSelector->FromString("Red");
            if(IsWritable(balanceRatio)) balanceRatio->SetValue(1.0);
            if(IsWritable(balanceRatioSelector)) balanceRatioSelector->FromString("Green");
            if(IsWritable(balanceRatio)) balanceRatio->SetValue(1.0);
            if(IsWritable(balanceRatioSelector)) balanceRatioSelector->FromString("Blue");
            if(IsWritable(balanceRatio)) balanceRatio->SetValue(1.0);

            camera.StartGrabbing(GrabStrategy_OneByOne);
            // camera.StartGrabbing(30, GrabStrategy_OneByOne);
            CGrabResultPtr ptrGrabResult;
            CPylonImage pylonImage;
            CImageFormatConverter converter;
            converter.OutputPixelFormat = PixelType_BGR8packed;
            while(camera.IsGrabbing()){
                camera.RetrieveResult(INFINITE, ptrGrabResult, TimeoutHandling_ThrowException);
                if(ptrGrabResult->GrabSucceeded()){
                  
                    auto start = std::chrono::high_resolution_clock::now();
                    converter.Convert(pylonImage, ptrGrabResult);
                    
                    cv::Mat frame(static_cast<int>(ptrGrabResult->GetHeight()), static_cast<int>(ptrGrabResult->GetWidth()), CV_8UC3, (uint8_t*)pylonImage.GetBuffer());
                   
                    double scaleX = screenWidth / double(frame.cols);
                    double scaleY = screenHeight / double(frame.rows);
                    double scale = min(scaleX, scaleY); // 保证完整显示
                    cv::Mat display;
                    
                    cv::resize(frame, display, cv::Size(), scale, scale); // 使用统一缩放比例
                    cv::namedWindow("Camera", cv::WINDOW_NORMAL);
                    cv::imshow("Camera", display);
                    // imwrite("Camera.bmp",display);
                    if (cv::waitKey(1) == 27) // ESC 退出
                    break;

                    // image_mla = cv::imread("data/original_20250617_180038.bmp");
                    // image_mla_roi = image_mla(roi).clone();
                    image_mla_roi = frame.clone();
                    if (image_mla_roi.empty()) {
                        std::cerr << "ROI is empty!" << std::endl;
                    }
                    // vector<cv::Vec3f> volume = refocus_pointer->imageprocess_cuda(image_mla_roi);
                    // cout <<"ROI type: " << image_mla_roi.type() << "ROI: " << roi << "  Image size_cpp: " 
                    // << image_mla_roi.cols << "x" << image_mla_roi.rows << endl;
        
                    float z0 = refocus_pointer->imageprocess_cuda(image_mla_roi, depth_range);
                    cout<<"best postion value:" << z0 <<endl;
                    vector<cv::Vec3f> volume = refocus_pointer->currentimage();
                    for (int u = 0; u < row; ++u) {
                        cv::Vec3f* ptr = img.ptr<cv::Vec3f>(u);
                        for (int v = 0; v < col; ++v) {
                            int idx =  u * col + v; 
                            ptr[v] = volume[idx];
                            
                        }
                    }
                    cv::Mat img8, img8_large;
                    img.convertTo(img8, CV_8UC3, 255.0);
                    cv::resize(img8, img8_large, cv::Size(), 10.0 ,10.0, cv::INTER_NEAREST);
                    // 显示
                    imshow("central view ", img8);
                    imshow("central view_large", img8_large);
                    // imwrite("current imagez0.bmp",img8);
                    // imwrite("current imagez0_large.bmp", img8_large);
                    // if (cv::waitKey(1) == 27) // ESC 退出
                    // break;
                    
                    //the part of automatic stage control
                    //code
                    
                    depth_range.assign({z0 - 2 * step, z0 - step, z0, z0 + step, z0 + 2 * step});
                   


                    auto end = chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                    cout<<"the running time of imageprocess_cuda + image_display :"<< elapsed.count()<<"ms"<<endl;
                    int interval_ms = 100;  
                    if(elapsed.count() < interval_ms){
                        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms - elapsed.count()));
                    }
                  

                }else{
                    cout<<"can not catch the camera successfully"<<endl;
                }
            }
            camera.StopGrabbing();
            camera.Close();
        }
        catch(const GenericException &e){
            std::cerr<<"An exception occurued:"<<e.GetDescription()<<std::endl;
        } 
        PylonTerminate();   
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