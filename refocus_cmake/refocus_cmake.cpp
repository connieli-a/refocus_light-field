#include "refocus_cmake.h"
int main1(){
    int max_threads = omp_get_max_threads();
    cout<<max_threads<<endl;
     omp_set_num_threads(4);
    #pragma omp parallel for
    for (int i = 0; i < 8; ++i) {
        int tid = omp_get_thread_num();
        // printf("Thread %d is working on i = %d\n", omp_get_thread_num(), i);
        #pragma omp critical
        cout << "Thread " << tid << " is working on i = " << i << std::endl;
    }
    return 0;
}
int main()
{
    omp_set_num_threads(omp_get_max_threads());
    // cout << "filepath: " << std::filesystem::current_path().string() << std::endl;
    Mat image = imread("data/Image__2025-05-23__16-37-19.bmp");
    Mat image_mla = imread("data/original_20250617_180038.bmp");
    Mat image_rgb;
    Mat image_gray;
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
    
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
    cvtColor(image_mla, image_rgb, COLOR_BGR2RGB);
    
    //hough transform
    vector<Vec3f> circles;
    HoughCircles(image_gray, circles, HOUGH_GRADIENT, 1.2, 28, 50, 30, Rmin, Rmax);
    vector<CircleInf> circleList;
    
    
    if (!circles.empty()) {
        for (int i = 0; i < circles.size(); i++)
        {
            float x = std::round(circles[i][0] * 100) / 100.0f;
            float y = std::round(circles[i][1] * 100) / 100.0f;
            Point2f center(x, y);  
            float radius = round(circles[i][2] * 100) / 100.0f;
            circleList.push_back({ center, radius });
            
        }
        
        //sort by the y-axis
        vector<int> idx(circleList.size());
        for (int i = 0; i < idx.size(); i++) idx[i] = i;
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            return circleList[a].center.y < circleList[b].center.y;
        });
        vector<CircleInf> sortedList;
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
            float x = sortedList[i].center.x;
            float y = sortedList[i].center.y;
            
            if (x >= rangex1 && x <= rangex2 && y >= rangey1 && y <= rangey2) {
                rangeList.push_back(sortedList[i]);
            }
        }
        sortedList = move(rangeList);
        
        
        //prepared work
        //extract the patch image to the array
        int patch_size = 64;
        float tolerance = 15;
        int n_cols = 0;
        //rows: collect the circle centers and sort by the y axis and x axis
        vector<vector<CircleInf>> rows = extract_rows(sortedList, tolerance,  n_cols);
        int n_rows = static_cast<int>(rows.size());
        
        int num_depth_plane = 1;
        float start = -5, end = 0;
        //create the disparity table
        int mid_idx = patch_size / 2;
        float pixel_size = 2.0;
        int s = 125;//the diameter of the lens
        int f = 2500;// the focal length of the lens
        vector<float> depth_range;
        //generate depth_range
        if(num_depth_plane == 1)
            depth_range.push_back(start);
        else{
            float step = static_cast<float>(end - start) / static_cast<float>(num_depth_plane - 1);
            for (int i = 0; i < num_depth_plane; ++i)
            {
                /* code */
                depth_range.push_back(start + step * i);
            }
        }
        vector<vector<vector<float>>> disparity_x(patch_size, vector<vector<float>>(patch_size, vector<float>(num_depth_plane, 0.0f)));
        vector<vector<vector<float>>> disparity_y(patch_size, vector<vector<float>>(patch_size, vector<float>(num_depth_plane, 0.0f)));
        
        for(int z_idx = 0; z_idx < num_depth_plane; z_idx++){
            float z = depth_range[z_idx];
            float factor = s * (z / (f + z)) / pixel_size;
            for(int h = 0; h < patch_size; h++){
                for(int w = 0; w < patch_size; w++){
                    disparity_x[h][w][z_idx] = (w - mid_idx) * factor ;
                    disparity_y[h][w][z_idx] = (h - mid_idx) * factor ;
                }
            }
        }
        
        // //patches: cut each microlens images based on each circle center and collect them into the two dimension array
        // vector<vector<Mat>>patches = extract_microlens_patches(image_mla, patch_size, rows);
        // vector<vector<vector<vector<Vec3b>>>> vp_img_arr;
        auto start2 = chrono::high_resolution_clock::now();
        // vector<Mat> views;
        vector<Vec3f> images;
        // int patch_h, patch_w;
        //vp_img_arr: change the sequence from [n_rows, n_cols, patch_h,patch_w] to [patch_h,patch_w,n_rows,n_cols]
        //images: the images array of each angle
        // transpose_patch_array(patches, vp_img_arr, images, n_rows, n_cols, patch_h, patch_w);
        //    cout<<"111";
        transform_image(image_mla, patch_size, rows, images, n_rows, n_cols);
        auto end2 = chrono::high_resolution_clock::now();
        
        
        auto start3 = chrono::high_resolution_clock::now();
        //refocus processing optical parament
        //volume: refocused image array
        vector<Mat> volume = shift_and_sum(images, n_rows, n_cols, num_depth_plane, patch_size, disparity_x, disparity_y);
        auto end3 = chrono::high_resolution_clock::now();
        
        
        auto duration2 = chrono::duration<double,milli>(end2 - start2);
        auto duration3 = chrono::duration<double,milli>(end3 - start3);

           
        std::cout << "running time2：" << duration2.count() << " milliseconds" << endl;    
        std::cout << "running time3：" << duration3.count() << " milliseconds" << endl;    

        // auto inner_start = chrono::high_resolution_clock::now();
        // auto inner_end = chrono::high_resolution_clock::now();
        // auto inner_dur = duration_cast<chrono::nanoseconds>(inner_end - inner_start).count();
        // cout<<inner_dur<<endl;
        for (size_t i = 0; i < volume.size(); i++)
        {
                /* code */
            imshow("image",volume[i]);
            waitKey(0);
        }
        
   
    }
    
    

    return 0;

}

vector<vector<CircleInf>>extract_rows(vector<CircleInf> sortedList, float y_tolerance,  int& n_cols) {
    
    vector<vector<CircleInf>> rows;
    int estimated_rows = max(1, (int)(sortedList.size()/34));
    rows.reserve(estimated_rows);
    while (!sortedList.empty())
    {
        /* code */
        float row_y = sortedList[0].center.y;
        for (const auto& c : sortedList)
        {
            /* code */
            if (c.center.y < row_y)
                row_y = c.center.y;
        }
        vector<CircleInf>current_row;
        current_row.reserve(34);
        vector<CircleInf>remaining_row;
        remaining_row.reserve(sortedList.size());
        for (const auto& c : sortedList) {
            if (abs(row_y - c.center.y) < y_tolerance)
                current_row.push_back(c);
            else remaining_row.push_back(c);
        }
        //sort by x-axis
        sort(current_row.begin(), current_row.end(), [](const CircleInf& a, const CircleInf& b) {
            return a.center.x < b.center.x;
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
    // cout<<"rows size"<<rows.size()<<endl;
    return rows;
}

void transform_image(const Mat& image_mla, const int patch_size, const vector<vector<CircleInf>>& rows, vector<Vec3f>& images, const int n_rows, const int n_cols){
    auto start1 = chrono::high_resolution_clock::now();
    assert(patch_size > 0);
    float half = (patch_size - 1) / 2.0f;
    int patch_area = patch_size * patch_size;
    // auto start = chrono::high_resolution_clock::now();
    // // assert(views.size() == patch_size * patch_size);
    // for (int i = 0; i < views.size(); i++)
    // {
        //     views[i] = Mat::zeros(n_rows, n_cols, CV_32FC3);
        // }
    images.reserve(patch_area * n_rows * n_cols);
    // auto end = chrono::high_resolution_clock::now();
    // vector<Vec3f> views_data(patch_size * patch_size * n_rows * n_cols);
    vector<int> uv_list;
    for(int u = 0; u < n_rows; u++){
        for(int v = 0; v < n_cols; v++){
            uv_list.emplace_back(u * n_cols + v);
        }
    }
    auto end1 = chrono::high_resolution_clock::now();
    auto start2 = chrono::high_resolution_clock::now();
   
    #pragma omp parallel for 
    for (int idx = 0; idx < uv_list.size() * patch_area; idx++){
        int uv_idx = idx / (patch_area);
        int s = (idx / patch_size) % patch_size;
        int t = idx % patch_size;

        int u = uv_list[uv_idx] / n_cols;
        int v = uv_list[uv_idx] % n_cols;
        // int view_idx = s * patch_size + t;
        int idx_in_image = (s * patch_size + t) * n_rows * n_cols + uv_list[uv_idx];
        const auto& row = rows[u][v];
        bool valid = row.valid;
        if(!valid){
            images[idx_in_image] = Vec3f(0, 0, 0);  
            continue;
        }
        Point2f center = row.center;        
        float x = center.x + (t - half);
        float y = center.y + (s - half);
        Vec3f color = bilinear_rgb(image_mla, Point2f(x, y));
        images[idx_in_image] = color;
    }
    // for (int i = 0; i < uv_list.size(); i++)
    // {
    //     int u = uv_list[i].first;
    //     int v = uv_list[i].second;

    //     Point2f center = rows[u][v].center;
    //     bool valid = rows[u][v].valid;
    //     int uv_offset = u * n_cols + v;
    //     for (int s = 0; s < patch_size; s++)//patch_h
    //     {           
    //         for(int t = 0; t < patch_size; t++)//patch_w
    //         {
    //             int view_idx = s * patch_size + t;
    //             if(!valid){
    //                 images[view_idx * n_rows * n_cols + uv_offset] = Vec3f(0, 0, 0);  
    //                 continue;
    //             }
                
    //             Point2f sample_point = center + Point2f(t - half, s - half);
    //             Vec3f color = bilinear_rgb(image_mla, sample_point);
    //             images[view_idx * n_rows * n_cols + uv_offset] = color;
                
    //             // Vec3f* row_ptr = views[view_idx].ptr<Vec3f>(u);
    //             // row_ptr[v] = color;
    //         }
    //     }

    // }
        
        
    auto end2 = chrono::high_resolution_clock::now();
    // std::cout << "initialization took " << chrono::duration<double, milli>(end - start).count() << " ms" << endl;
    // std::cout << "prepared took " << chrono::duration<double, milli>(end1 - start1).count() << " ms" << endl;
    // std::cout << "whole loop took " << chrono::duration<double, milli>(end2 - start2).count() << " ms" << endl;
    
}
//the running time of bilinear_rgb: ~0.0002ms
Vec3b bilinear_rgb(const Mat& image_mla, const Point2f& pt){
    
    int x = static_cast<int>(floor(pt.x));
    int y = static_cast<int>(floor(pt.y));
    float dx = pt.x - x;
    float dy = pt.y - y;
    if(x < 0 || x + 1 >= image_mla.cols || y < 0 || y + 1 >= image_mla.rows)
    return Vec3b(0, 0, 0);
    const Vec3b* row0 = image_mla.ptr<Vec3b>(y);
    const Vec3b* row1 = image_mla.ptr<Vec3b>(y + 1);
    
    Vec3b I00(row0[x]);
    Vec3b I01(row0[x + 1]);
    Vec3b I10(row1[x]);
    Vec3b I11(row1[x + 1]);
    
    
    
    return (1 - dx) * (1 - dy) * I00 +
    dx * (1 - dy) * I01 +
    (1 - dx) * dy * I10 +
    dx * dy * I11;
    
}
vector<Mat> shift_and_sum(const vector<Vec3f> &images,  const int n_rows, const int n_cols, const int num_depth_plane,const int patch_size, const vector<vector<vector<float>>>& disparity_x, const vector<vector<vector<float>>>& disparity_y){
    auto start1 = chrono::high_resolution_clock::now();
    vector<Mat> volume(num_depth_plane);
    int patch_area = patch_size * patch_size;
    vector<Mat> views(patch_area);
    for (int idx = 0; idx < views.size() ; idx++)
    {
        views[idx] = Mat(n_rows, n_cols, CV_32FC3, (void*)(images.data() + idx * n_rows * n_cols));
    }
    vector<vector<Mat>> all_local_refocused(num_depth_plane, vector<Mat>(patch_area));
    // cout << "image type: " << images[0].type() << endl;
    // auto start2 = chrono::high_resolution_clock::now();
    #pragma omp parallel for 
    for (int index = 0; index < num_depth_plane * patch_area; index++)
    {   
        // /* code */
        int idx = index / num_depth_plane;
        int z_idx = index % num_depth_plane;
        int h = idx / patch_size;
        int w = idx % patch_size;
        
        float dx = disparity_x[h][w][z_idx];
        float dy = disparity_y[h][w][z_idx];
        Mat& dst = all_local_refocused[z_idx][idx];
        dst.create(n_rows, n_cols, CV_32FC3);
        shift_img(views[idx], dx, dy, dst);

        // Mat shifted(views[idx].size(), CV_32FC3);
        // shift_img(views[idx], dx, dy, shifted);
        // all_local_refocused[z_idx][idx] = shifted;
        
        // #pragma omp parallel for collapse(2)
        // for (int h = 0; h < patch_size; h++)
        // {
            //     /* code */
            //     for (int w = 0; w < patch_size; w++)
            //     {
                //         /* code */
                //         // Mat img_float;
                //         // images[h][w].convertTo(img_float, CV_32FC3);
                //         int idx = h * patch_size + w;
                //         float dx = disparity_x[h][w][z_idx];
                //         float dy = disparity_y[h][w][z_idx];
                
                //         Mat shifted(images[idx].size(), CV_32FC3);
                //         shift_img(images[idx], dx, dy, shifted);
                //         local_refocused[idx] = shifted;
                
                
                //     }
                
                // }
                
    }   
    for(int z_idx = 0; z_idx < num_depth_plane; z_idx++){
        Mat refocused(n_rows, n_cols, CV_32FC3, Scalar(0,0,0));
        for(int idx = 0; idx < patch_area; idx++){
            refocused += all_local_refocused[z_idx][idx];
        }
        refocused /= static_cast<float>(patch_area);
        Mat refocused_unit8;
        refocused.convertTo(refocused_unit8, CV_8UC3, 1.0, 0.0);
        
        float scale = 10.0f;
        Mat large;
        cv::resize(refocused_unit8, large, Size(), scale, scale, INTER_LINEAR);
        volume[z_idx] = large;
    }
    auto end1 = chrono::high_resolution_clock::now();
    // auto end2 = chrono::high_resolution_clock::now();
   
    std::cout << "all took " << chrono::duration<double, milli>(end1 - start1).count() << " ms" << endl;
    // std::cout << "whole loop took " << chrono::duration<double, milli>(end2 - start2).count() << " ms" << endl;  
        
    
   
    return volume;
}
//the running time of the shift_img is ~0.0017ms
void shift_img(const Mat& img, float dx, float dy, Mat& shifted){
    

    //Affine transformation matrix
    Mat M = (Mat_<double>(2,3)<<1, 0, dx, 0 , 1, dy);
    warpAffine(img, shifted, M, img.size(), INTER_LINEAR, BORDER_REFLECT);
    
}
// vector<Mat> shift_and_sum(const vector<Mat> &images, const int& n_rows, const int& n_cols, const int& num_depth_plane,const int& patch_size, const vector<vector<vector<float>>>& disparity_x, const vector<vector<vector<float>>>& disparity_y){

    
//     vector<Mat> volume(num_depth_plane);
//     Mat shifted(images[0].size(), CV_32FC3);
//     cout << "image type: " << images[0].type() << endl;
//     for (int z_idx = 0; z_idx < num_depth_plane; z_idx++)
//     {   
//         /* code */
//         Mat refcoused(n_rows, n_cols, CV_32FC3, Scalar(0,0,0));
//         for (int h = 0; h < patch_size; h++)
//         {
//             /* code */
//             for (int w = 0; w < patch_size; w++)
//             {
//                 /* code */
//                 // Mat img_float;
//                 // images[h][w].convertTo(img_float, CV_32FC3);
//                 float dx = disparity_x[h][w][z_idx];
//                 float dy = disparity_y[h][w][z_idx];
                
//                 shift_img(images[h * patch_size + w], dx, dy, shifted);
                
//                 refcoused += shifted;
                
//             }
            
//         }
//         refcoused /= static_cast<float>(patch_size * patch_size);
//         Mat refocused_unit8;
//         refcoused.convertTo(refocused_unit8, CV_8UC3, 1.0, 0.0);
        
//         float scale = 10.0f;
//         Mat large;
//         cv::resize(refocused_unit8, large, Size(), scale, scale, INTER_LINEAR);
//         volume[z_idx] = large;
//     }
   
//     return volume;
// }
// vector<vector<Mat>>extract_microlens_patches(Mat image_mla, int patch_size, vector<vector<CircleInf>>rows){
    //     size_t n_rows = rows.size();
    //     int n_cols = 0;
//     for(const auto& row : rows){
//         n_cols = max(n_cols, (int)row.size());
//     }
//     vector<vector<Mat>> patches(n_rows,vector<Mat>(n_cols));//定义patches数组
    
    
//     for (int i = 0; i < n_rows; i++)
//     {
//         /* code */
//         for (int j = 0; j < (int)rows[i].size(); j++)
//         {
//             /* code */
//             patches[i][j] = extract_patch(image_mla, rows[i][j].center, patch_size) ;
//         }

//     }
//     return patches;
// }




// Mat extract_patch(const Mat& image, const Point2f& center, int patch_size){
//     static int half = patch_size / 2;
//     static Mat patch;
//     static Mat map_x(patch_size, patch_size, CV_32FC1);
//     static Mat map_y(patch_size, patch_size, CV_32FC1);

//     for (int y = 0; y < patch_size; y++)
//     {
//         /* code */
//         for (int x = 0; x < patch_size; x++)
//         {
//             /* code */
//             static float xf = center.x - half + 0.5f + x;
//             static float yf = center.y - half + 0.5f + y;
//             map_x.at<float>(y, x) = xf;
//             map_y.at<float>(y, x) = yf;
//         }

//     }
//     remap(image, patch, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
//     return patch;
// }

// void transpose_patch_array(const vector<vector<Mat>>& patches, vector<vector<vector<vector<Vec3b>>>> &vp_img_arr, vector<vector<Mat>> &images, int &n_rows, int &n_cols, int &patch_h, int &patch_w){
//     n_rows = static_cast<int>(patches.size());
//     n_cols = 0;
//     for(const auto& row : patches){
//         n_cols = max(n_cols, (int)row.size());
//     }
//     if (n_rows == 0 || n_cols == 0 || patches[0][0].empty()) {
//         cout << "Invalid patch array!" << endl;
//         return;
//     }
//     patch_h = patches[0][0].rows;
//     patch_w = patches[0][0].cols;

//     //set the zeros array vp_img_arr
//     vp_img_arr = vector<vector<vector<vector<Vec3b>>>>(patch_h, 
//         vector<vector<vector<Vec3b>>>(patch_w, vector<vector<Vec3b>>
//             (n_rows, vector<Vec3b>(n_cols))));
//     for(int u = 0; u < n_rows; u++){
//         for(int v = 0; v < patches[u].size(); v++){
//             const Mat& patch = patches[u][v];
//             // cout<<patches[i].size();
//             // cout<<"i:"<< i<<"j:"<<j<<endl;
//             if(patch.empty() || patch.rows != patch_h || patch.cols != patch_w ||
//             patch.channels() != 3){
//                 cout<<"skip invaild patch at ["<<u<<"]["<<v<<"]"<<endl;
//                 continue;
//             }
            
//             for(int s = 0 ; s < patch_h; s++){
//                 for(int t = 0; t < patch_w; t++){
                 
//                     vp_img_arr[s][t][u][v] = patch.at<Vec3b>(s,t);
//                 }
//             }
         
//         }
//     }

//     //convert vp_img_arr to whole mat corresponding on each [u][v]
//     images = vector<vector<Mat>>(patch_h, vector<Mat>(patch_w));
//     for(int s = 0; s < patch_h; s++){
//         for (int t = 0; t < patch_w; t++)
//         {
//             /* code */
//             //initialization
//             // Mat img(n_rows, n_cols, CV_8UC3, Scalar(0,0,0));
//             Mat img(n_rows, n_cols, CV_32FC3, Scalar(0,0,0));
//             for (int u = 0; u < n_rows; u++)
//             {
//                 /* code */
//                 for(int v = 0; v < n_cols; v++){
//                     if(v < patches[u].size() && !patches[u][v].empty() && patches[u][v].channels() == 3){
//                         // img.at<Vec3b>(u,v) = vp_img_arr[s][t][u][v];
//                         img.at<Vec3f>(u, v) = Vec3f(
//                         vp_img_arr[s][t][u][v][0],
//                         vp_img_arr[s][t][u][v][1],
//                         vp_img_arr[s][t][u][v][2]
//                         );
//                     }

//                 }
//             }
//             images[s][t] = img;
//         }
         
//     }

// }

// void shift_img_implace(const Mat& img, float dx, float dy, Mat& out){
//     // out.setTo(Scalar(0,0,0));
//     int n_rows = img.rows;
//     int n_cols = img.cols;
//     for(int i = 0; i < n_rows; i++){
//         for(int j = 0; j < n_cols; j++){
//             float x = j + dx;
//             float y = i + dy;
//             Vec3f color = bilinear_rgb(img, Point2f(x, y));
//             Vec3f* row_ptr = out.ptr<Vec3f>(i);
//             row_ptr[j] = color;
//         }
//     }
// }

