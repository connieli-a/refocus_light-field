#include "refocus_cmake.h"

using namespace std;
using namespace cv;

struct CircleInf
{
    /* data */
    Point2f center;
    float radius;
};

vector<vector<CircleInf>>extract_rows(vector<CircleInf> sortedList, float y_tolerance);
vector<vector<Mat>>extract_microlens_patches(Mat image_mla, int patch_size, vector<vector<CircleInf>>rows);
Mat extract_patch(const Mat& image, const Point2f& center, int patch_size);
void transpose_patch_array(const vector<vector<Mat>>& patches, vector<vector<vector<vector<Vec3b>>>> &vp_img_arr, vector<vector<Mat>> &img_arr, int &n_rows, int &n_cols, int &patch_h, int &patch_w);
// vector<Mat> shift_and_sum(const vector<vector<Mat>> &images, vector<vector<vector<float>>> & disparity_x, vector<vector<vector<float>>> & disparity_y, const vector<float>& depth_range );
vector<Mat> shift_and_sum(const vector<vector<Mat>> &images, const int& n_rows, const int& n_cols, const int& num_depth_plane, const float& start, const float& end);
Mat shift_img(const Mat& img, float dx, float dy);

int main()
{
    cout << "filepath: " << std::filesystem::current_path().string() << std::endl;
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
        
        
        
        //extract the patch image to the array
        int patch_size = 64;
        float tolerance = 15;
        //rows: collect the circle centers and sort by the y axis and x axis
        vector<vector<CircleInf>> rows = extract_rows(sortedList, tolerance);
        //patches: cut each microlens images based on each circle center and collect them into the two dimension array
        vector<vector<Mat>>patches = extract_microlens_patches(image_mla, patch_size, rows);
        vector<vector<vector<vector<Vec3b>>>> vp_img_arr;
        vector<vector<Mat>> images;
        int n_rows, n_cols, patch_h, patch_w;
        //vp_img_arr: change the sequence from [n_rows, n_cols, patch_h,patch_w] to [patch_h,patch_w,n_rows,n_cols]
        //images: the images array of each angle
        transpose_patch_array(patches, vp_img_arr, images, n_rows, n_cols, patch_h, patch_w);
        
        //refocus processing optical parament
        int num_depth_plane = 1;
        //volume: refocused image array
        float start = -5, end = 0;
        vector<Mat> volume = shift_and_sum(images, n_rows, n_cols, num_depth_plane, start, end);
        auto start1 = chrono::high_resolution_clock::now();
        auto end1 = chrono::high_resolution_clock::now(); 
        auto duration1 = duration_cast<chrono::milliseconds>(end1 - start1);
        cout << "running time：" << duration1.count() << " milliseconds" << endl;
       
       
        // for (size_t i = 0; i < volume.size(); i++)
        // {
        //     /* code */
        //     imshow("image",volume[i]);
        //     waitKey(0);
        // }
        
    }


    return 0;

}

vector<vector<CircleInf>>extract_rows(vector<CircleInf> sortedList, float y_tolerance) {
    
    vector<vector<CircleInf>> rows;
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
        vector<CircleInf>remaining_row;
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
        sortedList = move(remaining_row);

    }
     
    return rows;
}

vector<vector<Mat>>extract_microlens_patches(Mat image_mla, int patch_size, vector<vector<CircleInf>>rows){
    size_t n_rows = rows.size();
    int n_cols = 0;
    for(const auto& row : rows){
        n_cols = max(n_cols, (int)row.size());
    }
    vector<vector<Mat>> patches(n_rows,vector<Mat>(n_cols));//定义patches数组
    for (int i = 0; i < n_rows; i++)
    {
        /* code */
        for (int j = 0; j < (int)rows[i].size(); j++)
        {
            /* code */
            patches[i][j] = extract_patch(image_mla, rows[i][j].center, patch_size) ;
        }

    }
    return patches;
}
Mat extract_patch(const Mat& image, const Point2f& center, int patch_size){
    int half = patch_size / 2;
    Mat patch;
    Mat map_x(patch_size, patch_size, CV_32FC1);
    Mat map_y(patch_size, patch_size, CV_32FC1);

    for (int y = 0; y < patch_size; y++)
    {
        /* code */
        for (int x = 0; x < patch_size; x++)
        {
            /* code */
            float xf = center.x - half + 0.5f + x;
            float yf = center.y - half + 0.5f + y;
            map_x.at<float>(y, x) = xf;
            map_y.at<float>(y, x) = yf;
        }

    }
    remap(image, patch, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
    return patch;
}

void transpose_patch_array(const vector<vector<Mat>>& patches, vector<vector<vector<vector<Vec3b>>>> &vp_img_arr, vector<vector<Mat>> &images, int &n_rows, int &n_cols, int &patch_h, int &patch_w){
    n_rows = static_cast<int>(patches.size());
    n_cols = 0;
    for(const auto& row : patches){
        n_cols = max(n_cols, (int)row.size());
    }
    if (n_rows == 0 || n_cols == 0 || patches[0][0].empty()) {
        cout << "Invalid patch array!" << endl;
        return;
    }
    patch_h = patches[0][0].rows;
    patch_w = patches[0][0].cols;

    //set the zeros array vp_img_arr
    vp_img_arr = vector<vector<vector<vector<Vec3b>>>>(patch_h, 
        vector<vector<vector<Vec3b>>>(patch_w, vector<vector<Vec3b>>
            (n_rows, vector<Vec3b>(n_cols))));
    for(int u = 0; u < n_rows; u++){
        for(int v = 0; v < patches[u].size(); v++){
            const Mat& patch = patches[u][v];
            // cout<<patches[i].size();
            // cout<<"i:"<< i<<"j:"<<j<<endl;
            if(patch.empty() || patch.rows != patch_h || patch.cols != patch_w ||
            patch.channels() != 3){
                cout<<"skip invaild patch at ["<<u<<"]["<<v<<"]"<<endl;
                continue;
            }
            
            for(int s = 0 ; s < patch_h; s++){
                for(int t = 0; t < patch_w; t++){
                 
                    vp_img_arr[s][t][u][v] = patch.at<Vec3b>(s,t);
                }
            }
         
        }
    }

    //convert vp_img_arr to whole mat corresponding on each [u][v]
    images = vector<vector<Mat>>(patch_h, vector<Mat>(patch_w));
    for(int s = 0; s < patch_h; s++){
        for (int t = 0; t < patch_w; t++)
        {
            /* code */
            //initialization
            // Mat img(n_rows, n_cols, CV_8UC3, Scalar(0,0,0));
            Mat img(n_rows, n_cols, CV_32FC3, Scalar(0,0,0));
            for (int u = 0; u < n_rows; u++)
            {
                /* code */
                for(int v = 0; v < n_cols; v++){
                    if(v < patches[u].size() && !patches[u][v].empty() && patches[u][v].channels() == 3){
                        // img.at<Vec3b>(u,v) = vp_img_arr[s][t][u][v];
                        img.at<Vec3f>(u, v) = Vec3f(
                        vp_img_arr[s][t][u][v][0],
                        vp_img_arr[s][t][u][v][1],
                        vp_img_arr[s][t][u][v][2]
                        );
                    }

                }
            }
            images[s][t] = img;
        }
         
    }

}

vector<Mat> shift_and_sum(const vector<vector<Mat>> &images, const int& n_rows, const int& n_cols, const int& num_depth_plane, const float& start, const float& end){

    int mid_idx = static_cast<int>(images.size() / 2);
    float pixel_size = 2.0;
    int s = 125;//the diameter of the lens
    int f = 2500;// the focal length of the lens
    int patch_h = static_cast<int>(images.size());
    int patch_w = static_cast<int>((patch_h > 0 && !images[0].empty()) ? images[0].size() : 0);
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
    vector<vector<vector<float>>> disparity_x(patch_h, vector<vector<float>>(patch_w, vector<float>(num_depth_plane, 0.0f)));
    vector<vector<vector<float>>> disparity_y(patch_h, vector<vector<float>>(patch_w, vector<float>(num_depth_plane, 0.0f)));
    
    for(int z_idx = 0; z_idx < num_depth_plane; z_idx++){
        float z = depth_range[z_idx];
        float factor = s * (z / (f + z)) / pixel_size;
        for(int h = 0; h < patch_h; h++){
            for(int w = 0; w < patch_w; w++){
                disparity_x[h][w][z_idx] = (w - mid_idx) * factor ;
                disparity_y[h][w][z_idx] = (h - mid_idx) * factor ;
            }
        }
    }
    vector<Mat> volume(num_depth_plane);
   
    for (int z_idx = 0; z_idx < num_depth_plane; z_idx++)
    {   
        /* code */
        Mat refcoused(n_rows, n_cols, CV_32FC3, Scalar(0,0,0));
        for (int h = 0; h < patch_h; h++)
        {
            /* code */
            for (int w = 0; w < patch_w; w++)
            {
                /* code */
                // Mat img_float;
                // images[h][w].convertTo(img_float, CV_32FC3);
                float dx = disparity_x[h][w][z_idx];
                float dy = disparity_y[h][w][z_idx];
                
                Mat shifted = shift_img(images[h][w], dx,dy);
                refcoused += shifted;
                 
            }
            
        }
        refcoused /= static_cast<float>(patch_h * patch_w);
        Mat refocused_unit8;
        refcoused.convertTo(refocused_unit8, CV_8UC3, 1.0, 0.0);
        
        float scale = 10.0f;
        Mat large;
        resize(refocused_unit8, large, Size(), scale, scale, INTER_LINEAR);
        volume[z_idx] = large;
    }
     
    
    
    return volume;
}
Mat shift_img(const Mat& img, float dx, float dy){
    
    Mat shifted;
    //Affine transformation matrix
    Mat M = (Mat_<double>(2,3)<<1, 0, dx, 0 , 1, dy);
    warpAffine(img, shifted, M, img.size(), INTER_LINEAR, BORDER_REFLECT);
   
    return shifted;

    
}

