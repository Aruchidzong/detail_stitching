//
// Created by aruchid on 2019/12/8.
//

#ifndef IMAGEBASICS_ARUCHID_PRINTPIC_H
#define IMAGEBASICS_ARUCHID_PRINTPIC_H

#endif //IMAGEBASICS_ARUCHID_PRINTPIC_H

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/warpers.hpp"


#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

enum {
    left_top,
    left_bottom,
    right_top,
    right_bottom
};

//void CalcCorners(const Mat& H, const Mat& src, vector<Point2f> &corners);
Mat mywarpPerspective(Mat src,Mat T);
void Project_region_Get(Mat img,Mat H_trans,Mat &Translation_all,Size &OutputSize);

class Projection{
public:
    Projection();
    ~Projection();
    void CalcCorners(const Mat& H, const Mat& src, vector<Point2f> &corners);
    bool InfoStack(vector<Mat> Imgs,vector<Mat> Homo,int BestPOV);
    void Img_FieldTrans_Find();
    void Img_FieldTrans_Find(bool tar);
    Size Img_FieldScale_Find();
    void Homo_adjustment();
    Mat Pano_Projection();
    void Projection_forAll();
    void Store_the_Picture();

    int BestPOV;
    int PicNums_;
    Mat PanoResult;
    vector<Mat> Imgs;
    vector<Mat> Imgs_Projected;//投影以后的图片,扭转+黑边
    vector<Mat> Homo;//输入进来的两两之间的单应性矩阵
    vector<Mat> Homo_Projected;//相对于BestPOV 两两之间累加的单应性矩阵
    vector<Mat> Trans; //平移矩阵
    vector<Mat> mask;  //每张图片的mask
    vector<Size> SinglePrjSize;//
    vector<Point> CornersInPano;
    vector<vector<Point2f>> corners_single;
    Mat Translation_all;
    Size PanoSize;

};
