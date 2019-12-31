
#include "capture_stitching_mobile.hpp"

#include <iostream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
//#include "str_common.h"

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/util.hpp"

#include <numeric>
#include <time.h> //zt
#include <sys/time.h> //zt


#include <list>
#include <vector>
#include <opencv2/video/tracking.hpp>
#include <iomanip>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif


using namespace std;
using namespace cv;
using namespace cv::detail;





static float g_inlierConfThresh = 0.5;//0.9;  //匹配置信度
static int g_inlinerNum = 6;  //最小匹配内点数
static string g_featuresType = "sift";//"surf";//"gftt-sift";//"orb"; ;  //拼接使用特征
static float g_matchConfThresh = 0.3f;  //匹配最低置信度
static double g_scale = 0.25;
static Ptr<FeaturesFinder> g_finder; //特征计算指针
static vector<Mat> g_panoVec; //全景图list

static int g_inputImgNums = 0;

list<float> confidence_List(10);
vector<Point> corner_extern(4);
list<Point2d> move_extern(8);

typedef struct
{
    Mat inputImg;
    Mat img;
    ImageFeatures feature;
    float pitch;
    float yaw;
    float roll;
    Mat H;

}AngleImg;

std::vector <AngleImg> g_angleImg;

ofstream output_file;
std::string g_strLogPath;


list<cv::Point2f> origin_keypoints; //只是一个list存储所有的初始点
list< cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list

cv::Mat templateImg;
cv::Mat H_last_ = Mat::eye(3,3,CV_64FC1);
Mat H_optflow_ = Mat::eye(3,3,CV_64FC1);
int Max_origin_Size = 100;

///////

bool GFTTSiftFeaturesFind(InputArray image, ImageFeatures &features)
{

    int maxCorners = 0;
    double qualityLevel = 10e-4;
    double minDistance = 10e-3;
    int blockSize = 3;
    bool useHarrisDetector= true;
    double k = 0.04 ;
    int nLayers = 1;
    int maxKeypoints = 300;//10000;
    int nfeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;

    Ptr<GFTTDetector> gftt_ = GFTTDetector::create(maxCorners, qualityLevel, minDistance,
                                                   blockSize, useHarrisDetector, k);
    //features.img_size = image.size();
    if( !gftt_ )
    {
        std::cout << "OpenCV was built without GFTT support" << std::endl;
        return false;
    }
    Ptr<xfeatures2d::SIFT> sift_ = xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

    if( !sift_ )
    {
        std::cout << "OpenCV was built without SIFT support" << std::endl;
        return false;
    }

    Ptr<FeatureDetector> detector_;
    Ptr<DescriptorExtractor> extractor_;
//    Ptr<Feature2D> sift;

    detector_ = gftt_;
    extractor_ = sift_;
//    sift = sift_;

    int maxcorners = maxCorners;
    int maxkeypoint = maxKeypoints;
    int nlayers = nLayers;
    int blocksize = blockSize;
    double harris_k = k;

    UMat gray_image;
    if((image.type() != CV_8UC3) && (image.type() != CV_8UC1))
    {
        std::cout << "image type is not CV_8UC3 , CV_8UC1" << std::endl;

        return false;
    }
    if(image.type() == CV_8UC3)
    {
        cvtColor(image, gray_image, COLOR_BGR2GRAY);
    }
    else
    {
        //gray_image = image.getUMat(); //origin
        image.copyTo(gray_image);
    }

    if (nlayers == 1)
        detector_->detect(gray_image, features.keypoints);

    if(features.keypoints.size() < 10)
    {
        std::cout << "feature kp size < 10. " << std::endl;
        std::cout << "feature kp size = " << features.keypoints.size() << std::endl;
        return false;
    }

    KeyPointsFilter::retainBest(features.keypoints, maxkeypoint);
    extractor_->compute(gray_image, features.keypoints, features.descriptors);
//    extractor_->detectAndCompute(gray_image, cv::UMat(), features.keypoints, features.descriptors, true);
    std::cout << "finish descriptor. "  << std::endl;

    gftt_.release();
    sift_.release();
    detector_.release();
    extractor_.release();
    return true;
}

bool capture_stitching_init(double _scale, std::string & _strLogPath)
{
    g_scale = _scale;
    g_strLogPath = _strLogPath;
    if(_scale > 1 || _scale <= 0)
        g_scale = 0.25;

    if(!g_strLogPath.empty()) {
        output_file.open(g_strLogPath, ios::out);
    }

    //finder
    if (g_featuresType == "surf")
    {
        g_finder = makePtr<SurfFeaturesFinder>(200); //5000
    }
    else if (g_featuresType == "sift")
    {
        g_finder = makePtr<SiftFeaturesFinder>(0, 3, 0.01, 50, 1.6);
    }
    else if (g_featuresType == "gftt-sift")
    {
        //finder = makePtr<GFTTSiftFeaturesFinder>(0, 10e-4, 10e-3, 3, true, 0.04, 1, 10000, 0); //origin
        g_finder = makePtr<GFTTSiftFeaturesFinder>(0, 10e-4, 10e-3, 3, true, 0.04, 1, 10000, 0); //zt
    }
    else if (g_featuresType == "akaze")
    {
        g_finder = makePtr<AKAZEFeaturesFinder>();
    }
    else if (g_featuresType == "orb")
    {
        g_finder = makePtr<OrbFeaturesFinder>(cv::Size(1,1), 10000, 1.3f, 1);
    }
    else
    {
        cout << "Unknown 2D features type: '" << g_featuresType << "'.\n";
        return false;
    }

    //warm up
    cv::Mat zeromap = cv::Mat::zeros(100, 100, CV_8UC3);
    ImageFeatures feature;
    (*g_finder)(zeromap, feature);
    g_finder->collectGarbage();
    g_panoVec.clear();
    //opticalFlow
    origin_keypoints.clear();
    keypoints.clear();
    return true;
}

void delete_pic()
{
    g_angleImg.pop_back();
    g_panoVec.pop_back();
    g_inputImgNums--;
}

void CalcCorners(const Mat& H, const Mat& src, vector<Point2f> &corners)
{
    Point2f left_top, left_bottom, right_top, right_bottom;

    double v2[] = { 0, 0, 1 };//左上角
    double v1[3];//变换后的坐标值
    Mat V2 = Mat(3, 1, CV_64FC1, v2); //列向量
    Mat V1 = Mat(3, 1, CV_64FC1, v1); //列向量

    V1 = H * V2;
//左上角(0,0,1)
//    cout << "V2: " << V2 << endl;
    // cout << "V1: " << v1[2] << endl;
    left_top.x = v1[0] / v1[2];
    left_top.y = v1[1] / v1[2];
    corners.push_back(left_top);

//左下角(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2); //列向量
    V1 = Mat(3, 1, CV_64FC1, v1); //列向量
    V1 = H * V2;
    left_bottom.x = v1[0] / v1[2];
    left_bottom.y = v1[1] / v1[2];
    corners.push_back(left_bottom);

//右上角(src.cols,0,1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2); //列向量
    V1 = Mat(3, 1, CV_64FC1, v1); //列向量
    V1 = H * V2;
    right_top.x = v1[0] / v1[2];
    right_top.y = v1[1] / v1[2];
    corners.push_back(right_top);

//右下角(src.cols,src.rows,1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = Mat(3, 1, CV_64FC1, v2); //列向量
    V1 = Mat(3, 1, CV_64FC1, v1); //列向量
    V1 = H * V2;
    right_bottom.x = v1[0] / v1[2];
    right_bottom.y = v1[1] / v1[2];
    corners.push_back(right_bottom);
}

void mywarpPerspective0(Mat src,Mat &dst,Mat T) {

    //此处注意计算模型的坐标系与Mat的不同

    //图像以左上点为（0,0），向左为x轴，向下为y轴，所以前期搜索到的特征点 存的格式是（图像x，图像y）---（rows，cols）

    //而Mat矩阵的是向下为x轴，向左为y轴，所以存的方向为（图像y，图像x）----（cols，rows）----（width，height）

    //这个是计算的时候容易弄混的

    //创建原图的四个顶点的3*4矩阵（此处我的顺序为左上，右上，左下，右下）

    Mat tmp(3, 4, CV_64FC1, 1);
    tmp.at < double >(0, 0) = 0;
    tmp.at < double >(1, 0) = 0;
    tmp.at < double >(0, 1) = src.cols;
    tmp.at < double >(1, 1) = 0;
    tmp.at < double >(0, 2) = 0;
    tmp.at < double >(1, 2) = src.rows;
    tmp.at < double >(0, 3) = src.cols;
    tmp.at < double >(1, 3) = src.rows;

    //获得原图四个顶点变换后的坐标，计算变换后的图像尺寸
    Mat corner = T * tmp;      //corner=(x,y)=(cols,rows)

    float lt_x = (float)corner.at < double >(0, 0) / corner.at < double >(2,0);
    float lt_y = (float)corner.at < double >(1, 0) / corner.at < double >(2,0);
    float rt_x = (float)corner.at < double >(0, 1) / corner.at < double >(2,1);
    float rt_y = (float)corner.at < double >(1, 1) / corner.at < double >(2,1);
    float lb_x = (float)corner.at < double >(0, 2) / corner.at < double >(2,2);
    float lb_y = (float)corner.at < double >(1, 2) / corner.at < double >(2,2);
    float rb_x = (float)corner.at < double >(0, 3) / corner.at < double >(2,3);
    float rb_y = (float)corner.at < double >(1, 3) / corner.at < double >(2,3);


//    std::cout << "lt_x = " << lt_x << ", lt_y = " << lt_y << std::endl;
//    std::cout << "lb_x = " << lb_x << ", lb_y = " << lb_y << std::endl;
//    std::cout << "rt_x = " << rt_x << ", rt_y = " << rt_y << std::endl;
//    std::cout << "rb_x = " << rb_x << ", rb_y = " << rb_y << std::endl;

    int width = 0, height = 0;
    double maxw = corner.at < double >(0, 0)/ corner.at < double >(2,0);
    double minw = corner.at < double >(0, 0)/ corner.at < double >(2,0);
    double maxh = corner.at < double >(1, 0)/ corner.at < double >(2,0);
    double minh = corner.at < double >(1, 0)/ corner.at < double >(2,0);

    lt_x = lt_x + maxw;
    lt_y = lt_y - maxh;
    lb_x = lb_x + maxw;
    lb_y = lb_y - maxh;
    rt_x = rt_x + maxw;
    rt_y = rt_y - maxh;
    rb_x = rb_x + maxw;
    rb_y = rb_y - maxh;

    //  std::cout << "maxw = " << maxw << ", minw = " << minw << ", maxh = " << maxh << ", minh = " << minh << std::endl;
    for (int i = 1; i < 4; i++) {

        maxw = max(maxw, corner.at < double >(0, i) / corner.at < double >(2, i));
        minw = min(minw, corner.at < double >(0, i) / corner.at < double >(2, i));
        maxh = max(maxh, corner.at < double >(1, i) / corner.at < double >(2, i));
        minh = min(minh, corner.at < double >(1, i) / corner.at < double >(2, i));
    }
    //创建向前映射矩阵 map_x, map_y

    //size(height,width)
    dst.create(int(maxh - minh), int(maxw - minw), src.type());
    // std::cout << "height = " << maxh - minh << ", width = " << maxw - minw << std::endl;
    Mat map_x(dst.size(), CV_32FC1);
    Mat map_y(dst.size(), CV_32FC1);

    Mat proj(3,1, CV_32FC1,1);
    Mat point(3,1, CV_32FC1,1);

    std::cout << "point.at<float>(0,0) = " << point.at<float>(0,0)<< std::endl;
    std::cout << "point.at<float>(0,1) = " << point.at<float>(0,1)<< std::endl;
    std::cout << "point.at<float>(0,2) = " << point.at<float>(0,2)<< std::endl;
    T.convertTo(T, CV_32FC1);

    //本句是为了令T与point同类型（同类型才可以相乘，否则报错，也可以使用T.convertTo(T, point.type() );）
    Mat Tinv = T.inv();

    struct timeval corner_start, corner_end;
    gettimeofday( &corner_start, NULL );

    ///opencv
//    for (phi = 0; phi < dsize.height; phi++)
//    {
//        double KKy = Kangle * phi;
//        double cp = std::cos(KKy);
//        double sp = std::sin(KKy);
//        float* mx = (float*)(mapx.data + phi*mapx.step);
//        float* my = (float*)(mapy.data + phi*mapy.step);
//
//        for (rho = 0; rho < dsize.width; rho++)
//        {
//            double x = bufRhos[rho] * cp + center.x;
//            double y = bufRhos[rho] * sp + center.y;
//
//            mx[rho] = (float)x;
//            my[rho] = (float)y;
//        }
//    }


    for (int i = 0; i < dst.rows; i++) {
        point.at<float>(1) = i + minh;
        for (int j = 0; j < dst.cols; j++) {
            point.at<float>(0) = j + minw;

            proj = Tinv * point;
            map_x.at<float>(i, j) = proj.at<float>(0) / proj.at<float>(2);
            map_y.at<float>(i, j) = proj.at<float>(1) / proj.at<float>(2);
        }
    }

//    for (int i = 0; i < dst.rows; i++) {
//        point.at<float>(1) = i + minh;
//        for (int j = 0; j < dst.cols; j++) {
//            point.at<float>(0) = j + minw;
//
//            proj = Tinv * point;
//            map_x.at<float>(i, j) = proj.at<float>(0) / proj.at<float>(2);
//            map_y.at<float>(i, j) = proj.at<float>(1) / proj.at<float>(2);
//        }
//    }

    gettimeofday( &corner_end, NULL );
    //求出两次时间的差值，单位为us
    int corner_timeuse = 1000000 * (corner_end.tv_sec - corner_start.tv_sec ) + corner_end.tv_usec - corner_start.tv_usec;
    printf("corner time: %d us\n", corner_timeuse);

    remap(src,dst,map_x,map_y, CV_INTER_LINEAR);

//    std::cout << "lt_x = " << lt_x + abs(rt_x) << ", lt_y = " << lt_y << std::endl;
//    std::cout << "lb_x = " << lb_x + abs(rt_x) << ", lb_y = " << lb_y << std::endl;
//    std::cout << "rt_x = " << rt_x + abs(rt_x) << ", rt_y = " << rt_y << std::endl;
//    std::cout << "rb_x = " << rb_x + abs(rt_x) << ", rb_y = " << rb_y << std::endl;
//
//    circle(dst, Point2f(lt_x, lt_y), 3, cv::Scalar(255, 255, 255), -1);
//    circle(dst, Point2f(lb_x, lb_y), 3, cv::Scalar(255, 255, 255), -1);
//    circle(dst, Point2f(rt_x, rt_y), 3, cv::Scalar(255, 255, 255), -1);
//    circle(dst, Point2f(rb_x, rb_y), 3, cv::Scalar(255, 255, 255), -1);

//    namedWindow("dst", 0);
//    imshow("dst", dst);
////    imwrite("dst.jpg", dst);
//    waitKey();
}

bool mywarpPerspective(Mat src, Mat &dst, Mat T, vector<Point2f> &corners) {

    //此处注意计算模型的坐标系与Mat的不同

    //图像以左上点为（0,0），向左为x轴，向下为y轴，所以前期搜索到的特征点 存的格式是（图像x，图像y）---（rows，cols）

    //而Mat矩阵的是向下为x轴，向左为y轴，所以存的方向为（图像y，图像x）----（cols，rows）----（width，height）

    //这个是计算的时候容易弄混的

    //创建原图的四个顶点的3*4矩阵（此处我的顺序为左上，右上，左下，右下）

    Mat tmp(3, 4, CV_64FC1, 1);
    tmp.at < double >(0, 0) = 0;
    tmp.at < double >(1, 0) = 0;
    tmp.at < double >(0, 1) = src.cols;
    tmp.at < double >(1, 1) = 0;
    tmp.at < double >(0, 2) = 0;
    tmp.at < double >(1, 2) = src.rows;
    tmp.at < double >(0, 3) = src.cols;
    tmp.at < double >(1, 3) = src.rows;

    //获得原图四个顶点变换后的坐标，计算变换后的图像尺寸
    Mat corner = T * tmp;  //corner=(x,y)=(cols,rows)

    float lt_x = (float)corner.at < double >(0, 0) / corner.at < double >(2,0);
    float lt_y = (float)corner.at < double >(1, 0) / corner.at < double >(2,0);
    float rt_x = (float)corner.at < double >(0, 1) / corner.at < double >(2,1);
    float rt_y = (float)corner.at < double >(1, 1) / corner.at < double >(2,1);
    float lb_x = (float)corner.at < double >(0, 2) / corner.at < double >(2,2);
    float lb_y = (float)corner.at < double >(1, 2) / corner.at < double >(2,2);
    float rb_x = (float)corner.at < double >(0, 3) / corner.at < double >(2,3);
    float rb_y = (float)corner.at < double >(1, 3) / corner.at < double >(2,3);

//    std::cout << "lt_x = " << lt_x << ", lt_y = " << lt_y << std::endl;
//    std::cout << "lb_x = " << lb_x << ", lb_y = " << lb_y << std::endl;
//    std::cout << "rt_x = " << rt_x << ", rt_y = " << rt_y << std::endl;
//    std::cout << "rb_x = " << rb_x << ", rb_y = " << rb_y << std::endl;


    double maxw = corner.at < double >(0, 0)/ corner.at < double >(2,0);
    double minw = corner.at < double >(0, 0)/ corner.at < double >(2,0);
    double maxh = corner.at < double >(1, 0)/ corner.at < double >(2,0);
    double minh = corner.at < double >(1, 0)/ corner.at < double >(2,0);

    //std::cout << "maxw = " << maxw << ", minw = " << minw << ", maxh = " << maxh << ",minh = " << minh << std::endl;
    for (int i = 1; i < 4; i++) {

        maxw = max(maxw, corner.at < double >(0, i) / corner.at < double >(2, i));
        minw = min(minw, corner.at < double >(0, i) / corner.at < double >(2, i));
        maxh = max(maxh, corner.at < double >(1, i) / corner.at < double >(2, i));
        minh = min(minh, corner.at < double >(1, i) / corner.at < double >(2, i));
    }

    int width = 0, height = 0;
    //创建向前映射矩阵 map_x, map_y
    width = int(maxw - minw);
    height = int(maxh - minh);

    if(width < src.cols) width = src.cols;
    if(height < src.rows) height = src.rows;

    if(width > 5 * src.cols || height > 5 * src.rows)
    {
        return false;
    }

    cv::Mat tmpImg;
    tmpImg.create(height, width, src.type());
    tmpImg.setTo(0);

    src.copyTo(tmpImg(Rect(0,0,src.cols, src.rows)));

    vector<Point2f> src_point;
    src_point.push_back(Point2f(0, 0));
    src_point.push_back(Point2f(0, src.rows));
    src_point.push_back(Point2f(src.cols, 0));
    src_point.push_back(Point2f(src.cols, src.rows));

    Point2f point0, point1, point2, point3;

    point0.x = lt_x;
    point0.y = lt_y;
    point1.x = lb_x;
    point1.y = lb_y;
    point2.x = rt_x;
    point2.y = rt_y;
    point3.x = rb_x;
    point3.y = rb_y;

    corners.push_back(point0);
    corners.push_back(point1);
    corners.push_back(point2);
    corners.push_back(point3);

    Point tmp0, tmp1, tl;
    tmp0.x = min(point0.x, point1.x);
    tmp1.x = min(point2.x, point3.x);
    tl.x = min(tmp0.x, tmp1.x);

    tmp0.y = min(point0.y, point1.y);
    tmp1.y = min(point2.y, point3.y);
    tl.y = min(tmp0.y, tmp1.y);

    point0.x = point0.x - tl.x;
    point0.y = point0.y - tl.y;
    point1.x = point1.x - tl.x;
    point1.y = point1.y - tl.y;
    point2.x = point2.x - tl.x;
    point2.y = point2.y - tl.y;
    point3.x = point3.x - tl.x;
    point3.y = point3.y - tl.y;

    vector<Point2f> obj_point;
    obj_point.push_back(point0);  //左上
    obj_point.push_back(point1);  //左下
    obj_point.push_back(point2);  //右上
    obj_point.push_back(point3);  //右下
    Mat H_ = findHomography(src_point, obj_point, CV_RANSAC);
    warpPerspective(tmpImg, dst, H_, cv::Size(width, height));

    return true;
}

bool getBestPOV(std::vector <AngleImg> g_angleImg, int & bestPOV){
    vector<Mat> Homos;
    Homos.clear();

    for(int i = 0; i < g_angleImg.size(); ++i)
    {
        Homos.push_back(g_angleImg[i].H);
    }
    cv::Mat img0;
    g_angleImg[g_angleImg.size() - 1].inputImg.copyTo(img0);

    int PanoSize_area;
    double app_start_time = getTickCount();
    int Img_Numbers = Homos.size();
    int PicNums_ = Img_Numbers;

    for (int bestp = 0; bestp < Img_Numbers; ++bestp) {

        Mat tempEye = Mat::eye(3,3,CV_64F);
        vector<Mat> Homo_Projected;

        int BestPOV = bestp;
        for (int picnum = 0; picnum < PicNums_;) {
            //        cout <<  "turn = " << picnum << endl;
            if(picnum < BestPOV){
                Mat tempHomo;
                tempEye.copyTo(tempHomo);
                for (int step = picnum+1; step <= BestPOV ;) {
                    tempHomo = Homos[step].inv() * tempHomo;
                    step++;
                }
                Homo_Projected.push_back(tempHomo);
            }
            else if (picnum > BestPOV){

                Mat tempHomo;
                tempEye.copyTo(tempHomo);
                for (int step = BestPOV+1; step <= (picnum);) {
                    tempHomo =  tempHomo * Homos[step];
                    step++;
                }
                Homo_Projected.push_back(tempHomo);
            }
            else if (picnum == BestPOV){
                Mat tempHomo;//深拷贝问题
                tempEye.copyTo(tempHomo);
                Homo_Projected.push_back(tempHomo);
            }
            picnum++;
        }

        vector<Size> SinglePrjSize;//
        vector<Point> CornersInPano;
        vector<vector<Point2f>> corners(PicNums_);

        for (int i = 0; i < PicNums_; ++i) {

            vector<Point2f> tempcorners;
            CalcCorners(Homo_Projected[i], img0, corners[i]);

            //TODO CalcCorners 计算的是原始图片四个顶点在warp后的图像位置
            // 这里需要改成四个顶点的外接矩形坐标
            Point tmp0, tmp1, LT_position, RB_position;
            //LT_position 是该图片在全景图中的左上角位置坐标
            tmp0.x = min(corners[i][0].x, corners[i][1].x);
            tmp1.x = min(corners[i][2].x, corners[i][3].x);
            LT_position.x = min(tmp0.x, tmp1.x);

            tmp0.y = min(corners[i][0].y, corners[i][1].y);
            tmp1.y = min(corners[i][2].y, corners[i][3].y);
            LT_position.y = min(tmp0.y, tmp1.y);
            //SinglePrjSize 尺寸
            tmp0.x = max(corners[i][0].x, corners[i][1].x);
            tmp1.x = max(corners[i][2].x, corners[i][3].x);
            RB_position.x = max(tmp0.x, tmp1.x) - LT_position.x;

            tmp0.y = max(corners[i][0].y, corners[i][1].y);
            tmp1.y = max(corners[i][2].y, corners[i][3].y);
            RB_position.y = max(tmp0.y, tmp1.y) - LT_position.y;
            //CornersInPano 全局位置

            SinglePrjSize.push_back(Size(RB_position.x, RB_position.y));
            CornersInPano.push_back(LT_position);//corners.left_top;
        }

        Rect dst_roi = resultRoi(CornersInPano, SinglePrjSize);

        if (bestp == 0){
            bestPOV = 0;
            PanoSize_area = dst_roi.height * dst_roi.width;
        }
        else{
            int PanoSize_area_tmp = dst_roi.height*2 + dst_roi.width;
            if(PanoSize_area > PanoSize_area_tmp){
                PanoSize_area = PanoSize_area_tmp;
                bestPOV = bestp;
            }
        }
    }
    return true;
}

bool findHomo(ImageFeatures &src_features, ImageFeatures &dst_features, Mat & Homo){

    //FlannBasedMatcher matcher;
    BFMatcher matcher;
    vector<vector<DMatch> > matchePoints;
    vector<DMatch> GoodMatchePoints;

    cv::Mat src_des;
    src_features.descriptors.copyTo(src_des);
    vector<Mat> train_desc(1,src_des);
    matcher.add(train_desc);
    matcher.train();

    cv::Mat dst_des;
    dst_features.descriptors.copyTo(dst_des);
    matcher.knnMatch(dst_des, matchePoints, 2);

    cout << "total match points: " << matchePoints.size() << endl;

    // overlap comput
    std::vector<Point2f> obj_pt, scene_pt;  //重复区域匹配点list
    obj_pt.clear();
    scene_pt.clear();

    // Lowe's algorithm,获取优秀匹配点
    for (int i = 0; i < matchePoints.size(); i++)
    {
        if (matchePoints[i][0].distance < 0.75 * matchePoints[i][1].distance)
        {
            GoodMatchePoints.push_back(matchePoints[i][0]);
        }
    }

    cout << "Good match points: " << GoodMatchePoints.size() << std::endl;
    std::vector<KeyPoint> match_src_keypoints;
    match_src_keypoints.clear();
    std::vector<KeyPoint> match_dst_keypoints;
    match_dst_keypoints.clear();

    for(int k = 0; k < GoodMatchePoints.size(); ++k)
    {
        obj_pt.push_back(src_features.keypoints[GoodMatchePoints[k].trainIdx].pt );
        match_src_keypoints.push_back(src_features.keypoints[GoodMatchePoints[k].trainIdx]);
        scene_pt.push_back(dst_features.keypoints[GoodMatchePoints[k].queryIdx].pt);
        match_dst_keypoints.push_back(dst_features.keypoints[GoodMatchePoints[k].queryIdx]);
    }

//    if (obj_pt.size() < 50)
//    {
//        std::cout << "obj_pt.size() < 50. " << std::endl;
//        return false;
//    }

    std::vector <uchar> inliers_mask;
    Homo = findHomography(scene_pt, obj_pt, RANSAC, 4.0, inliers_mask, 500, 0.9999);

    int good_num = 0;
    for (int i = 0; i < inliers_mask.size();++i){
        if (inliers_mask[i] != '\0')
            good_num++;
    }

    float conf = good_num /(8 + 0.3 * (obj_pt.size()));
    if (good_num < 6 || conf < 0.5)
    {
        if(!g_strLogPath.empty())
            output_file << "good_num < 6 or conf < 0.5" << std::endl;
        else
            std::cout << "good_num < inliner_num or conf < conf_thresh" << std::endl;
        return false;
    }

    return true;
}

bool capture_pic(cv::Mat &src, cv::Mat &result, Point3f angle, int mode, int & bestPOV, double Homo[9])
{
    //删除最后一张图片返回倒数第二次的拼接图
    if (mode == 1 ){
        if (g_panoVec.size() > 0){
            delete_pic();
            if (g_panoVec.size() > 0){
                g_panoVec[g_panoVec.size() - 1].copyTo(result);
            }
            return true;
        }
        else
            return false;
    }

    struct timeval stitch_start, stitch_end{};
    gettimeofday( &stitch_start, NULL );

    if(!g_strLogPath.empty())
    {
        if (src.empty())
        {
            output_file << "input img is empty." << std::endl;
            return false;
        }
    }
    else
    {
        if (src.empty())
        {
            std::cout << "input img is empty." << std::endl;
            return false;
        }
    }

    //每次只能送一张进行stitching
    if(g_inputImgNums == 0)
        g_inputImgNums = 1; // 当前图片就1张

    if(g_inputImgNums == 1)  //first image , return current img
    {
        Mat img;
        resize(src, img, Size(), g_scale, g_scale, INTER_NEAREST);
        AngleImg firstAngleImg;
        firstAngleImg.pitch = angle.x;
        firstAngleImg.yaw = angle.y;
        firstAngleImg.roll = angle.z;

        src.copyTo(firstAngleImg.inputImg);
        img.copyTo(firstAngleImg.img);

        firstAngleImg.H = Mat::eye(3, 3, CV_64FC1);
        //find feature
        struct timeval first_find_start, first_find_end;
        gettimeofday( &first_find_start, NULL);

        (*g_finder)(firstAngleImg.img, firstAngleImg.feature); //origin

        gettimeofday( &first_find_end, NULL);

        //求出两次时间的差值，单位为us
        int firstfindtimeuse = 1000000 * ( first_find_end.tv_sec - first_find_start.tv_sec ) + first_find_end.tv_usec - first_find_start.tv_usec;
        if(!g_strLogPath.empty())
            output_file << "first img find time is " << firstfindtimeuse << "  us."<< std::endl;
        else
            printf("first img find time: %d us\n", firstfindtimeuse);

        g_panoVec.push_back(src); //save first img ,zt
        g_inputImgNums ++;
        g_angleImg.push_back(firstAngleImg);
        g_panoVec[g_panoVec.size() - 1].copyTo(result);

        Homo[0] = 1;
        Homo[1] = 0;
        Homo[2] = 0;
        Homo[3] = 0;
        Homo[4] = 1;
        Homo[5] = 0;
        Homo[6] = 0;
        Homo[7] = 0;
        Homo[8] = 1;

        bestPOV = 0;
        return true;
    }

    //find feature
    struct timeval find_start, find_end;
    gettimeofday( &find_start, NULL );
    //output_file << "into find Feature. " << std::endl;

    Mat img;
    resize(src, img, Size(), g_scale, g_scale, INTER_LINEAR);

    AngleImg angleImg;
    angleImg.pitch = angle.x;
    angleImg.yaw = angle.y;
    angleImg.roll = angle.z;

    src.copyTo(angleImg.inputImg);
    img.copyTo(angleImg.img);
    (*g_finder)(angleImg.img, angleImg.feature); //origin

    //output_file << "finish find Feature. " << std::endl;

    gettimeofday( &find_end, NULL );
    //求出两次时间的差值，单位为us
    int find_timeuse = 1000000 * ( find_end.tv_sec - find_start.tv_sec ) + find_end.tv_usec - find_start.tv_usec;
    if(!g_strLogPath.empty())
        output_file << "LOGINFO: Finder time is " << find_timeuse << " us."<< std::endl;
    else
        printf("Finder time: %d us\n", find_timeuse);

    //TODO 求采集区域并删除非采集区域的特征点
    // offset
//    float v_offset2 = (1 - cos(rad(MatVecs[1].pitch - MatVecs[0].pitch))) * img2.rows / 2;
//    float h_offset2 = (1 - cos(rad(MatVecs[1].yaw - MatVecs[0].yaw))) * img2.cols / 2;
//
//    float NewWidth = img2.cols - h_offset2 * 2;
//    float NewHeight = img2.rows - v_offset2 * 2;

//    std::cout << "h_offset2 = " << h_offset2 << std::endl;
//    std::cout << "v_offset2 = " << v_offset2 << std::endl;

    int kpSize = angleImg.feature.keypoints.size();

//    for(size_t i = 0; i < kpSize; ++i)
//    {
//       // if(angleImg.feature.keypoints[i].pt.x > )
//    }

    //得到新的特征点

    struct timeval homo_start, homo_end;
    gettimeofday( &homo_start, NULL );

    // 单应矩阵求解
    if(findHomo(g_angleImg[g_inputImgNums - 2].feature, angleImg.feature, angleImg.H))
    {
        //对单应矩阵里面的尺度因子做缩放
        angleImg.H.at<double>(0, 2) = angleImg.H.at<double>(0, 2) / g_scale;
        angleImg.H.at<double>(1, 2) = angleImg.H.at<double>(1, 2) / g_scale;
        angleImg.H.at<double>(2, 0) = angleImg.H.at<double>(2, 0) * g_scale;
        angleImg.H.at<double>(2, 1) = angleImg.H.at<double>(2, 1) * g_scale;

        Homo[0] = angleImg.H.at<double>(0, 0);
        Homo[1] = angleImg.H.at<double>(0, 1);
        Homo[2] = angleImg.H.at<double>(0, 2);
        Homo[3] = angleImg.H.at<double>(1, 0);
        Homo[4] = angleImg.H.at<double>(1, 1);
        Homo[5] = angleImg.H.at<double>(1, 2);
        Homo[6] = angleImg.H.at<double>(2, 0);
        Homo[7] = angleImg.H.at<double>(2, 1);
        Homo[8] = angleImg.H.at<double>(2, 2);

        if(abs(Homo[0]) >= 2 || abs(Homo[0]) <= 0.2 || abs(Homo[4]) >= 2 || abs(Homo[4]) <= 0.2)
        {
            if(!g_strLogPath.empty())
            {
                output_file << "homo abnormal." << std::endl;
                output_file << "Homo[0] = " << Homo[0] << ", Homo[4] = " << Homo[4] << std::endl;
            }
            else
                std::cout << "homo abnormal." << std::endl;
            return false;
        }

        g_angleImg.push_back(angleImg);

    } else
    {
        if(!g_strLogPath.empty())
            output_file << "Match failed." << std::endl;
        else
            std::cout << "Match failed." << std::endl;
        return false;
    }

    gettimeofday( &homo_end, NULL );
    //求出两次时间的差值，单位为us
    int homo_timeuse = 1000000 * ( homo_end.tv_sec - homo_start.tv_sec ) + homo_end.tv_usec - homo_start.tv_usec;
    if(!g_strLogPath.empty())
        output_file << "homo time is " << homo_timeuse << "  us." << std::endl;
    else
        printf("homo time: %d us\n", homo_timeuse);

    vector<Point> resultCorners(g_inputImgNums);
    vector<Size> sizes(g_inputImgNums);
    vector<vector <Point2f>> corners(g_inputImgNums);
    std::vector <cv::Mat> HomoVecs;
    std::vector <cv::Mat> NewHomoVecs;
    std::vector <cv::Mat> warped(g_inputImgNums);

    for(int i = 0;i < g_inputImgNums; ++i)
    {
        Mat H = Mat::eye(3, 3, CV_64FC1);
        NewHomoVecs.push_back(H);
    }

    // ref pic , 第 refIndx + 1 图 作为参考图像
    int refIndx = 0;

#ifndef BESTPOV
    getBestPOV(g_angleImg,refIndx);
#else
    if(g_inputImgNums % 2 == 0)
        refIndx = g_inputImgNums / 2 - 1;
    else
        refIndx = g_inputImgNums / 2;
#endif
    bestPOV = refIndx;

    // std::cout << "refIndx = " << refIndx << std::endl;

    struct timeval warp_start, warp_end;
    gettimeofday( &warp_start, NULL );

    for(int i = 0; i < g_inputImgNums; ++i)
    {
        // std::cout << "HomoVecs[i] = " << HomoVecs[i] << std::endl;
        if(i < refIndx)
        {
            Mat H_tmp = Mat::eye(3, 3, CV_64FC1);
            for(int k = i; k < refIndx; ++k){
                H_tmp = H_tmp * g_angleImg[k+1].H;
            }
            //HomoVecs[i] = H_tmp.inv();
            NewHomoVecs[i] = H_tmp.inv();
        }
        else if(i > refIndx)
        {
            Mat H_tmp = Mat::eye(3, 3, CV_64FC1);
            for(int k = refIndx; k < i; ++k){
//                    std::cout << "H_tmp = " << H_tmp << std::endl;
                H_tmp = H_tmp * g_angleImg[k+1].H;
            }
            NewHomoVecs[i] = H_tmp;
        } else
        {
            //HomoVecs[i] = Mat::eye(3, 3, CV_64FC1);
            NewHomoVecs[i] = Mat::eye(3, 3, CV_64FC1);
        }
        if(abs(NewHomoVecs[i].at<double>(0,0)) >= 2 || abs(NewHomoVecs[i].at<double>(0,0)) <= 0.2 || abs(NewHomoVecs[i].at<double>(1,1)) >= 2 || abs(NewHomoVecs[i].at<double>(1,1)) <= 0.2)
        {
            if(!g_strLogPath.empty())
            {
                output_file << "homo abnormal." << std::endl;
                output_file << "homo = " << NewHomoVecs[i] << std::endl;
            }
            else
                std::cout << "homo abnormal." << std::endl;
            return false;
        }
//        mywarpPerspective(g_angleImg[i].inputImg, warped[i], NewHomoVecs[i]);
        if(!mywarpPerspective(g_angleImg[i].inputImg, warped[i], NewHomoVecs[i], corners[i]))
        {
            if(!g_strLogPath.empty())
            {
                output_file << "warp failed. " << std::endl;
            }
            else
                std::cout << "warp failed. " << std::endl;

            return false;
        }

//        CalcCorners(NewHomoVecs[i], g_angleImg[i].inputImg, corners[i]);

        //TODO CalcCorners 计算的是原始图片四个顶点在warp后的图像位置， 这里需要改成四个顶点的外接矩形坐标
        Point tmp0, tmp1, tl;
        tmp0.x = min(corners[i][0].x, corners[i][1].x);
        tmp1.x = min(corners[i][2].x, corners[i][3].x);
        tl.x = min(tmp0.x, tmp1.x);

        tmp0.y = min(corners[i][0].y, corners[i][1].y);
        tmp1.y = min(corners[i][2].y, corners[i][3].y);
        tl.y = min(tmp0.y, tmp1.y);

        sizes[i] = warped[i].size();
        resultCorners[i] = tl;
    }

    gettimeofday( &warp_end, NULL );
    //求出两次时间的差值，单位为us
    int warp_timeuse = 1000000 * (warp_end.tv_sec - warp_start.tv_sec ) + warp_end.tv_usec - warp_start.tv_usec;
    printf("warp time: %d us\n", warp_timeuse);

    Rect dst_roi = resultRoi(resultCorners, sizes);
    // std::cout << "dst_roi size = " << dst_roi.size() << std::endl;
    Mat dst;
    dst.create(dst_roi.size(), CV_8UC3);
    dst.setTo(cv::Scalar::all(0));

    for(int i = 0; i < g_inputImgNums; ++i)
    {
        Mat gray;
        cvtColor(warped[i], gray, CV_BGR2GRAY);

        int dx = resultCorners[i].x - dst_roi.x;
        int dy = resultCorners[i].y - dst_roi.y;
        //std::cout << "dx = " << dx << ", dy = " << dy << std::endl;

        warped[i].copyTo(dst(Rect(dx, dy, warped[i].cols, warped[i].rows)), gray);

        corners[i][0].x = corners[i][0].x - dst_roi.x;
        corners[i][0].y = corners[i][0].y - dst_roi.y;
        corners[i][1].x = corners[i][1].x - dst_roi.x;
        corners[i][1].y = corners[i][1].y - dst_roi.y;
        corners[i][2].x = corners[i][2].x - dst_roi.x;
        corners[i][2].y = corners[i][2].y - dst_roi.y;
        corners[i][3].x = corners[i][3].x - dst_roi.x;
        corners[i][3].y = corners[i][3].y - dst_roi.y;
        //std::cout << "corners[i][0] X = " << corners[i][0].x << ", Y = " << corners[i][0].y << std::endl;
        if(i == g_inputImgNums - 1)
        {
            line(dst, Point(corners[i][0]),Point(corners[i][2]), cv::Scalar(0, 0, 255), 2);
            line(dst, Point(corners[i][2]),Point(corners[i][3]), cv::Scalar(0, 0, 255), 2);
            line(dst, Point(corners[i][3]),Point(corners[i][1]), cv::Scalar(0, 0, 255), 2);
            line(dst, Point(corners[i][0]),Point(corners[i][1]), cv::Scalar(0, 0, 255), 2);
        } else
        {
            line(dst, Point(corners[i][0]),Point(corners[i][2]), cv::Scalar(255, 255, 255), 2);
            line(dst, Point(corners[i][2]),Point(corners[i][3]), cv::Scalar(255, 255, 255), 2);
            line(dst, Point(corners[i][3]),Point(corners[i][1]), cv::Scalar(255, 255, 255), 2);
            line(dst, Point(corners[i][0]),Point(corners[i][1]), cv::Scalar(255, 255, 255), 2);
        }
    }

    resultCorners.clear();
    sizes.clear();
    corners.clear();
    HomoVecs.clear();
    NewHomoVecs.clear();
    warped.clear();

//    imwrite(AllStitchResult, result);
    dst.copyTo(result);
    g_panoVec.push_back(result); //save pano ,by //zt
    g_inputImgNums++;

    gettimeofday( &stitch_end, NULL );
    //求出两次时间的差值，单位为us
    int stitch_timeuse = 1000000 * ( stitch_end.tv_sec - stitch_start.tv_sec ) + stitch_end.tv_usec - stitch_start.tv_usec;
    if(!g_strLogPath.empty())
        output_file << "total time is " << stitch_timeuse << "  us." << std::endl;
    else
        printf("total time: %d us\n", stitch_timeuse);

//    cv::namedWindow("result", 0);
//    cv::imshow("result", result);
//    cv::waitKey();

    return true;
}

void capture_stitching_release()
{
    g_inputImgNums = 0;
    if(!g_finder.empty())
        g_finder->collectGarbage();
    g_finder.release();
    g_panoVec.clear();
    g_angleImg.clear();
    if(!g_strLogPath.empty())
        output_file.close();

    if(origin_keypoints.size() > 0)
        origin_keypoints.clear();
    if(keypoints.size() > 0)
        keypoints.clear();
}


#define Rsize_scale 0.25
#define Trans_rate 0.75
#define Max_origin_  100
#define Detail_Max_  80
#define Resize_gool 300.
;

bool find_back_tracking = false;

vector<KeyPoint> startKeypoints;
Mat start_Descriptor;

void KnnMatcher(Mat descriptors_1,Mat descriptors_2,vector<DMatch>& Dmatchinfo,float match_conf_);
float reprojError(vector<Point2f> pt_org, vector<Point2f> pt_prj, const Mat& H_front);
Size Get_Overlap_Area(vector<Point2f> corners);
bool Calc_Overlap(Mat& dst,int &maxOriginSize);
bool Start_Overlap(int &Max_origin_size);
void updateKeyPoint(const Mat& frame,int &MOS);
bool Retrieve_Overlap(Mat &dst,int Max_origin_Size);
bool Check_rads_bias(float size,const Mat& pic_,const Point2f& center);
void Overlap_calculate_ket(int& Max_origin_size);
void cal_area_corner(const Mat& pic,const Mat& H_final, vector<Point>&);
bool Check_center_crossborder(const Mat& pic,float border_rate,Mat H);
bool Check_Area_Correct(vector<Point>dst_pnts,cv::Mat &dst);


int Max_origin_size = Max_origin_;

typedef std::set<std::pair<int,int> > MatchesSet;

inline float getSideLength(const Point2f &p1, const Point2f &p2);
inline float getSideVec(const Point2f &p1, const Point2f &p2);
inline Mat resize_input(Mat &frame);
inline void resize_output(vector<Point>& dst_pt);

inline float getSideLength(const Point2f &p1, const Point2f &p2)
{
    float sideLength = sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
    return sideLength;
}
inline float getSideVec(const Point2f &p1, const Point2f &p2, const Point2f & p3)
{
    float sideVec = (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y);
    return sideVec;
}
bool Check_Area_Correct(vector<Point2f>dst_pnts,cv::Mat &dst);

//
Mat resize_input(Mat &frame){
    Mat output;
    float scale = Rsize_scale;
//    scale = Resize_gool/frame.cols;
    resize(frame,output,Size(),scale,scale,INTER_LINEAR);
    return output;
}
void resize_output(vector<Point>& dst_pt){
    for (int i = 0; i < 4; ++i) {
        dst_pt[i] =dst_pt[i]/Rsize_scale;
    }
}
bool Check_Area_Correct(vector<Point>dst_pnts,cv::Mat &dst){

    //判断重叠区域是否为四边形
    if(dst_pnts.size() != 4 || dst_pnts[0].x >= dst_pnts[1].x || dst_pnts[3].x >= dst_pnts[2].x || dst_pnts[0].y >= dst_pnts[3].y || dst_pnts[1].y >= dst_pnts[2].y)
    {
        std::cout << "OverLap: overlap area is not Quadrilateral. " << std::endl;
        return false;
    }

    //判断四边形内角
    float topSide = getSideLength(dst_pnts[0], dst_pnts[1]); //四边形的上边
    float bottomSide = getSideLength(dst_pnts[2], dst_pnts[3]); //四边形的下边
    float leftSide = getSideLength(dst_pnts[0], dst_pnts[3]); //四边形的左边
    float rightSide = getSideLength(dst_pnts[1], dst_pnts[2]); //四边形的右边
    std::cout << "topSide = " << topSide << ", bottomSide = " << bottomSide << ", leftSide = " << leftSide << ", rightSide = " << rightSide << std::endl;


    float lefttopVec = getSideVec(dst_pnts[0], dst_pnts[1], dst_pnts[3]);
    float righttopVec = getSideVec(dst_pnts[1], dst_pnts[0], dst_pnts[2]);
    float rightbottomVec = getSideVec(dst_pnts[2], dst_pnts[1], dst_pnts[3]);
    float leftbottomVec = getSideVec(dst_pnts[3], dst_pnts[0], dst_pnts[2]);


    int angle0 = acos(lefttopVec / (topSide * leftSide)) * 180 / 3.1416 + 0.5;
    int angle1 = acos(righttopVec / (topSide * rightSide)) * 180 / 3.1416 + 0.5;
    int angle2 = acos(leftbottomVec / (bottomSide * leftSide)) * 180 / 3.1416 + 0.5;
    int angle3 = acos(rightbottomVec / (bottomSide * rightSide)) * 180 / 3.1416 + 0.5;

    std::cout << "angle0 = " << angle0 << ", angle1 = " << angle1 << ", angle2 = " << angle2 << ", angle3 = " << angle3 << std::endl;
    if(abs(angle0) < 30 || abs(angle1) < 30 || abs(angle2) < 30 || abs(angle3) < 30)
    {
        //else
        std::cout << "OverLap: inline angle < 30. " << std::endl;
        return false;
    }
    return true;


}

bool overlap_point(cv::Mat &dst,std::vector<cv::Point> &src_pnts,std::vector<cv::Point> &dst_pnts){
    //找回机制
    dst = resize_input(dst);
    if(find_back_tracking){
        cout<<"find_back_tracking" <<endl;
        if(Retrieve_Overlap(dst,Max_origin_))
            find_back_tracking  = !find_back_tracking;
        if (!find_back_tracking){
            Max_origin_size = Max_origin_;
            updateKeyPoint(dst,Max_origin_size);
        }
    }
    else{
        if(!Calc_Overlap(dst,Max_origin_size))
            find_back_tracking  = true;
    }


    //测量中心点的偏移
    vector<Point2f> Pic_centre;
    Pic_centre.emplace_back((float)(dst.cols/2),(float)(dst.rows/2));
    perspectiveTransform(Pic_centre,Pic_centre,H_optflow_);
    if(!Check_rads_bias(-0.2,dst,Pic_centre[0])) {
        find_back_tracking = true;
        return false;
    }
    vector<Point> dst_(4);
    cal_area_corner(dst,H_optflow_,dst_pnts);
    cout << dst_pnts <<endl;
    if(!Check_Area_Correct(dst_pnts,dst))
        find_back_tracking = true;
    if (find_back_tracking)
        return false;
    resize_output(dst_pnts);
    for(const auto& pp:keypoints)
        circle(dst,pp,2,Scalar(0,200,0),2);
    return true;
}
bool set_src_feature( cv::Mat & frame){

    frame = resize_input(frame);
    Max_origin_size = (int)Max_origin_;
    H_last_ = Mat::eye(3,3,CV_64FC1);
    H_optflow_ = Mat::eye(3,3,CV_64FC1);
    find_back_tracking = false;

    keypoints.clear();
    origin_keypoints.clear();

    // 对第一帧提取FAST特征点
    vector<cv::KeyPoint> kps;
    Ptr<GFTTDetector> gftt = GFTTDetector::create(Max_origin_);
    gftt->detect(frame,kps);
    //    KeyPointsFilter::retainBest(kps, Max_origin_);
    for ( const auto& kp:kps ){
        keypoints.push_back( kp.pt );
        origin_keypoints.push_back( kp.pt);
    }

    cout  << "keypoints? = " << keypoints.size() <<endl;
    Ptr<xfeatures2d::SIFT> detail_detector = xfeatures2d::SIFT::create(Max_origin_,2);
    //    Ptr<DescriptorExtractor> detail_extractor= xfeatures2d::SIFT::create(Max_origin_);
    detail_detector->detectAndCompute(frame, Mat(), startKeypoints, start_Descriptor);
    //    cout << start_Descriptor.size <<  "? = " << startKeypoints.size() <<endl;
    gftt.release();
    detail_detector.release();
    //    detail_extractor.release();
    kps.clear();
    frame.copyTo(templateImg);

    return true;
}

bool Retrieve_Overlap(Mat& dst,int Max_origin_Size){
    vector<KeyPoint> frame_keypoints;
    Mat frame_descriptor;

    //先不断地去寻找 SIFT特征匹配
    Ptr<xfeatures2d::SIFT> detail_detector = xfeatures2d::SIFT::create(Detail_Max_,2);
    detail_detector->detectAndCompute(dst,Mat(),frame_keypoints,frame_descriptor);


    if(frame_keypoints.size() < startKeypoints.size() / 20)
        return false;
    vector<DMatch> matches;
    std::vector<Point2f> start_pt, frame_pt;  //重复区域匹配点list
    KnnMatcher(start_Descriptor, frame_descriptor, matches, 0.25);
    //将kNN的配对填入两个缓冲 vector
    for (auto m : matches) {
        //得到第一幅图像的当前匹配点对的特征点坐标
        Point2f p = startKeypoints[m.queryIdx].pt;
        start_pt.push_back(p);    //特征点坐标赋值
        p = frame_keypoints[m.trainIdx].pt;
        frame_pt.push_back(p);
    }

    if( start_pt.size()<6||frame_pt.size()<6)
        return false;

    std::vector<uchar> inliers_mask;
    cv::Mat H = findHomography(start_pt, frame_pt,inliers_mask,LMEDS);
    int num_inliers = 0;    //匹配点对的内点数先清零
    //由内点掩码得到内点数
    //在KNN基础上再得到内点 vector
    vector<Point2f>inlier_org,inlier_prj;
    for (size_t i = 0; i < inliers_mask.size(); ++i)    //遍历匹配点对，得到内点
    {
        if (!inliers_mask[i])    //不是内点
            continue;
        Point2f p = start_pt[i];    //第一幅图像的内点坐标
        inlier_org.push_back(p);
        p = frame_pt[i];    //第一幅图像的内点坐标
        inlier_prj.push_back(p);
        num_inliers++;
    }

    float confidence = num_inliers / (8 + 0.3 * start_pt.size());
    cout << "confidence = " << confidence << endl;
    float err = reprojError(inlier_org, inlier_prj, H);
    cout << "err = " << err <<endl;
    if (err>10){
        return false;
    }
    if(confidence<1.2)
        return false;
    if(!Check_center_crossborder(dst,0.1,H.inv())){
        cout << "check false;" <<endl;
        return false;
    }
    //再次更新 两个H矩阵
    H.copyTo(H_optflow_);
    H.copyTo(H_last_);
//    cout << "auto_retrieve" <<endl;
    return true;
}
void cal_area_corner(const Mat& pic,const Mat& H_final,vector<Point>& output_pt){
    std::vector<Point2f> template_pt(4);
    std::vector<Point2f> output_temp_corner(4);
    template_pt[0] = Point2f(0, 0);
    template_pt[1] = Point2f((float)pic.cols, 0 );
    template_pt[2] = Point2f((float)pic.cols, (float)pic.rows );
    template_pt[3] = Point2f( 0, (float)pic.rows );
    perspectiveTransform(template_pt, output_temp_corner, H_final);
    for (int j = 0; j < 4; ++j) {
        output_pt[j] = Point((int)output_temp_corner[j].x,(int)output_temp_corner[j].y);
    }
}
void KnnMatcher(const Mat descriptors_1,const Mat descriptors_2,vector<DMatch>& Dmatchinfo,const float match_conf_) {
    Dmatchinfo.clear();    //清空
    //定义K-D树形式的索引
    Ptr<flann::IndexParams> indexParams = new flann::KDTreeIndexParams();
    //定义搜索参数
    Ptr<flann::SearchParams> searchParams = new flann::SearchParams();

    if (descriptors_1.depth() == CV_8U) {
        indexParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
        searchParams->setAlgorithm(cvflann::FLANN_INDEX_LSH);
    }
    //使用FLANN方法匹配，定义matcher变量
    FlannBasedMatcher matcher(indexParams, searchParams);
    vector<vector<DMatch> > pair_matches;    //表示邻域特征点
    MatchesSet matches;    //表示匹配点对

    // Find 1->2 matches
    //在第二幅图像中，找到与第一幅图像的特征点最相近的两个特征点
    matcher.knnMatch(descriptors_1, descriptors_2, pair_matches, 2);
    for (auto & pair_matche : pair_matches)    //遍历这两次匹配结果
    {
        //如果相近的特征点少于2个，则继续下个匹配
        if (pair_matche.size() < 2)
            continue;
        //得到两个最相近的特征点
        const DMatch &m0 = pair_matche[0];
        const DMatch &m1 = pair_matche[1];
        //比较这两个最相近的特征点的相似程度，当满足一定条件时（用match_conf_变量来衡量），才能认为匹配成功
        //TODO match_conf_越大说明第二相似的匹配点，要与第一相似的匹配点差距越大，也就是匹配的要求
        // 特例性，match_conf_ 越大 粗匹配点数越少
        if (m0.distance < (1.f - match_conf_) * m1.distance)    //式1
        {
            //把匹配点对分别保存在matches_info和matches中
            Dmatchinfo.push_back(m0);
            matches.insert(make_pair(m0.queryIdx, m0.trainIdx));
        }
    }

    pair_matches.clear();    //变量清零
}
bool Check_center_crossborder(const Mat& pic,float border_rate,Mat H){
    vector<Point2f> Pic_centre;
    Pic_centre.emplace_back((pic.cols/2),(pic.rows/2));
    perspectiveTransform(Pic_centre,Pic_centre,H);
    return Check_rads_bias(border_rate, pic, Pic_centre[0]);
}
bool Calc_Overlap(Mat &dst,int &maxOriginSize){
    // 对其他帧用LK跟踪特征点
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    //填入 buffer
    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> prev_keypoints;
    for ( const auto& kp:keypoints )
        prev_keypoints.push_back(kp);
    vector<unsigned char> status;
    vector<float> error;

    //计算光流跟踪点
    cv::calcOpticalFlowPyrLK( templateImg, dst, prev_keypoints, next_keypoints, status, error );
//        cout << prev_keypoints.size() << " =? " << next_keypoints.size() << endl;
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration <double> time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );
    cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;

    // 把跟丢的点删掉
    int i=0;int number_stable = 0;
    auto iter_tmp = origin_keypoints.begin();
    for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
    {
        if (status[i] == 0 ){
            //同步删除那些origin_keypoints 中的对应点
            iter = keypoints.erase(iter);
            iter_tmp = origin_keypoints.erase(iter_tmp);
            continue;
        }
        iter_tmp++;
        *iter = next_keypoints[i];
        iter++;
    }

    cout << "tracked keypoints: " << keypoints.size()<<endl;
    cout << "origin_keypoints keypoints: " << origin_keypoints.size()<<endl;
    //如果光流跟踪到的点特别少直接退出
    if(keypoints.size()<Max_origin_/10)
        return false;

    Mat src_points(1, static_cast<int>(prev_keypoints.size()), CV_32FC2);
    Mat dst_points(1, static_cast<int>(prev_keypoints.size()), CV_32FC2);
    vector<KeyPoint> srcKeypoint,dstKeypoint;
    //遍历所有匹配点对，得到匹配点对的特征点坐标
    auto psrc =origin_keypoints.begin();
    auto pdst =keypoints.begin();
    for (size_t step = 0; step < prev_keypoints.size(); ++step) {
        src_points.at<Point2f>(0, static_cast<int>(step)) = *psrc;    //特征点坐标赋值
        dst_points.at<Point2f>(0, static_cast<int>(step)) = *pdst;    //特征点坐标赋值
        psrc++;
        pdst++;
    }
    vector<uchar> origin_kp_mask;
    Mat H_temp = findHomography(src_points,dst_points,LMEDS,5,
                                origin_kp_mask,1000,0.99);
    H_optflow_ = H_temp*H_last_;
    //    cout << H_optflow_ <<endl;
    //todo 如果光流跟踪点数量少于0.9
//    if(Max_origin_size < Max_origin_/3)
//        return false;
    if (keypoints.size() < maxOriginSize * Trans_rate)
    {
//        return false;
        updateKeyPoint(dst, maxOriginSize);
        H_optflow_.copyTo(H_last_);
    }
    dst.copyTo(templateImg);

    time_used = chrono::duration_cast <chrono::duration<double>> ( t2 - t1 );
    cout<<"algorithm use time："<<time_used.count()<<" seconds.\n"<<endl;
    return true;
}
void updateKeyPoint(const Mat& frame,int &MOS){
    //先清空 vector point buffer
    keypoints.clear();
    origin_keypoints.clear();
    vector<cv::KeyPoint> kps;
    Ptr<GFTTDetector> gftt =
            GFTTDetector::create(Max_origin_,
                                 0.01,
                                 1,
                                 3,
                                 true);
    gftt->detect(frame,kps);
    MOS = kps.size();

    for ( const auto& kp:kps ){
        keypoints.push_back( kp.pt );
        origin_keypoints.push_back( kp.pt);
    }
    frame.copyTo(templateImg);
}
Size Get_Overlap_Area(vector<Point2f> corners){
    // 这里需要改成四个顶点的外接矩形坐标
    Point tmp0, tmp1, LT_position,RB_position;
    //LT_position 是该图片在全景图中的左上角位置坐标
    tmp0.x = min(corners[0].x, corners[1].x);
    tmp1.x = min(corners[2].x, corners[3].x);
    LT_position.x = min(tmp0.x, tmp1.x);

    tmp0.y = min(corners[0].y, corners[1].y);
    tmp1.y = min(corners[2].y, corners[3].y);
    LT_position.y = min(tmp0.y, tmp1.y);

    tmp0.x = max(corners[0].x, corners[1].x);
    tmp1.x = max(corners[2].x, corners[3].x);
    RB_position.x = max(tmp0.x, tmp1.x);

    tmp0.y = max(corners[0].y, corners[1].y);
    tmp1.y = max(corners[2].y, corners[3].y);
    RB_position.y = max(tmp0.y, tmp1.y);
    Size returnout(-LT_position.x+RB_position.x,
                   -LT_position.y+RB_position.y);
    return returnout;
}
bool Check_rads_bias(float size,const Mat& pic_,const Point2f& center){

    float rad_least = min(pic_.rows/2,pic_.cols/2);
    float centre_x = center.x - pic_.cols/2;
    float centre_y = center.y - pic_.rows/2;
//    cout << centre_x << " " <<centre_y << endl;
//    float centre_y;
    float rad_dis = sqrtf(centre_x*centre_x+centre_y*centre_y);
//    cout << "rand_dis" << rad_dis <<endl;
    return (rad_dis < rad_least*(1-size));

}
float reprojError(vector<Point2f> pt_org, vector<Point2f> pt_prj, const Mat& H_front)
{
    //m1和m2为匹配点对
//model表示单应矩阵H
//_err表示所有匹配点对的重映射误差，即式21几何距离的平方
    Mat H_inerr;
    H_front.copyTo(H_inerr);
//    cout << "H_inerr : " <<H_inerr <<endl;
    vector<double> err;
    err.clear();
//    int count = pt_org.size();
//    const double* H = reinterpret_cast<const double*>(H_front.data);
    for(int i = 0; i < pt_org.size(); i++ )    //遍历所有特征点，计算重映射误差
    {
        double ww = 1./(H_inerr.at<double>(2,0)*pt_org[i].x + H_inerr.at<double>(2,1)*pt_org[i].y + 1.);    //式21中分式的分母部分
        double dx = (H_inerr.at<double>(0,0)*pt_org[i].x +
                     H_inerr.at<double>(0,1)*pt_org[i].y +
                     H_inerr.at<double>(0,2))*
                    ww - pt_prj[i].x;    //式21中x坐标之差
        double dy = (H_inerr.at<double>(1,0)*pt_org[i].x +
                     H_inerr.at<double>(1,1)*pt_org[i].y +
                     H_inerr.at<double>(1,2))*
                    ww - pt_prj[i].y;    //式21中y坐标之差
        err.push_back((double)dx*dx + (double)dy*dy);    //得到当前匹配点对的重映射误差
    }
    double err_mean =  (accumulate(err.begin(),err.end(),0.0))/(double)pt_org.size() ;
    return (float)err_mean;
}