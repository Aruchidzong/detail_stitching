//
// Created by aruchid on 2019/12/7.
//



#include<vector>
#include <opencv2/opencv.hpp>
#include<limits.h>
#include <opencv2/opencv_modules.hpp>


#include <opencv2/line_descriptor.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;

#define WORKING_NEW 0
#define WORKING_RESPAWN 1

int FindhBPOV_native(vector<Mat> Homos,Mat img0);
class Pipeline{
public:
    Pipeline();
    void Init(char ** argv);
    void GetDataFrame();
    bool FindBestPOV();
    //todo 这里开始是findhomo的过程
    void GetHomoOfAll();
    bool GetHomoOfNew(Mat Newimg,Point3d NewEularAngle);
    void AllImgProjection();
    void ImgProjection(bool test);
    void SingleImgProjection();
    void Release();
    bool Check_Info_Integrity();

    int Img_Numbers;
    int Best_POV;
    bool WorkingStates;
    //输入部分
    detail::CameraParams Kamera;
    vector<Point3d> Eular_angle;
    vector<double> timestamp;
    vector<Mat> Homo;
    vector<Mat> imgs;
	
	Mat PanoResult;
private:

    //todo 2pic  pair features
    vector<KeyPoint> keypoints_target, keypoints_trans;
    vector<KeyLine> keylines;
    vector<DMatch> PointMatch;
    vector<DMatch> LineMatch;

    string Path_datafolder_;
    string Path_data_rotation_Info_;
//    string Path_data;
    int Target  = 1;
    int Rotated = 2;
    Mat H_2pic;
    Mat dst_pic;
};




