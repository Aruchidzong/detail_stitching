//
// Created by aruchid on 2019/12/2.
//

#ifndef IMAGEBASICS_ROTATION_INIT_H
#define IMAGEBASICS_ROTATION_INIT_H


#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES
#include <Eigen/Core>
// Eigen 几何模块
#include <Eigen/Geometry>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

//#include "Rotation_Init.h"
#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;


void Find_Overlap_area(Mat img_trans,
                       Mat img_target,
                       const Point3d Eular_Angle,
                       const Point3d target_Angle,
                       vector<Point> &src_pnts,
                       vector<Point> &dst_pnts
);
class Rotation{
public:
    struct point{
        double x, y, z;
        friend ostream& operator<<(ostream& os, const point p){
            os << "(" << p.x << "," << p.y << "," << p.z << ")";
            return os;
        }
    };
    Mat Get_Used_inv_H(const Mat H_input,const vector<Mat>corners,const vector<Mat>corners_trans);
    Mat Get_RotationHomo(   Mat img_input,const Point3d Eular_Angle,const Point3d target_Angle,
                            CameraParams Kamera);
    Mat Center_WarpPerspective(const Mat H,const Mat src,Mat dst,const vector <Mat> corners,const int target,const string downloadpath);
};


#endif //IMAGEBASICS_ROTATION_INIT_H
