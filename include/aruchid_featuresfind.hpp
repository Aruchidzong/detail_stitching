//
// Created by aruchid on 2019/12/8.
//

#ifndef IMAGEBASICS_ARUCHID_MATCHER_NEW_H
#define IMAGEBASICS_ARUCHID_MATCHER_NEW_H
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d.hpp>
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

#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

//#include "./aruchid_log.h"
// #include "extra.h" // use this if in OpenCV2
using namespace std;
using namespace cv;
using namespace cv::detail;

typedef struct {
    double pitch;
    double yaw;
    double roll;
}Rotate_angle;

typedef std::set<std::pair<int,int> > MatchesSet;



int test_featurefind (const Mat img1,const Mat img2);
Mat FindMatch_CurseH_ (  Mat img1,Mat img2,
                                                                             std::vector<KeyPoint>* keypoints_1,
                                                                             std::vector<KeyPoint>* keypoints_2);
Mat FindMatch_CurseH (vector<Mat> imgs,
                      int Rotated_,
                      int Target_,
                      std::vector<KeyPoint>* keypoints_1,
                      std::vector<KeyPoint>* keypoints_2,
                      vector<Point3d> Eular_Angle,
                      int Best_POV);

void pose_estimation_2d2d (
        std::vector<KeyPoint> keypoints_1,
        std::vector<KeyPoint> keypoints_2,
        std::vector< DMatch > matches,
        Mat& R, Mat& t,
        std::vector< char > inlier_mask);
void KnnMatcher(const Mat descriptors_1, const Mat descriptors_2,
                vector<DMatch>& Dmatchinfo,const float match_conf_);
Point2d pixel2cam ( const Point2d& p, const Mat& K );
#endif //IMAGEBASICS_ARUCHID_MATCHER_NEW_H
