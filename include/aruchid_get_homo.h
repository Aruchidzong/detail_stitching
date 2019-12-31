//
// Created by aruchid on 2019/12/8.
//

#ifndef IMAGEBASICS_ARUCHID_GET_HOMO_H
#define IMAGEBASICS_ARUCHID_GET_HOMO_H
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/opencv.hpp>
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

Mat ECC_refineHomo(const Mat imgTemplate,
                   const Mat imgTrans,
                   vector<KeyPoint> keypoints_trans,
                   vector<KeyPoint> keypoints_target);
Mat Simplely_findHomo(vector<KeyPoint> keypoints_trans,vector<KeyPoint> keypoints_target);

#endif //IMAGEBASICS_ARUCHID_GET_HOMO_H
