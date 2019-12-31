//
// Created by aruchid on 2019/12/11.
//

#ifndef IMAGEBASICS_STITCH_CONNECTOR_H
#define IMAGEBASICS_STITCH_CONNECTOR_H



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


bool Stitch_Init();
bool StitchGroup_Finish();
bool Photo_Delete();
bool Photo_Increase (Mat Img_New,Point3d EularAngles_New,double timestamp_New,Mat &PanoResult);
void Projectall_opticalflow(Mat Img_New,Point3d EularAngles_New,double timestamp_New,Mat Homo_New,Mat &PanoResultOutput);

bool set_src_feature(cv::Mat &src, cv::detail::ImageFeatures &src_features);

bool set_src_feature(cv::Mat &src, vector<KeyPoint> &src_keypoints,Mat src_descriptor);
/**
 * calculate the projected points
 * @param dst target image
 * @param src_features template img feature
 * @param target_Angle template img pose
 * @param Eular_Angle target img pose
 * @param src_pnts target image projected points in template image coordinate
 * @param dst_pnts template image projected points in target image coordinate
 * @return 1 success
 *         0 fail
 */
bool overlap_point(
        cv::Mat &dst,
        ImageFeatures &src_features ,
        const Point3d target_Angle,
        const Point3d Eular_Angle,
        vector<Point> &src_pnts,
        vector<Point> &dst_pnts);

/**
 * capture a new image. if image stitching succeeds, return true; if not return false.
 * @param src   input image
 * @param dst   output panorama
 * @param mode   0: stitch
 *               1: delete last pic
 * @param strLogPath   output algorithm log file
 * @return true(stitching succeed)
 *         false(stitching fail)
 */
bool overlap_point(cv::Mat &dst,
        ImageFeatures &src_features,
        vector<Point> &src_pnts,
        vector<Point> &dst_pnts);

bool trans(cv::Mat src,
           cv::Mat dst,
           std::vector<cv::Point> &src_pnts,
           std::vector<cv::Point> &dst_pnts);
#endif //IMAGEBASICS_STITCH_CONNECTOR_H
