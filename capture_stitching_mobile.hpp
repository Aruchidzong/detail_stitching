
#pragma once
#ifndef CAPTURE_STITCHING_MOBILE_HPP
#define CAPTURE_STITCHING_MOBILE_HPP

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/matchers.hpp"


/**
 * stitching module initialization. if initialization succeeds, return true; if not return false.
 * @return true(initialization succeed)
 *         false(initialization fail)
 */
bool capture_stitching_init(double scale, std::string & strLogPath);

/**
 * set template pic, save features
 * @param src template image
 * @param src_features template img feature
 * @return 1 success
 *         0 fail
 */
bool set_src_feature(cv::Mat &src);

/**
 * calculate the projected points
 * @param dst target image
 * @param src_features template img feature
 * @param src_pnts target image projected points in template image coordinate
 * @param dst_pnts template image projected points in target image coordinate
 * @return 1 success
 *         0 fail
 */
//bool overlap_point(cv::Mat &dst, std::vector<cv::Point> &src_pnts, std::vector<cv::Point> &dst_pnts);
bool overlap_point(cv::Mat & frame, std::vector<cv::Point> & srcOverLapPoint, std::vector<cv::Point> & dstOverLapPoint);
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
bool capture_pic(cv::Mat &src, cv::Mat &dst, cv::Point3f angle, int mode, int & bestPOV, double Homo[9]);

/**
 * before exiting the program, please run the function below to release resources
 */
void capture_stitching_release();

#endif // CAPTURE_STITCHING_MOBILE_HPP
