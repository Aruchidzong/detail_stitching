//
// Created by abc on 19-12-26.
//

#ifndef OPTICALFLOW_OPTICAL_HPP
#define OPTICALFLOW_OPTICAL_HPP

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/matchers.hpp"

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
bool overlap_point(cv::Mat &dst,
                      std::vector<cv::Point> &src_pnts,
                      std::vector<cv::Point> &dst_pnts);
bool calOverlap(cv::Mat &dst, std::vector<cv::Point> &src_pnts, std::vector<cv::Point> &dst_pnts);

#endif //OPTICALFLOW_OPTICAL_HPP
