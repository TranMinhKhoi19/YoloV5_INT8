#pragma once
#include <opencv2/opencv.hpp>
#include <array>

cv::Mat crop_with_margin(const cv::Mat& img, const cv::Rect& box,
                         float margin_ratio, int out_w, int out_h);

cv::Mat align_by_5pts(const cv::Mat& img,
                      const std::array<cv::Point2f,5>& lmk,
                      int out_w=160, int out_h=160);

