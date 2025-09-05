#include "align.h"
#include <vector>

cv::Mat crop_with_margin(const cv::Mat& img, const cv::Rect& box,
                         float margin_ratio, int out_w, int out_h)
{
    int cx = box.x + box.width/2;
    int cy = box.y + box.height/2;
    int w = (int)(box.width  * (1.f + margin_ratio));
    int h = (int)(box.height * (1.f + margin_ratio));
    int x = std::max(0, cx - w/2);
    int y = std::max(0, cy - h/2);
    int x2 = std::min(img.cols, x + w);
    int y2 = std::min(img.rows, y + h);
    cv::Rect r(x, y, std::max(1, x2-x), std::max(1, y2-y));
    cv::Mat face = img(r).clone();
    cv::resize(face, face, cv::Size(out_w, out_h));
    return face;
}

cv::Mat align_by_5pts(const cv::Mat& img,
                      const std::array<cv::Point2f,5>& lmk,
                      int out_w, int out_h)
{
    // Template 160x160 -> scale theo out_w/out_h
    std::array<cv::Point2f,5> ref = {
        cv::Point2f(54.0f, 73.0f),
        cv::Point2f(106.0f, 73.0f),
        cv::Point2f(80.0f, 96.0f),
        cv::Point2f(60.0f, 118.0f),
        cv::Point2f(100.0f,118.0f)
    };
    float sx = out_w / 160.0f, sy = out_h / 160.0f;

    std::vector<cv::Point2f> src(5), dst(5);
    for (int i=0;i<5;i++) {
        src[i] = lmk[i];
        dst[i] = cv::Point2f(ref[i].x * sx, ref[i].y * sy);
    }

    cv::Mat M = cv::estimateAffinePartial2D(src, dst);
    cv::Mat aligned;
    cv::warpAffine(img, aligned, M, cv::Size(out_w, out_h),
                   cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return aligned;
}

