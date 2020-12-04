#ifndef MVS_MATCHER_WRAPPER_H
#define MVS_MATCHER_WRAPPER_H


#include <vector>
#include <opencv2/core/core.hpp>

#include "data_io.h"
#include "data_types.h"

class MVSMatcherWrapper
{
public:
  MVSMatcherWrapper() : matcher_(nullptr) {}

  MVSMatcherWrapper(PatchMatchOptions *options);

  ~MVSMatcherWrapper() {  ReleaseAll(); delete matcher_; }

  void ReleaseAll() {
    img_ref_.release();
    Rrw_.release();
    trw_.release();
    for(int i = 0; i < (int)imgs_src_.size(); ++i) {
      imgs_src_[i].release();
      Rsws_[i].release();
      tsws_[i].release();
    }
  }

  void SetReferenceView(const cv::Mat& img, const cv::Mat_<float> &K, const cv::Mat_<float> &Rcw, const cv::Mat_<float> &tcw);

  void AddSourceView(const cv::Mat& img, const cv::Mat_<float> &K, const cv::Mat_<float> &Rcw, const cv::Mat_<float> &tcw);

  void Initialize();

  void Run();

  MultiViewStereoMatcherCuda *matcher_;

  // host buffer
  cv::Mat img_ref_;
  cv::Mat Kref_;
  cv::Mat Rrw_;
  cv::Mat trw_;

  std::vector<cv::Mat> imgs_src_;
  std::vector<cv::Mat> Ksrcs_;
  std::vector<cv::Mat> Rsws_;
  std::vector<cv::Mat> tsws_;
};

#endif