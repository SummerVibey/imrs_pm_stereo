#ifndef MVS_MATCHER_WRAPPER_H
#define MVS_MATCHER_WRAPPER_H


#include <vector>
#include <opencv2/core/core.hpp>

#include "data_io.h"
#include "data_types.h"

#define SafeReleaseMat(mat) {if(!mat.empty()) mat.release();}

#define SafeReleaseMatBuffer(mat_buf) \
  {for(int i=0;i<(int)mat_buf.size();++i) {if(!mat_buf[i].empty()) {mat_buf[i].release();}}}

const int min_width = 300;
const int min_height = 200;

class MVSMatcherWrapper
{
public:
  MVSMatcherWrapper() : matcher_(nullptr) {}

  MVSMatcherWrapper(PatchMatchOptions *options, int height, int width);

  ~MVSMatcherWrapper() {  ReleaseAll(); delete matcher_; }

  void ReleaseAll() {

    SafeReleaseMat(img_ref_);
    SafeReleaseMat(Kref_);
    SafeReleaseMat(Rrw_);
    SafeReleaseMat(trw_);
    SafeReleaseMat(Pref_);

    SafeReleaseMat(img_ref_lvl_);
    SafeReleaseMat(Kref_lvl_);
    SafeReleaseMat(Rrw_lvl_);
    SafeReleaseMat(trw_lvl_);
    SafeReleaseMat(Pref_lvl_);

    SafeReleaseMatBuffer(imgs_src_);
    SafeReleaseMatBuffer(Ksrcs_);
    SafeReleaseMatBuffer(Rsws_);
    SafeReleaseMatBuffer(tsws_);
    SafeReleaseMatBuffer(Psrcs_);

    SafeReleaseMatBuffer(imgs_src_lvl_);
    SafeReleaseMatBuffer(Ksrcs_lvl_);
    SafeReleaseMatBuffer(Rsws_lvl_);
    SafeReleaseMatBuffer(tsws_lvl_);
    SafeReleaseMatBuffer(Psrcs_lvl_);
  }

  void ReleaseLevel() {
    SafeReleaseMat(img_ref_lvl_);
    SafeReleaseMat(Kref_lvl_);
    SafeReleaseMat(Rrw_lvl_);
    SafeReleaseMat(trw_lvl_);
    SafeReleaseMat(Pref_lvl_);

    SafeReleaseMatBuffer(imgs_src_lvl_);
    SafeReleaseMatBuffer(Ksrcs_lvl_);
    SafeReleaseMatBuffer(Rsws_lvl_);
    SafeReleaseMatBuffer(tsws_lvl_);
    SafeReleaseMatBuffer(Psrcs_lvl_);
  }

  void SetReferenceView(const cv::Mat& img, const cv::Mat_<float> &K, const cv::Mat_<float> &Rcw, const cv::Mat_<float> &tcw);

  void SetReferenceView(const cv::Mat& img, const cv::Mat_<float> &P);

  void AddSourceView(const cv::Mat& img, const cv::Mat_<float> &K, const cv::Mat_<float> &Rcw, const cv::Mat_<float> &tcw);

  void AddSourceView(const cv::Mat& img, const cv::Mat_<float> &P);

  void InitializeKRT(int level);

  void InitializeP();

  void BuildLevel(int level);

  void Run(cv::Mat &depth, cv::Mat &normal);

  MultiViewStereoMatcherCuda *matcher_;

  int width_, height_;
  int max_level_;
  float width_scale_;
  float height_scale_;

  // host buffer
  cv::Mat img_ref_;
  cv::Mat Kref_;
  cv::Mat Rrw_;
  cv::Mat trw_;
  cv::Mat Pref_;

  std::vector<cv::Mat> imgs_src_;
  std::vector<cv::Mat> Ksrcs_;
  std::vector<cv::Mat> Rsws_;
  std::vector<cv::Mat> tsws_;
  std::vector<cv::Mat> Psrcs_;

  cv::Mat img_ref_lvl_;
  cv::Mat Kref_lvl_;
  cv::Mat Rrw_lvl_;
  cv::Mat trw_lvl_;
  cv::Mat Pref_lvl_;

  std::vector<cv::Mat> imgs_src_lvl_;
  std::vector<cv::Mat> Ksrcs_lvl_;
  std::vector<cv::Mat> Rsws_lvl_;
  std::vector<cv::Mat> tsws_lvl_;
  std::vector<cv::Mat> Psrcs_lvl_;

};

#endif