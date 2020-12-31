#ifndef MVS_MATCHER_WRAPPER_H
#define MVS_MATCHER_WRAPPER_H


#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl-1.8/pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <pcl-1.8/pcl/visualization/cloud_viewer.h>

#include "data_io.h"
#include "data_types.h"

#define SafeReleaseMat(mat) {if(!mat.empty()) mat.release();}

#define SafeReleaseMatBuffer(mat_buf) \
  {for(int i=0;i<(int)mat_buf.size();++i) SafeReleaseMat(mat_buf[i]);}

#define SafeReleaseMatDoubleBuffer(mat_dbuf) \
  {for(int i=0;i<(int)mat_dbuf[0].size();++i) \
    for(int j=0;j<(int)mat_dbuf.size();++j) \
      SafeReleaseMat(mat_dbuf[j][i]);}

const int min_width = 300;
const int min_height = 200;

class MVSMatcherWrapper
{
public:
  MVSMatcherWrapper() : matcher_(nullptr) {}

  MVSMatcherWrapper(PatchMatchOptions *options, int height, int width);

  ~MVSMatcherWrapper() {  ReleaseAll(); delete matcher_; }

  void ReleaseAll() {
    SafeReleaseMatBuffer(img_ref_);img_ref_.clear();
    SafeReleaseMatBuffer(Kref_);Kref_.clear();
    SafeReleaseMat(Rrw_);
    SafeReleaseMat(trw_);
    
    SafeReleaseMatDoubleBuffer(imgs_src_);imgs_src_.clear();
    SafeReleaseMatDoubleBuffer(Ksrcs_);Ksrcs_.clear();
    SafeReleaseMatBuffer(Rsws_);Rsws_.clear();
    SafeReleaseMatBuffer(tsws_);tsws_.clear();
  }

  void ReleaseLevel(int level) {
    SafeReleaseMat(img_ref_[level]);
    SafeReleaseMat(Kref_[level]);

    SafeReleaseMatBuffer(imgs_src_[level]);
    SafeReleaseMatBuffer(Ksrcs_[level]);
  }

  void SetReferenceView(const cv::Mat& img, const cv::Mat_<float> &K, const cv::Mat_<float> &Rcw, const cv::Mat_<float> &tcw);

  void SetReferenceView(const cv::Mat& img, const cv::Mat_<float> &P);

  void AddSourceView(const cv::Mat& img, const cv::Mat_<float> &K, const cv::Mat_<float> &Rcw, const cv::Mat_<float> &tcw);

  void AddSourceView(const cv::Mat& img, const cv::Mat_<float> &P);

  void Initialize();

  void InitializeP();

  void BuildPyramid();

  void Run(cv::Mat &depth, cv::Mat &normal, int level);

  void Run(cv::Mat &depth, cv::Mat &normal);

  // void GetPointCloud(cv::Mat &depth, cv::Mat &normal, int level);

  MultiViewStereoMatcherCuda *matcher_;
  PatchMatchOptions *options_;

  int width_, height_;
  int max_level_;
  float width_scale_;
  float height_scale_;

  // host buffer
  std::vector<cv::Mat> img_ref_;
  std::vector<cv::Mat> Kref_;
  std::vector<cv::Mat> Pref_;
  cv::Mat Rrw_;
  cv::Mat trw_;
  

  std::vector<std::vector<cv::Mat>> imgs_src_;
  std::vector<std::vector<cv::Mat>> Ksrcs_;
  std::vector<std::vector<cv::Mat>> Psrcs_;
  std::vector<cv::Mat> Rsws_;
  std::vector<cv::Mat> tsws_;
  

};

#endif