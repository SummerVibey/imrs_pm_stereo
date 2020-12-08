#include "mvs_matcher_wrapper.h"

MVSMatcherWrapper::MVSMatcherWrapper(PatchMatchOptions *options, int height, int width)
{
  width_ = width;
  height_ = height;
  matcher_ = new MultiViewStereoMatcherCuda(options);
  int height_lvl = height;
  int width_lvl = width;
  max_level_ = 0;
  while(height_lvl > min_height && width_lvl > min_width) {
    max_level_++;
    height_lvl >> max_level_;
    width_lvl >> max_level_;
  }
  printf("You have constructed MVS Matcher Successfully! The image pyramid has %d levels!\n", max_level_);
  printf("Now you can add registered images as reference view and source views!\n");
}

void MVSMatcherWrapper::SetReferenceView(const cv::Mat& img, const cv::Mat_<float> &K, const cv::Mat_<float> &Rcw, const cv::Mat_<float> &tcw)
{

  img_ref_ = img.clone();
  Kref_ = K.clone();
  Rrw_ = Rcw.clone();
  trw_ = tcw.clone();
}

void MVSMatcherWrapper::AddSourceView(const cv::Mat& img, const cv::Mat_<float> &K, const cv::Mat_<float> &Rcw, const cv::Mat_<float> &tcw)
{
  imgs_src_.push_back(img);
  Ksrcs_.push_back(K);
  Rsws_.push_back(Rcw);
  tsws_.push_back(tcw);
}

void MVSMatcherWrapper::SetReferenceView(const cv::Mat& img, const cv::Mat_<float> &P)
{
  img_ref_ = img.clone();
  Pref_ = P.clone();
}

void MVSMatcherWrapper::AddSourceView(const cv::Mat& img, const cv::Mat_<float> &P)
{
  imgs_src_.push_back(img.clone());
  Psrcs_.push_back(P.clone());
}

void MVSMatcherWrapper::BuildLevel(int level)
{

} 

void MVSMatcherWrapper::InitializeKRT(int level)
{
  int height_lvl = height_ >> level;
  int width_lvl = width_ >> level;
  height_scale_ = height_lvl / height_;
  width_scale_ = width_lvl / width_;

  matcher_->ref_ = new RefViewSpace(Kref_, Rrw_, trw_, img_ref_.rows, img_ref_.cols);
  if(img_ref_.type() == CV_8U)
    img_ref_.convertTo(img_ref_, CV_32F);

  // CreateTextureObject(img_ref_, matcher_->ref_->tex_, matcher_->ref_->arr_);

  assert(imgs_src_.size() == Rsws_.size() == tsws_.size() == Ksrcs_.size());
  matcher_->image_size_ = imgs_src_.size(); 
  for(int i = 0; i < matcher_->image_size_; ++i) {
    if(imgs_src_[i].type() == CV_8U)
      imgs_src_[i].convertTo(imgs_src_[i], CV_32F);

    matcher_->src_[i] = new ViewSpace(Ksrcs_[i], Rsws_[i], tsws_[i], imgs_src_[i].rows, imgs_src_[i].cols);
    // CreateTextureObject(imgs_src_[i], matcher_->src_[i]->tex_, matcher_->src_[i]->arr_);
  }
  printf("Multi-View Stereo Matcher has been initialized!\n");
  matcher_->options_->Print();
  printf("You have added %d source view!\n", matcher_->image_size_);

  // ReleaseAll();
}

void MVSMatcherWrapper::InitializeP()
{
  matcher_->ref_ = new RefViewSpace(Pref_, img_ref_.rows, img_ref_.cols);
  if(img_ref_.type() == CV_8U)
    img_ref_.convertTo(img_ref_, CV_32F);

  CreateTextureObject(img_ref_, matcher_->ref_->tex_, matcher_->ref_->arr_);

  assert(imgs_src_.size() == Rsws_.size() == tsws_.size() == Ksrcs_.size());
  matcher_->image_size_ = imgs_src_.size(); 
  for(int i = 0; i < matcher_->image_size_; ++i) {
    if(imgs_src_[i].type() == CV_8U)
      imgs_src_[i].convertTo(imgs_src_[i], CV_32F);

    matcher_->src_[i] = new ViewSpace(Psrcs_[i], imgs_src_[i].rows, imgs_src_[i].cols);
    CreateTextureObject(imgs_src_[i], matcher_->src_[i]->tex_, matcher_->src_[i]->arr_);
  }
  printf("Multi-View Stereo Matcher has been initialized!\n");
  matcher_->options_->Print();
  printf("You have added %d source view!\n", matcher_->image_size_);

  ReleaseAll();
}

void MVSMatcherWrapper::Run(cv::Mat &depth, cv::Mat &normal)
{
  matcher_->Match(depth, normal);
}
