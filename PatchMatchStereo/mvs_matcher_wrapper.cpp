#include "mvs_matcher_wrapper.h"

MVSMatcherWrapper::MVSMatcherWrapper(PatchMatchOptions *options, int height, int width)
{
  width_ = width;
  height_ = height;
  options_ = options; 

  // int height_lvl = height;
  // int width_lvl = width;
  // max_level_ = 0;
  // while(height_lvl > min_height && width_lvl > min_width) {
  //   max_level_++;
  //   height_lvl /= 2;
  //   width_lvl /= 2;
  // }

  // img_ref_.resize(max_level_ + 1);
  // imgs_src_.resize(max_level_ + 1);
  // Kref_.resize(max_level_ + 1);
  // Ksrcs_.resize(max_level_ + 1);

  printf("You have constructed MVS Matcher Successfully! The image pyramid has %d levels!\n", max_level_ + 1);
  printf("Now you can add registered images as reference view and source views!\n");
}

void MVSMatcherWrapper::SetReferenceView(const cv::Mat& img, const cv::Mat_<float> &K, const cv::Mat_<float> &Rcw, const cv::Mat_<float> &tcw)
{

  // img_ref_[0] = img.clone();
  // Kref_[0] = K.clone();
  // Rrw_ = Rcw.clone();
  // trw_ = tcw.clone();
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

void MVSMatcherWrapper::Initialize()
{
  int images_size = imgs_src_.size();
  // for(int lvl = 1; lvl <= max_level_; lvl++) {
  //   cv::pyrDown(img_ref_[lvl-1], img_ref_[lvl], cv::Size(img_ref_[lvl-1].cols/2, img_ref_[lvl-1].rows/2));

  //   width_scale_ = (float)img_ref_[lvl].cols / (float)img_ref_[lvl-1].cols;
  //   height_scale_ = (float)img_ref_[lvl].rows / (float)img_ref_[lvl-1].rows;
  //   Kref_[lvl] = Kref_[lvl-1].clone();
  //   Kref_[lvl].rowRange(0,1) *= width_scale_;
  //   Kref_[lvl].rowRange(1,2) *= height_scale_;
    
  //   imgs_src_[lvl].resize(images_size);
  //   Ksrcs_[lvl].resize(images_size);
  //   for(int i = 0; i < images_size; ++i) {
  //     cv::pyrDown(imgs_src_[lvl-1][i], imgs_src_[lvl][i], cv::Size(imgs_src_[lvl-1][i].cols/2, imgs_src_[lvl-1][i].rows/2));
  //     width_scale_ = img_ref_[lvl].cols/2 / img_ref_[lvl-1].cols/2;
  //     height_scale_ = img_ref_[lvl].rows/2 / img_ref_[lvl-1].rows/2;
  //     width_scale_ = (float)imgs_src_[lvl][i].cols / (float)imgs_src_[lvl-1][i].cols;
  //     height_scale_ = (float)imgs_src_[lvl][i].rows / (float)imgs_src_[lvl-1][i].rows;
  //     Ksrcs_[lvl][i] = Ksrcs_[lvl-1][i].clone();
  //     Ksrcs_[lvl][i].rowRange(0,1) *= width_scale_;
  //     Ksrcs_[lvl][i].rowRange(1,2) *= height_scale_;
  //   }
  // }
} 

void MVSMatcherWrapper::Run(cv::Mat &depth, cv::Mat &normal, int level)
{
  matcher_->Reset(options_, imgs_src_.size());
  matcher_->ref_ = new RefViewType(Kref_, Rrw_, trw_, img_ref_.rows, img_ref_.cols);
  CreateTextureObject(img_ref_, matcher_->ref_->tex_, matcher_->ref_->arr_);

  assert(imgs_src_.size() == Rsws_.size() == tsws_.size() == Ksrcs_.size());
  for(int i = 0; i < matcher_->image_size_; ++i) {
    matcher_->src_[i] = new ViewType(Ksrcs_[i], Rsws_[i], tsws_[i], imgs_src_[i].rows, imgs_src_[i].cols);
    CreateTextureObject(imgs_src_[i], matcher_->src_[i]->tex_, matcher_->src_[i]->arr_);
  }
  printf("Multi-View Stereo Matcher has been initialized!\n");
  matcher_->options_->Print();
  printf("You have added %d source view!\n", matcher_->image_size_);

  matcher_->Match(depth, normal);
}

void MVSMatcherWrapper::Run(cv::Mat &depth, cv::Mat &normal)
{
  matcher_ = new MVSMatcherCuda();
  Run(depth, normal, 0);
  ReleaseAll();
}

void MVSMatcherWrapper::RunBottom()
{
  std::cout << "here" << std::endl;
  matcher_->Reset(options_, imgs_src_.size());
  std::cout << "here" << std::endl;
  matcher_->ref_ = new RefViewType(Kref_, Rrw_, trw_, img_ref_.rows, img_ref_.cols);
  std::cout << "here" << std::endl;
  CreateTextureObject(img_ref_, matcher_->ref_->tex_, matcher_->ref_->arr_);
  std::cout << "here" << std::endl;
  matcher_->ref_->Allocate(imgs_src_.size());

  assert(imgs_src_.size() == Rsws_.size() == tsws_.size() == Ksrcs_.size());
  for(int i = 0; i < matcher_->image_size_; ++i) {
    matcher_->src_[i] = new ViewType(Ksrcs_[i], Rsws_[i], tsws_[i], imgs_src_[i].rows, imgs_src_[i].cols);
    CreateTextureObject(imgs_src_[i], matcher_->src_[i]->tex_, matcher_->src_[i]->arr_);
  }
  printf("Multi-View Stereo Matcher has been initialized!\n");
  matcher_->options_->Print();
  printf("You have added %d source view!\n", matcher_->image_size_);

  matcher_->Match(imgs_src_.size());
  ReleaseAll();
}

void MVSMatcherWrapper::RunDebug()
{
  matcher_ = new MVSMatcherCuda();
  printf("Start to run mvs on cuda!\n");
  RunBottom();
  ReleaseAll();
}

// void MVSMatcherWrapper::GetPointCloud( cv::Mat &depth, cv::Mat &normal, int level, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
// {
//   const float &fx = K.at<float>(0,0), &fy = K.at<float>(1,1), &cx = K.at<float>(0,2), &cy = K.at<float>(1,2);
//   cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());
//   for(int y = 0; y < height; ++y) {
//     for(int x = 0; x < width; ++x) {
//       float xc, yc, zc;
//       const float depth_ij = depth.at<float>(y,x);
//       if(depth_ij > 20) continue;
//       xc = depth_ij * (x - cx) / fx;
//       yc = depth_ij * (y - cy) / fy;
//       zc = depth_ij;
//       pcl::PointXYZRGB point;
//       point.x = xc;
//       point.y = yc;
//       point.z = zc;
//       point.b = color.at<cv::Vec3b>(y,x)[0];
//       point.g = color.at<cv::Vec3b>(y,x)[1];
//       point.r = color.at<cv::Vec3b>(y,x)[2];
//       cloud->points.push_back(point);
//     }
//   }
// }

