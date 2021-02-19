#include "display_utils.h"

void ConvertDisparityForDisplay(cv::Mat &disp, cv::Mat &disp_color)
{

  double max_val, min_val;
  cv::Point2i max_loc, min_loc;
  cv::minMaxLoc(disp, &min_val, &max_val, &min_loc, &max_loc);
  printf("max disp = %f, min disp = %f\n", (float)max_val, (float)min_val);
  printf("max loc = (%d, %d) min loc = (%d, %d)\n", max_loc.x, max_loc.y, min_loc.x, min_loc.y);

  cv::Mat disp_gray;
  disp.convertTo(disp_gray, CV_8U, 255.0f/(max_val-min_val),-min_val/(max_val-min_val));
  // disp_gray.convertTo(disp_gray, CV_8U, 1.f/(max_val-min_val),-min_val/(max_val-min_val));
  
	// for (int i = 0; i < height; i++) {
	// 	for (int j = 0; j < width; j++) {
	// 		const float disp_ij = disp.ptr<float>(i)[j];
  //     // disp_gray.data[i * width + j] = static_cast<uchar>((disp_ij - min_val)/(max_val-min_val));
  //     disp_gray.data[i * width + j] = static_cast<uchar>((disp_ij - min_val)/(max_val-min_val)*255);
	// 	}
	// }
	applyColorMap(disp_gray, disp_color, cv::COLORMAP_JET);
}

void ConvertDepthForDisplay(cv::Mat &inv_depth, cv::Mat &inv_depth_color)
{

  double max_val, min_val;
  cv::Point2i max_loc, min_loc;
  cv::minMaxLoc(inv_depth, &min_val, &max_val, &min_loc, &max_loc);
  printf("max inv_depth = %f, min inv_depth = %f\n", (float)max_val, (float)min_val);
  printf("max loc = (%d, %d) min loc = (%d, %d)\n", max_loc.x, max_loc.y, min_loc.x, min_loc.y);

  cv::Mat inv_depth_gray;
  inv_depth.convertTo(inv_depth_gray, CV_8U, 255.0f/(max_val-min_val),-min_val/(max_val-min_val));
	applyColorMap(inv_depth_gray, inv_depth_color, cv::COLORMAP_JET);

  // cv::minMaxLoc(inv_depth_gray, &min_val, &max_val, &min_loc, &max_loc);
  // printf("max depth = %f, min depth = %f\n", (float)max_val, (float)min_val);
  // printf("max loc = (%d, %d) min loc = (%d, %d)\n", max_loc.x, max_loc.y, min_loc.x, min_loc.y);
}

void ConvertNormalsForDisplay(const cv::Mat &normals, cv::Mat &normals_display)
{
  normals.convertTo(normals_display,CV_16U,32767,32767);
	cv::cvtColor(normals_display,normals_display,cv::COLOR_RGB2BGR);
}

void ConvertCostForDisplay(const cv::Mat cost, cv::Mat &cost_display)
{
  double max_val, min_val;
  cv::Point2i max_loc, min_loc;
  cv::minMaxLoc(cost, &min_val, &max_val, &min_loc, &max_loc);
  printf("max cost = %f, min cost = %f\n", (float)max_val, (float)min_val);
  printf("max loc = (%d, %d) min loc = (%d, %d)\n", max_loc.x, max_loc.y, min_loc.x, min_loc.y);
  cost.convertTo(cost_display, CV_32F, 1.f/(max_val-min_val),-min_val/(max_val-min_val));
}

void ConvertGradientForDisplay(const cv::Mat &grad, cv::Mat &grad_display)
{
  double max_val, min_val;
  cv::Point max_loc, min_loc;
  cv::minMaxLoc(grad, &min_val, &max_val, &min_loc, &max_loc);
  printf("max cost = %f, min cost = %f\n", max_val, min_val);
  grad.convertTo(grad_display, CV_32F, 1.f/(max_val-min_val),-min_val/(max_val-min_val));
}

void ComputeHistForDisplay(const cv::Mat& img, cv::Mat &hist_display)
{
  int channels = 0;
  double max_val, min_val;
  cv::Point2i max_loc, min_loc;
  cv::minMaxLoc(img, &min_val, &max_val, &min_loc, &max_loc);
  const int hist_size[] = {256};
  float min_ranges[] = {(float)min_val, (float)max_val}; 
  const float *ranges[] = { min_ranges }; 
  cv::MatND hist;
  cv::calcHist(&img, 1, &channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);
	cv::minMaxLoc(hist, 0, &max_val, 0, 0);
	int rows=cvRound(max_val);
  cv::Mat hist_image=cv::Mat::zeros(rows,256,CV_8UC1);
	for(int i=0;i<256;i++)
	{
		int temp=(int)(hist.at<float>(i,0));
		if(temp)
		{
			hist_image.col(i).rowRange(cv::Range(rows-temp,rows))=255; 
    }
  }
	resize(hist_image,hist_display,cv::Size(512,512));
}

void ConvertPlaneToDepthAndNormal(const InvDepthPlane *plane, cv::Mat_<float>& inv_depth, cv::Mat_<cv::Vec3f>& norm, int height, int width)
{
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      inv_depth(i, j) = plane[i*width+j].a * j + plane[i*width+j].b * i + plane[i*width+j].c;
      Vector3f norm_ij = plane[i*width+j].to_normal();
      norm(i, j) = cv::Vec3f(norm_ij.x, norm_ij.y, norm_ij.z);
    }
  }
}

void ConvertPlaneToDepthAndNormal(const PlaneState *plane, cv::Mat_<float>& depth, cv::Mat_<cv::Vec3f>& normal, int height, int width)
{
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      depth(i, j) = 180.0f / plane[i*width+j].idp_;
      normal(i, j) = cv::Vec3f(plane[i*width+j].nx_, plane[i*width+j].ny_, plane[i*width+j].nz_);
    }
  }
}




void ShowDepthAndNormal(const PlaneState *plane_data, int height, int width)
{
  cv::Mat_<float> inv_depth_img(height, width);
  cv::Mat_<cv::Vec3f> norm_img(height, width);
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      const PlaneState *plane_ij = &plane_data[i*width+j];
      // if(plane_ij->idp_ < 0.3f && plane_ij->idp_ > 0.01f)
        inv_depth_img(i, j) = plane_ij->idp_;
      // else 
      //   inv_depth_img(i, j) = 0;
      norm_img(i, j) = cv::Vec3f(plane_ij->nx_, plane_ij->ny_, plane_ij->nz_);
    }
  }
  
  cv::Mat inv_depth_display, norm_display;
  ConvertDepthForDisplay(inv_depth_img, inv_depth_display);
  ConvertNormalsForDisplay(norm_img, norm_display);
  cv::imshow("inv_depth_display", inv_depth_display);
  cv::imshow("norm_display", norm_display); 
	while(1)
	{
		if(cv::waitKey(0) == 'q')
			break;
	}
}

void ShowCostAndHistogram(float *cost, int height, int width)
{
  cv::Mat cost_img(height, width, CV_32F, cv::Scalar(0));
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      cost_img.ptr<float>(i)[j] = (float)cost[i*width+j];
    }
  }
  cv::Mat cost_display;
  ConvertCostForDisplay(cost_img, cost_display);
  cv::Mat hist_display;
  ComputeHistForDisplay(cost_img, hist_display);
  cv::imshow("cost", cost_display);
  cv::imshow("hist", hist_display);
  while(1) {
    if(cv::waitKey(0) == 'q') break;
  }
}

void ShowCostSliceAndHistogram(float *cost_volume, int src_size, int channel, int height, int width)
{
  cv::Mat cost_img(height, width, CV_32F, cv::Scalar(0));
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      cost_img.ptr<float>(i)[j] = (float)cost_volume[(i*width+j)*src_size + channel];
    }
  }
  cv::Mat cost_display;
  ConvertCostForDisplay(cost_img, cost_display);
  cv::Mat hist_display;
  ComputeHistForDisplay(cost_img, hist_display);
  // cv::namedWindow("cost");
  // cv::namedWindow("hist");
  cv::imshow("cost", cost_display);
  cv::imshow("hist", hist_display);
  while(1) {
    if(cv::waitKey(0) == 'q') break;
  }
}

void RenderDepthAndNormalMap(const float *depth_data, const float *normal_data, const int height, const int width)
{
  cv::Mat_<float> depth_image(height, width);
  cv::Mat_<cv::Vec3f> normal_image(height, width);
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      const int center = i*width+j;
      depth_image(i, j) = depth_data[center];
      normal_image(i, j) = cv::Vec3f(normal_data[center*3], normal_data[center*3+1], normal_data[center*3+2]);
    }
  }
  cv::Mat inv_depth_display, norm_display;
  ConvertDepthForDisplay(depth_image, inv_depth_display);
  ConvertNormalsForDisplay(normal_image, norm_display);
  cv::imshow("inv_depth_display", inv_depth_display);
  cv::imshow("norm_display", norm_display); 
	while(1)
	{
		if(cv::waitKey(0) == 'q')
			break;
	}
}