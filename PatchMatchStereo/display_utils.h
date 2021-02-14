#ifndef DISPLAY_UTILS_H
#define DISPLAY_UTILS_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "data_types.h"

void ConvertDisparityForDisplay(cv::Mat &disp, cv::Mat &disp_gray);

void ConvertDepthForDisplay(cv::Mat &inv_depth, cv::Mat &inv_depth_color);

void ConvertNormalsForDisplay(const cv::Mat &normals, cv::Mat &normals_display);

void ConvertCostForDisplay(const cv::Mat cost, cv::Mat &cost_display);

void ConvertGradientForDisplay(const cv::Mat &grad, cv::Mat &grad_display);

void ComputeHistForDisplay(const cv::Mat& img, cv::Mat &hist_display);

void ConvertPlaneToDepthAndNormal(const InvDepthPlane *plane, cv::Mat_<float>& inv_depth, cv::Mat_<cv::Vec3f>& norm, int height, int width);

void ConvertPlaneToDepthAndNormal(const PlaneState *plane, cv::Mat& depth, cv::Mat& normal, int height, int width);

void ShowDepthAndNormal(const PlaneState *plane_data, int height, int width);

void ShowCostAndHistogram(float *cost, int height, int width);

void RenderDepthAndNormalMap(const float *depth_data, const float *normal_data, const int height, const int width);

#endif