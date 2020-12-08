#ifndef VIEWER_H
#define VIEWER_H

#include <pcl-1.8/pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <pcl-1.8/pcl/visualization/cloud_viewer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include "data_types.h"

void ConvertDepthToCloud(const cv::Mat &color, const cv::Mat &depth, const cv::Mat &K, int height, int width)
{
  const float &fx = K.at<float>(0,0), &fy = K.at<float>(1,1), &cx = K.at<float>(0,2), &cy = K.at<float>(1,2);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  for(int y = 0; y < height; ++y) {
    for(int x = 0; x < width; ++x) {
      float xc, yc, zc;
      const float depth_ij = depth.at<float>(y,x);
      if(depth_ij > 20) continue;
      xc = depth_ij * (x - cx) / fx;
      yc = depth_ij * (y - cy) / fy;
      zc = depth_ij;
      pcl::PointXYZRGB point;
      point.x = xc;
      point.y = yc;
      point.z = zc;
      point.b = color.at<cv::Vec3b>(y,x)[0];
      point.g = color.at<cv::Vec3b>(y,x)[1];
      point.r = color.at<cv::Vec3b>(y,x)[2];
      rgb_cloud->points.push_back(point);
    }
  }

  pcl::visualization::CloudViewer viewer("Cloud Viewer");
 
  //blocks until the cloud is actually rendered
  viewer.showCloud(rgb_cloud);
  while (!viewer.wasStopped ()) {}


}

#endif