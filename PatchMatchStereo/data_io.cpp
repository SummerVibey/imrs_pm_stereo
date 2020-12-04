#include "data_io.h"

void ConvertPlaneToDispAndNormal(DispPlane *plane, cv::Mat_<float>& disp, cv::Mat_<cv::Vec3f>& norm, int height, int width)
{
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      disp(i, j)  = plane[i*width+j].to_disparity(j, i);
      Vector3f norm_ij = plane[i*width+j].to_normal();
      norm(i, j) = cv::Vec3f(norm_ij.x, norm_ij.y, norm_ij.z);
    }
  }
}
