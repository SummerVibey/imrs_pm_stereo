#include <iostream>
#include <chrono>
#include <stdio.h>
#include <dirent.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <time.h>
#include <stdio.h>
#include <random>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>

#include <pcl-1.8/pcl/io/pcd_io.h>
#include <pcl-1.8/pcl/point_types.h>
#include <pcl-1.8/pcl/point_cloud.h>
#include <pcl-1.8/pcl/features/normal_3d.h>
#include <pcl-1.8/pcl/visualization/cloud_viewer.h>


// #include "patch_match_cuda.h"
// #include "device_utils.h"
// #include "multi_view_stereo.h"
// #include "mvs_matcher_wrapper.h"

// ------------------------------------------------------------------
// parameters
const int boarder = 20;         // 边缘宽度
const int width = 640;          // 图像宽度
const int height = 480;         // 图像高度
const double fx = 481.2f;       // 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 10;    // NCC 取的窗口半宽度
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC窗口面积

struct PatchMatchOptions
{
	
	float min_disparity;		// ��С�Ӳ�
	float	max_disparity;		// ����Ӳ�
  float min_depth;
  float max_depth;

  int	patch_size;			// patch�ߴ磬�ֲ�����Ϊ patch_size*patch_size
  int step_size;
	float	sigma_spatial;			//
  float sigma_color;       
	float	alpha;				// alpha ���ƶ�ƽ������
	float	tau_col;			// tau for color	���ƶȼ�����ɫ�ռ�ľ��Բ���½ض���ֵ
	float	tau_grad;			// tau for gradient ���ƶȼ����ݶȿռ�ľ��Բ��½ض���ֵ

	int	num_iters;			// ������������

	bool	is_check_lr;		// �Ƿ�������һ����
	float	lrcheck_thres;		// ����һ����Լ����ֵ

	bool	is_fill_holes;		// �Ƿ�����Ӳ�ն�

	bool	is_fource_fpw;		// �Ƿ�ǿ��ΪFrontal-Parallel Window
	bool	is_integer_disp;	// �Ƿ�Ϊ�������Ӳ�

	PatchMatchOptions() : 
    min_disparity(0.0f), max_disparity(0.5f), min_depth(0.1f), max_depth(10.0f),
    patch_size(7), step_size(1), sigma_spatial(3.0f), sigma_color(0.2f), alpha(0.9f), tau_col(10.0f),
    tau_grad(2.0f), num_iters(3), is_check_lr(false), lrcheck_thres(0),
    is_fill_holes(false), is_fource_fpw(false), is_integer_disp(false) {}
	
	PatchMatchOptions(float _min_depth, float _max_depth) : 
    min_disparity(0.0f), max_disparity(0.5f), min_depth(_min_depth), max_depth(_max_depth),
    patch_size(21), step_size(1), sigma_spatial(3.0f), sigma_color(0.2f), alpha(0.9f), tau_col(10.0f),
    tau_grad(2.0f), num_iters(3), is_check_lr(false), lrcheck_thres(0),
    is_fill_holes(false), is_fource_fpw(false), is_integer_disp(false) {}

  void Print()
  {
    printf("Patch Match Stereo Matcher Options\n");
    printf("----------------------------------\n");
    printf("   patch_size: %d\n", patch_size);
    printf("    step_size: %d\n", step_size);
    printf("min_disparity: %f\n", min_disparity);
    printf("max_disparity: %f\n", max_disparity);
    printf("    min_depth: %f\n", min_depth);
    printf("    max_depth: %f\n", max_depth);
    printf("sigma_spatial: %f\n", sigma_spatial);
    printf("  sigma_color: %f\n", sigma_color);
    printf("        alpha: %f\n", alpha);
    printf("      tau_col: %f\n", tau_col);
    printf("     tau_grad: %f\n", tau_grad);
  }
};

struct View
{
  unsigned long id;
  int height, width;
  Eigen::Matrix3d K;
  Eigen::Matrix3d Rcw;
  Eigen::Vector3d tcw;

  cv::Mat gray_map;
  cv::Mat depth_map;
  cv::Mat normal_map;
  cv::Mat cost_map;

  View(const int _id, const cv::Mat& _gray_map, const Eigen::Matrix3d& _K, const Eigen::Matrix3d& _R, const Eigen::Vector3d& _t)
  {
    id = _id;
    height = _gray_map.rows;
    width = _gray_map.cols;
    
    K = _K;
    Rcw = _R;
    tcw = _t;

    gray_map = _gray_map.clone();
    depth_map = cv::Mat(height, width, CV_32F);
    normal_map = cv::Mat(height, width, CV_32FC3);
    cost_map = cv::Mat(height, width, CV_32F);
  }
};


bool readDatasetFiles(
  const std::string &path,
  std::vector<std::string> &color_image_files,
  std::vector<cv::Mat> &Rcws,
  std::vector<cv::Mat> &tcws,
  cv::Mat &ref_depth) {
  std::ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
  if (!fin) return false;
  while (!fin.eof()) {
    // 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
    std::string image;
    fin >> image;
    double data[7];
    for (double &d:data) fin >> d;

    color_image_files.push_back(path + std::string("/images/") + image);

    Eigen::Matrix3f Rcw = Eigen::Quaternionf(data[6], data[3], data[4], data[5]).toRotationMatrix().transpose();
    cv::Mat Rcw_i = (cv::Mat_<float>(3, 3) << Rcw(0, 0), Rcw(0, 1), Rcw(0, 2), 
                                              Rcw(1, 0), Rcw(1, 1), Rcw(1, 2), 
                                              Rcw(2, 0), Rcw(2, 1), Rcw(2, 2));
    cv::Mat tcw_i = (cv::Mat_<float>(3, 1) << data[0], data[1], data[2]);
    Rcws.push_back(Rcw_i);
    tcws.push_back(tcw_i);
    if (!fin.good()) break;
  }
  fin.close();
  // load reference depth
  fin.open(path + "/depthmaps/scene_000.depth");
  ref_depth = cv::Mat(height, width, CV_64F);
  if (!fin) return false;
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      double depth = 0;
      fin >> depth;
      ref_depth.ptr<double>(y)[x] = depth / 100.0;
    }

  return true;
}

std::vector<std::string> ReadICLImageName(std::string& folder_path) 
{
  if(folder_path[folder_path.length()-1] != '/') {
    folder_path += "/";
  }
  std::vector<std::string> file_list; 
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (folder_path.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
        continue;
      file_list.push_back(ent->d_name);      
    }
    closedir(dir);
  } else {
    /* could not open directory */
    std::runtime_error("unable to list files in the selected directory");
  }
  std::sort(file_list.begin(), file_list.end(), [&](const std::string& file1, const std::string& file2){ return file1 < file2; });
  return file_list;
}

void ReadICLPose(const std::string filename, std::vector<Eigen::Matrix3d> &Rcws, std::vector<Eigen::Vector3d> &tcws)
{
	std::ifstream fin;
	fin.open(filename.c_str());
	if (!fin.is_open()) {
		cerr << "Failed to open file! " << filename << endl;
		return;
	}

	std::string line;
  Rcws.clear();
  tcws.clear();
	while (std::getline(fin, line) && !line.empty()) {
    double x, y, z, qx, qy, qz, qw;
		std::istringstream data(line);
		data >> x >> y >> z >> qx >> qy >> qz >> qw;

    Eigen::Matrix3d Rwc = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();
    Eigen::Vector3d twc(x, y, z);
    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = -Rwc.transpose() * twc;
    Rcws.push_back(Rcw);
    tcws.push_back(tcw);
    if (!fin.good()) break;
	}
	fin.close();
}

cv::Mat ReadICLDepthImage(const std::string filename, const int height, const int width)
{
  cv::Mat depth_map = cv::Mat(height, width, CV_64F);
  std::ifstream fin;
  fin.open(filename);
  for (int y = 0; y < height; y++)
    for (int x = 0; x < width; x++) {
      double depth = 0;
      fin >> depth;
      depth_map.ptr<double>(y)[x] = depth;
    }

  fin.close();
  return depth_map;
}

void Warp(const float x, const float y, const float depth, const Eigen::Matrix3d& K,
          const Eigen::Matrix3d& Rcw1, const Eigen::Matrix3d& Rcw2, const Eigen::Vector3d& tcw1, const Eigen::Vector3d& tcw2,
          float& xwp, float& ywp, float& depth_wp)
{
  Eigen::Vector3d p1(x, y, 1);
  Eigen::Vector3d pc1 = K.inverse() * p1 * depth;
  Eigen::Matrix3d R21 = Rcw2 * Rcw1.inverse();
  Eigen::Vector3d t21 = -R21 * tcw1 + tcw2;
  Eigen::Vector3d p2 = K * (R21 * pc1 + t21);
  depth_wp = p2(2);
  p2 /= depth_wp;
  xwp = p2(0);
  ywp = p2(1);
}

// 双线性灰度插值
inline double getBilinearInterpolatedValue(const cv::Mat &img, const Eigen::Vector2d &pt) {
  uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
  double xx = pt(0, 0) - floor(pt(0, 0));
  double yy = pt(1, 0) - floor(pt(1, 0));
  return ((1 - xx) * (1 - yy) * double(d[0]) +
          xx * (1 - yy) * double(d[1]) +
          (1 - xx) * yy * double(d[img.step]) +
          xx * yy * double(d[img.step + 1])) / 255.0;
}

double NCC(const cv::Mat &ref, const cv::Mat &curr,
  const Eigen::Vector2d &pt_ref, const Eigen::Vector2d &pt_curr) 
{
  // 零均值-归一化互相关
  // 先算均值
  double mean_ref = 0, mean_curr = 0;
  std::vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
  for (int x = -ncc_window_size; x <= ncc_window_size; x++)
    for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
      double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
      mean_ref += value_ref;

      double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Eigen::Vector2d(x, y));
      mean_curr += value_curr;

      values_ref.push_back(value_ref);
      values_curr.push_back(value_curr);
    }

  mean_ref /= ncc_area;
  mean_curr /= ncc_area;

  // 计算 Zero mean NCC
  double numerator = 0, demoniator1 = 0, demoniator2 = 0;
  for (int i = 0; i < values_ref.size(); i++) {
    double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
    numerator += n;
    demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
    demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
  }
  return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);   // 防止分母出现零
}

inline Eigen::Matrix3d ComputeHomographyEigen(
  const Eigen::Matrix3d &K_ref, 
  const Eigen::Matrix3d &K_src, 
  const Eigen::Matrix3d &R_ref,
  const Eigen::Vector3d &t_ref,
  const Eigen::Matrix3d &R_src,
  const Eigen::Vector3d &t_src,
  const int x, 
  const int y, 
  const float depth, 
  const Eigen::Vector3d &normal)
{
  const float &ref_fx = K_ref(0,0), &ref_fy = K_ref(1,1),
              &ref_cx = K_ref(0,2), &ref_cy = K_ref(1,2);

  const float ref_inv_fx = 1 / ref_fx, ref_inv_cx = -ref_cx / ref_fx,
              ref_inv_fy = 1 / ref_fy, ref_inv_cy = -ref_cy / ref_fy;

  // const float pt3d[3] = {depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth};
  Eigen::Vector3d pt3d(depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth);

  // Distance to the plane.
  const float dist = pt3d.dot(normal);
  Eigen::Matrix4d Tcw1; Tcw1.setIdentity();
  Tcw1.block<3, 3>(0, 0) = R_ref;
  Tcw1.block<3, 1>(0, 3) = t_ref;
  Eigen::Matrix4d Tcw2; Tcw2.setIdentity();
  Tcw2.block<3, 3>(0, 0) = R_src;
  Tcw2.block<3, 1>(0, 3) = t_src;
  Eigen::Matrix4d T21 = Tcw2 * Tcw1.inverse();

  Eigen::Matrix3d R21 = R_src * R_ref.inverse();
  Eigen::Vector3d t21 = -R21 * t_ref + t_src;
  Eigen::Matrix3d H = K_src * (T21.block<3, 3>(0, 0) + T21.block<3, 1>(0, 3) * normal.transpose() / dist) * K_ref.inverse();
  return H;
}


Eigen::Vector2d HomographyWarp(const Eigen::Matrix3d& H, const Eigen::Vector2d& pt)
{
  Eigen::Vector3d pt_wp = H * pt.homogeneous();
  pt_wp /= pt_wp(2);
  return pt_wp.head<2>();
}

double BilateralWeight(const double delta_gray, const double delta_dist, const double sigma_gray, const double sigma_spatial)
{
  return std::exp(- delta_gray * delta_gray / (2 * sigma_gray * sigma_gray) - delta_dist * delta_dist / (2 * sigma_spatial * sigma_spatial));
} 

double ComputeNCC(const View& view_ref, const View& view_src,
    const int x, const int y, const float depth, const cv::Vec3f normal,
    const int patch_size, const int window_step, const PatchMatchOptions *options) 
{
  const int max_cost = 2.0;
  const double min_var = 1e-5;
  const int patch_halfsize = patch_size / 2;
  const int patch_area = patch_size * patch_size;
  const cv::Point2i lt(x - patch_halfsize, y - patch_halfsize);
  const cv::Point2i rb(x + patch_halfsize, y + patch_halfsize); 
  if(lt.x < 0 || lt.y < 0 || lt.x + 1 >= view_ref.width || lt.y + 1 >= view_ref.height) return max_cost;
  if(rb.x < 0 || rb.y < 0 || rb.x + 1 >= view_ref.width || rb.y + 1 >= view_ref.height) return max_cost;

  Eigen::Vector3d normal_eg(normal(0), normal(1), normal(2));
  Eigen::Matrix3d H = ComputeHomographyEigen(view_ref.K, view_src.K, view_ref.Rcw, view_ref.tcw, view_src.Rcw, view_src.tcw, x, y, depth, normal_eg);

  double mean_ref = 0, mean_src = 0;
  for (int dx = -patch_halfsize; dx <= patch_halfsize; dx+=window_step) {
    for (int dy = -patch_halfsize; dy <= patch_halfsize; dy+=window_step) {

      Eigen::Vector2d ref_pt = Eigen::Vector2d(x + dx, y + dy);
      double value_ref = getBilinearInterpolatedValue(view_ref.gray_map, ref_pt);
      mean_ref += value_ref;

      Eigen::Vector2d src_pt = HomographyWarp(H, ref_pt);
      if(src_pt(0) < 0 || src_pt(1) < 0 || src_pt(0) + 1 >= view_src.width || src_pt(1) + 1 >= view_src.height) 
        return max_cost;
      
      double value_src = getBilinearInterpolatedValue(view_src.gray_map, src_pt);
      mean_src += value_src;
    }
  }
  mean_ref /= patch_area;
  mean_src /= patch_area;

  double ref_src_color_sum = 0, ref_color_squared_sum = 0, src_color_squared_sum = 0;
  for (int dx = -patch_halfsize; dx <= patch_halfsize; dx+=window_step) {
    for (int dy = -patch_halfsize; dy <= patch_halfsize; dy+=window_step) {

      Eigen::Vector2d ref_pt = Eigen::Vector2d(x + dx, y + dy);
      double value_ref = getBilinearInterpolatedValue(view_ref.gray_map, ref_pt);
      Eigen::Vector2d src_pt = HomographyWarp(H, ref_pt);
      if(src_pt(0) < 0 || src_pt(1) < 0 || src_pt(0) + 1 >= view_src.width || src_pt(1) + 1 >= view_src.height) 
        return max_cost;
      
      double value_src = getBilinearInterpolatedValue(view_src.gray_map, src_pt);

      // const double delta_gray = (value_ref - value_src) / 255.0;
      // const double delta_dist = std::sqrt(dx*dx + dy*dy);
      // const double weight = BilateralWeight(delta_gray, delta_dist, options->sigma_color, options->sigma_spatial);
      
      ref_src_color_sum += (value_ref - mean_ref) * (value_src - mean_src);
      ref_color_squared_sum += (value_ref - mean_ref) * (value_ref - mean_ref);
      src_color_squared_sum += (value_src - mean_src) * (value_src - mean_src);
    }
  }
  return 1 - ref_src_color_sum / (sqrt(ref_color_squared_sum * src_color_squared_sum) + 1e-10);

  //     const double delta_gray = (value_ref - value_src) / 255.0;
  //     const double delta_dist = std::sqrt(dx*dx + dy*dy);
  //     const double weight = BilateralWeight(delta_gray, delta_dist, options->sigma_color, options->sigma_spatial);

  //     bilateral_weight_sum += weight;
  //     ref_color_sum += weight * value_ref;
  //     ref_color_squared_sum += weight * value_ref * value_ref;
  //     src_color_sum += weight * value_src;
  //     src_color_squared_sum += weight * value_src * value_src;

  //     ref_src_color_sum += weight * value_ref * value_src;
  //   }

  // ref_color_sum /= bilateral_weight_sum;
  // ref_color_squared_sum /= bilateral_weight_sum;
  // src_color_sum /= bilateral_weight_sum;
  // src_color_squared_sum /= bilateral_weight_sum;
  // ref_src_color_sum /= bilateral_weight_sum;

  // const float ref_color_var =
  //     ref_color_squared_sum - ref_color_sum * ref_color_sum;
  // const float src_color_var =
  //     src_color_squared_sum - src_color_sum * src_color_sum;
  // const float src_ref_color_covar =
  //     ref_src_color_sum - ref_color_sum * src_color_sum;

  // if (ref_color_var < 1e-5 || src_color_var < 1e-5)
  //   return max_cost;

  // const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
  // double ncc = src_ref_color_covar / (src_ref_color_var + 1e-10);
  // if(ncc < -1.0) ncc = -1.0;
}



void ComputeNormalImage(const cv::Mat& depth_image, const cv::Mat& K, cv::Mat& normal_image)
{ 
  pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  for(int y = 0; y < height; ++y) {
    for(int x = 0; x < width; ++x) {
      cv::Mat point_img(3, 1, CV_32F);
      point_img.at<float>(0, 0) = x;
      point_img.at<float>(1, 0) = y;
      point_img.at<float>(2, 0) = 1;
      float depth = depth_image.at<float>(y, x);
      cv::Mat point_cam = K.inv() * point_img * depth;
      ref_cloud->push_back(pcl::PointXYZ(point_cam.at<float>(0, 0), point_cam.at<float>(1, 0), point_cam.at<float>(2, 0)));
    }
  }

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud (ref_cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
  ne.setSearchMethod(tree);
  ne.setKSearch(10);
  pcl::PointCloud<pcl::Normal>::Ptr ref_normals(new pcl::PointCloud<pcl::Normal>());
  ne.compute(*ref_normals);

  int cnt = 0;
  normal_image = cv::Mat(depth_image.rows, depth_image.cols, CV_32FC3);
  for(int y = 0; y < height; ++y) {
    for(int x = 0; x < width; ++x) {
      pcl::Normal norm = ref_normals->points[cnt];
      normal_image.at<cv::Vec3f>(y, x) = cv::Vec3f(norm.normal_x, norm.normal_y, norm.normal_z);
      cnt++;
    }
  }
}

void RenderDepthImage(const cv::Mat& depth, cv::Mat& depth_color)
{
  double max_val, min_val;
  cv::Point2i max_loc, min_loc;
  cv::minMaxLoc(depth, &min_val, &max_val, &min_loc, &max_loc);
  printf("max inv_depth = %f, min inv_depth = %f\n", (float)max_val, (float)min_val);
  printf("max loc = (%d, %d) min loc = (%d, %d)\n", max_loc.x, max_loc.y, min_loc.x, min_loc.y);

  cv::Mat depth_gray;
  depth.convertTo(depth_gray, CV_8U, 255.0f/(max_val), 0);
  cv::applyColorMap(depth_gray, depth_color, cv::COLORMAP_JET);
}

void RenderNormalImage(const cv::Mat &normals, cv::Mat &normals_display)
{
  normals.convertTo(normals_display,CV_16U,32767,32767);
	cv::cvtColor(normals_display,normals_display,cv::COLOR_RGB2BGR);
}

void RenderCostImage(const cv::Mat cost, cv::Mat &cost_display)
{
  double max_val, min_val;
  cv::Point2i max_loc, min_loc;
  cv::minMaxLoc(cost, &min_val, &max_val, &min_loc, &max_loc);
  printf("max cost = %f, min cost = %f\n", (float)max_val, (float)min_val);
  printf("max loc = (%d, %d) min loc = (%d, %d)\n", max_loc.x, max_loc.y, min_loc.x, min_loc.y);
  cost.convertTo(cost_display, CV_32F, 1.f/max_val,0);
}

void RenderCostHistogramImage(const cv::Mat& cost_img, cv::Mat &hist_display)
{
  int channels = 0;
  double max_val, min_val;
  cv::Point2i max_loc, min_loc;
  cv::minMaxLoc(cost_img, &min_val, &max_val, &min_loc, &max_loc);
  const int hist_size[] = {256};
  float min_ranges[] = {(float)min_val, (float)max_val}; 
  const float *ranges[] = { min_ranges }; 
  cv::MatND hist;
  cv::calcHist(&cost_img, 1, &channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);
	cv::minMaxLoc(hist, 0, &max_val, 0, 0);
	int rows = cvRound(max_val);
  cv::Mat hist_image = cv::Mat::zeros(rows, 256, CV_8UC1);
	for(int i = 0; i < 256; i++) {
		int temp = (int)(hist.at<float>(i, 0));
		if(temp) {
			hist_image.col(i).rowRange(cv::Range(rows - temp, rows)) = 255; 
    }
  }
	resize(hist_image,hist_display,cv::Size(512,512));
}



void RenderColorPointCloud(const cv::Mat& color_img, const cv::Mat& depth_img, const cv::Mat& K)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  for(int y = 0; y < height; ++y) {
    for(int x = 0; x < width; ++x) {
      float depth_ij = depth_img.ptr<float>(y)[x]; // 深度值
      if (depth_ij == 0) continue; // 为0表示没有测量到
      Eigen::Vector3d pc;
      pc[2] = depth_ij;
      pc[0] = (x - cx) * depth_ij / fx;
      pc[1] = (y - cy) * depth_ij / fy;
      pcl::PointXYZRGB point;
      point.x = pc[0];
      point.y = pc[1];
      point.z = pc[2];
      point.b = color_img.at<uchar>(y, x);
      point.g = color_img.at<uchar>(y, x);
      point.r = color_img.at<uchar>(y, x);
      rgb_cloud->push_back(point);
    }
  }

  pcl::visualization::CloudViewer viewer("viewer");
  viewer.showCloud(rgb_cloud);
  while(!viewer.wasStopped()) {}

}

void WarpImage(
  const cv::Mat& ref_color, 
  const cv::Mat& ref_depth,
  const Eigen::Matrix3d& K, 
  const Eigen::Matrix3d& Rcw1,
  const Eigen::Matrix3d& Rcw2,
  const Eigen::Vector3d& tcw1,
  const Eigen::Vector3d& tcw2,
  cv::Mat& wp_color,
  cv::Mat& wp_depth
)
{
  wp_color = cv::Mat(height, width, CV_8U, cv::Scalar(0));
  wp_depth = cv::Mat(height, width, CV_16U, cv::Scalar(65535));
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      uchar color = ref_color.at<uchar>(i, j);
      cv::Mat point_img(3, 1, CV_32F);
      float x = j, y = i, depth = (float)ref_depth.at<unsigned short>(i, j) / 5000.0f;
      float xwp, ywp, depth_wp;
      Warp(x, y, depth, K, Rcw1, Rcw2, tcw1, tcw2, xwp, ywp, depth_wp);
      if(xwp < 0 || xwp > width - 1 ||
        ywp < 0 || ywp > height - 1) {
        continue;
      } 
      const int x0 = std::floor(xwp);
      const int x1 = std::ceil(xwp);
      const int y0 = std::floor(ywp);
      const int y1 = std::ceil(ywp);

      if(depth_wp < wp_depth.at<float>(y0, x0)) {
        wp_color.at<uchar>(y0, x0) = color;
        wp_depth.at<float>(y0, x0) = depth_wp;
      }
        
      if(depth_wp < wp_depth.at<float>(y0, x1)) {
        wp_color.at<uchar>(y0, x1) = color;
        wp_depth.at<float>(y0, x1) = depth_wp;    
      }

      if(depth_wp < wp_depth.at<float>(y1, x0)) {
        wp_color.at<uchar>(y1, x0) = color;
        wp_depth.at<float>(y1, x0) = depth_wp;   
      }

      if(depth_wp < wp_depth.at<float>(y1, x1)) {
        wp_color.at<uchar>(y1, x1) = color;
        wp_depth.at<float>(y1, x1) = depth_wp;      
      }

    }
  }
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      if(wp_depth.at<float>(i, j) == 100.0f)  {
        wp_depth.at<float>(i, j) = 0.0f;
      }
    }
  }
}

void ComputeRelativePose(
  const float R1[9], const float t1[3], // ref pose
  const float R2[9], const float t2[3], // src pose 
  float R21[9], float t21[3])           // relative pose ref to src 
{
  R21[0] = R2[0] * R1[0] + R2[1] * R1[1] + R2[2] * R1[2];
  R21[1] = R2[0] * R1[3] + R2[1] * R1[4] + R2[2] * R1[5];
  R21[2] = R2[0] * R1[6] + R2[1] * R1[7] + R2[2] * R1[8];
  R21[3] = R2[3] * R1[0] + R2[4] * R1[1] + R2[5] * R1[2];
  R21[4] = R2[3] * R1[3] + R2[4] * R1[4] + R2[5] * R1[5];
  R21[5] = R2[3] * R1[6] + R2[4] * R1[7] + R2[5] * R1[8];
  R21[6] = R2[6] * R1[0] + R2[7] * R1[1] + R2[8] * R1[2];
  R21[7] = R2[6] * R1[3] + R2[7] * R1[4] + R2[8] * R1[5];
  R21[8] = R2[6] * R1[6] + R2[7] * R1[7] + R2[8] * R1[8];

  t21[0] = - R21[0] * t1[0] - R21[1] * t1[1] - R21[2] * t1[2] + t2[0];
  t21[1] = - R21[3] * t1[0] - R21[4] * t1[1] - R21[5] * t1[2] + t2[1];
  t21[2] = - R21[6] * t1[0] - R21[7] * t1[1] - R21[8] * t1[2] + t2[2];
}

void ComputeHomographyCuda(
  const float *K_ref, 
  const float *K_src, 
  const float *R1,
  const float *R2,
  const float *t1,
  const float *t2,
  const int x, 
  const int y, 
  const float idp, 
  const float normal[3],
  float H[9]
)
{
  const float &ref_fx = K_ref[0], &ref_fy = K_ref[4],
              &ref_cx = K_ref[2], &ref_cy = K_ref[5];

  const float &src_fx = K_src[0], &src_fy = K_src[4],
              &src_cx = K_src[2], &src_cy = K_src[5];

  const float ref_inv_fx = 1 / ref_fx, ref_inv_cx = -ref_cx / ref_fx,
              ref_inv_fy = 1 / ref_fy, ref_inv_cy = -ref_cy / ref_fy;
  
  const float depth = 1 / idp;
  const float pt3d[3] = {depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth};

  // Distance to the plane.
  const float dist = normal[0] * pt3d[0] + normal[1] * pt3d[1] + normal[2] * pt3d[2];
  const float inv_dist = 1.0f / dist;
  const float inv_dist_N0 = inv_dist * normal[0];
  const float inv_dist_N1 = inv_dist * normal[1];
  const float inv_dist_N2 = inv_dist * normal[2];

  // Relative Pose from ref to src
  float R21[9], t21[3];
  ComputeRelativePose(R1, t1, R2, t2, R21, t21);

  H[0] = ref_inv_fx * (src_fx * (R21[0] + inv_dist_N0 * t21[0]) +
                         src_cx * (R21[6] + inv_dist_N0 * t21[2]));
  H[1] = ref_inv_fy * (src_fx * (R21[1] + inv_dist_N1 * t21[0]) +
                         src_cx * (R21[7] + inv_dist_N1 * t21[2]));
  H[2] = src_fx * (R21[2] + inv_dist_N2 * t21[0]) +
         src_cx * (R21[8] + inv_dist_N2 * t21[2]) +
         ref_inv_cx * (src_fx * (R21[0] + inv_dist_N0 * t21[0]) +
                         src_cx * (R21[6] + inv_dist_N0 * t21[2])) +
         ref_inv_cy * (src_fx * (R21[1] + inv_dist_N1 * t21[0]) +
                         src_cx * (R21[7] + inv_dist_N1 * t21[2]));
  H[3] = ref_inv_fx * (src_fy * (R21[3] + inv_dist_N0 * t21[1]) +
                         src_cy * (R21[6] + inv_dist_N0 * t21[2]));
  H[4] = ref_inv_fy * (src_fy * (R21[4] + inv_dist_N1 * t21[1]) +
                         src_cy * (R21[7] + inv_dist_N1 * t21[2]));
  H[5] = src_fy * (R21[5] + inv_dist_N2 * t21[1]) +
         src_cy * (R21[8] + inv_dist_N2 * t21[2]) +
         ref_inv_cx * (src_fy * (R21[3] + inv_dist_N0 * t21[1]) +
                         src_cy * (R21[6] + inv_dist_N0 * t21[2])) +
         ref_inv_cy * (src_fy * (R21[4] + inv_dist_N1 * t21[1]) +
                         src_cy * (R21[7] + inv_dist_N1 * t21[2]));
  H[6] = ref_inv_fx * (R21[6] + inv_dist_N0 * t21[2]);
  H[7] = ref_inv_fy * (R21[7] + inv_dist_N1 * t21[2]);
  H[8] = R21[8] + ref_inv_cx * (R21[6] + inv_dist_N0 * t21[2]) +
         ref_inv_cy * (R21[7] + inv_dist_N1 * t21[2]) + inv_dist_N2 * t21[2];


}

cv::Vec3f GenerateRandomNormal(cv::Point2i pt, const Eigen::Matrix3d& K)
{
  float v1 = 0.0f;
  float v2 = 0.0f;
  float s = 2.0f;
  cv::RNG rng(cv::getTickCount());
  while (s >= 1.0f)
  {
    v1 = 2.0f * rng.uniform(0.f, 1.f) - 1.0f;
    v2 = 2.0f * rng.uniform(0.f, 1.f) - 1.0f;
    s = v1 * v1 + v2 * v2;
  }
  cv::Vec3f normal(0, 0, 0);
  const float s_norm = sqrt(1.0f - s);
  normal[0] = 2.0f * v1 * s_norm;
  normal[1] = 2.0f * v2 * s_norm;
  normal[2] = 1.0f - 2.0f * s;

  // Make sure normal is looking away from camera.
  cv::Vec3f view_ray((pt.x - K(0, 2)) / K(0, 0), // (col-cx)/fx
                      (pt.y - K(1, 2)) / K(1, 1), // (row-cy)/fy
                      1.f);
  if (normal.dot(view_ray) > 0)
  {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
  return normal;
}


void RandomInit(View &view, const PatchMatchOptions *options)
{
  cv::RNG rng(cv::getTickCount());
  rng.fill(view.depth_map, cv::RNG::UNIFORM, options->min_depth, options->max_depth);

// #pragma omp parallel for schedule(dynamic) 
  for (int m = 0; m < width; m++)
    for (int n = 0; n < height; n++) {
      cv::Point2i pt(m, n);
      view.normal_map.at<cv::Vec3f>(pt) = GenerateRandomNormal(pt, view.K);
    }
}

void CostInit(View &view_ref, View &view_src, const PatchMatchOptions *options)
{
// #pragma omp parallel for schedule(dynamic) 
  for(int y = 0; y < view_ref.height; ++y) {
    for(int x = 0; x < view_ref.width; ++x) {
      // std::cout << x << " " << y << std::endl;
      view_ref.cost_map.at<float>(y, x) = (float)ComputeNCC(view_ref, 
                                                                view_src, 
                                                                x, y, 
                                                                view_ref.depth_map.at<float>(y, x), 
                                                                view_ref.normal_map.at<cv::Vec3f>(y, x), 
                                                                21, 2, options);
      // std::cout << view_ref.cost_map.at<double>(y, x) << std::endl;
    }
  }


}

const int ref_id = 0;
const int src_id = 1;

int main(int argc, char** argv)
{

  clock_t start,end;
  if(argc < 3) {
    std::cout << "Usage: ./icl_demo <color_folder> <depth_folder> <dataset_calib_file> ..." << std::endl;
    return -1;
  } 
  std::string path_color = argv[1];
  std::string path_depth = argv[2];
  std::string calib_file = argv[3];

  cv::Mat K = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

  std::vector<Eigen::Matrix3d> Rcws;
  std::vector<Eigen::Vector3d> tcws;
  std::vector<std::string> color_filenames = ReadICLImageName(path_color);
  std::vector<std::string> depth_filenames = ReadICLImageName(path_depth);
  ReadICLPose(calib_file, Rcws, tcws);

  cv::Mat ref_color = cv::imread(path_color + color_filenames[ref_id], cv::IMREAD_GRAYSCALE);
  cv::Mat ref_depth = cv::imread(path_depth + depth_filenames[ref_id], cv::IMREAD_UNCHANGED);
  ref_depth.convertTo(ref_depth, CV_32F, 1 / 5000.0f, 0);

  // RenderColorPointCloud(ref_color, ref_depth, K);

  cv::Mat src_color = cv::imread(path_color + color_filenames[src_id], cv::IMREAD_GRAYSCALE);
  cv::Mat src_depth = cv::imread(path_depth + depth_filenames[src_id], cv::IMREAD_UNCHANGED);
  src_depth.convertTo(src_depth, CV_32F, 1 / 5000.0f, 0);

  cv::Mat ref_normal;
  ComputeNormalImage(ref_depth, K, ref_normal);


  // cv::imshow("color", ref_color);
  // cv::imshow("depth", ref_depth);
  cv::Mat normal_gt_color;
  RenderNormalImage(ref_normal, normal_gt_color);  
  


  Eigen::Matrix3d Ke;
  Ke << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  PatchMatchOptions *options = new PatchMatchOptions(0.1f, 10.0f);

  View view1(0, ref_color, Ke, Rcws[ref_id], tcws[ref_id]);
  View view2(1, src_color, Ke, Rcws[src_id], tcws[src_id]);

  cv::Mat depth_map, normal_map, cost_map;
  RandomInit(view1, options);
  RandomInit(view2, options);

  cv::Mat depth_color, normal_color;
  RenderDepthImage(view1.depth_map, depth_color);
  RenderNormalImage(view1.normal_map, normal_color);
  cv::imshow("depth", depth_color);
  cv::imshow("normal", normal_color);
  cv::imshow("normal_gt", normal_gt_color);


  CostInit(view1, view2, options);
  cv::Mat cost_display, hist_display;
  RenderCostImage(view1.cost_map, cost_display);
  RenderCostHistogramImage(view1.cost_map, hist_display);
  cv::imshow("cost", cost_display);
  cv::imshow("cost hist", hist_display);
  while (1)
  {
    if(cv::waitKey(0) == 'q') break;
  }
  // cv::Mat ref_color_show, src_color_show;
  // cv::cvtColor(ref_color, ref_color_show, cv::COLOR_GRAY2BGR);
  // cv::cvtColor(src_color, src_color_show, cv::COLOR_GRAY2BGR);

  // int x = 450, y = 250;
  // float depth = ref_depth.at<float>(y, x);
  // cv::Vec3f normal = ref_normal.at<cv::Vec3f>(y, x);

  // epipolar search
  // for(double depth = 0.1; depth < 1000; depth += 0.1) {
  //   float x_wp, y_wp, depth_wp;
  //   Warp(x, y, depth, Ke, Rcws[0], Rcws[1], tcws[0], tcws[1], x_wp, y_wp, depth_wp);
  //   int xs = (int)(x_wp + 0.5); int ys = (int)(y_wp + 0.5);
  //   if(xs > 0 && xs < width && ys > 0 && ys < height) {
  //     cv::circle(src_color_show, cv::Point2i(xs, ys), 0, cv::Scalar(0, 255, 0), -1);
  //   }
  //   std::cout << "depth: " << depth << std::endl;
  //   cv::imshow("show", src_color_show);
  //   while (1)
  //   {
  //     if(cv::waitKey(0) == 'q') break;
  //   }
  
  // }



  // // image warp
  // Eigen::Vector3d normal_mat(normal(0), normal(1), normal(2));
  // Eigen::Matrix3d H = ComputeHomographyEigen(Ke, Ke, Rcws[0], tcws[0], Rcws[1], tcws[1], x, y, depth, normal_mat);
  // std::cout << H << std::endl;

  // float ref_color_sum = 0, src_color_sum = 0, ref_color_squared_sum = 0, src_color_squared_sum = 0, ref_src_color_sum = 0, bilateral_weight_sum = 0;

  // for (int dx = -ncc_window_size; dx <= ncc_window_size; dx+=2)
  //   for (int dy = -ncc_window_size; dy <= ncc_window_size; dy+=2) {

  //     Eigen::Vector2d ref_pt = Eigen::Vector2d(x+dx, y+dy);
  //     double value_ref = getBilinearInterpolatedValue(ref_color, ref_pt);
  //     int xr = (int)(ref_pt(0) + 0.5); int yr = (int)(ref_pt(1) + 0.5);
  //     cv::circle(ref_color_show, cv::Point2i(xr, yr), 0, cv::Scalar(0, 255, 0), -1);

  //     Eigen::Vector2d src_pt = HomographyWarp(H, ref_pt);
  //     double value_src = getBilinearInterpolatedValue(src_color, src_pt);
  //     int xs = (int)(src_pt(0) + 0.5); int ys = (int)(src_pt(1) + 0.5);
  //     cv::circle(src_color_show, cv::Point2i(xs, ys), 0, cv::Scalar(0, 255, 0), -1);

  //     const double delta_gray = (value_ref - value_src) / 255.0;
  //     const double delta_dist = std::sqrt(dx*dx + dy*dy);
  //     // const double weight = BilateralWeight(delta_gray, delta_dist, options->sigma_color, options->sigma_spatial);
  //     const double weight = 1;
  //     bilateral_weight_sum += weight;
  //     ref_color_sum += weight * value_ref;
  //     ref_color_squared_sum += weight * value_ref * value_ref;
  //     src_color_sum += weight * value_src;
  //     src_color_squared_sum += weight * value_src * value_src;

  //     ref_src_color_sum += weight * value_ref * value_src;
  //     cv::imshow("ref", ref_color_show);
  //     cv::imshow("src", src_color_show);
  //     while (1)
  //     {
  //       if(cv::waitKey(0) == 'q') break;
  //     }
  //   }

  // double cost = 0;
  // ref_color_sum /= bilateral_weight_sum;
  // ref_color_squared_sum /= bilateral_weight_sum;
  // src_color_sum /= bilateral_weight_sum;
  // src_color_squared_sum /= bilateral_weight_sum;
  // ref_src_color_sum /= bilateral_weight_sum;

  // const float ref_color_var =
  //     ref_color_squared_sum - ref_color_sum * ref_color_sum;
  // const float src_color_var =
  //     src_color_squared_sum - src_color_sum * src_color_sum;
  // const float src_ref_color_covar =
  //     ref_src_color_sum - ref_color_sum * src_color_sum;

  // if (ref_color_var < 1e-5 || src_color_var < 1e-5)
  //   cost = 2.0;

  // const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
  // double ncc = src_ref_color_covar / (src_ref_color_var + 1e-10);
  // if(ncc < -1.0) ncc = -1.0;
  
  // cost = 1 - ncc;
  // std::cout << "cost: " << cost << std::endl;

  // std::cout << ComputeNCC(view1, view2, x, y, depth, normal, 11, 2, options) << std::endl;
  

  // CostInit(view1, view2, options);

  // // display color and depth
  // for(int i = 0; i < color_filenames.size(); ++i) {
  //   cv::Mat color = cv::imread(path_color + color_filenames[i], cv::IMREAD_UNCHANGED);
  //   cv::Mat depth = ReadICLDepthImage(path_depth + depth_filenames[i], 480, 640);
  //   cv::Mat depth_color;
  //   RenderDepthImage(depth, depth_color);
  //   cv::imshow("color", color);
  //   cv::imshow("depth", depth_color);
  //   cv::waitKey(1);
  // }

  // std::cout  << view1.Rcw << std::endl;
  // std::cout  << view2.Rcw << std::endl;

  // for(int y = 0; y < height; ++y) {
  //   for(int x = 0; x < width; ++x) {
  //     float depth_ij = ref_depth.at<float>(y, x);
  //     cv::Vec3f normal = ref_normal.at<cv::Vec3f>(y, x);
  //     // std::cout << "x: " << x << " y: " << y  << " depth: " << depth_ij << " normal: " << normal << std::endl;
  //     std::cout << ComputeNCC(view1, view2, x, y, depth_ij, normal, 15, 1, options) << std::endl;
  //   }
  //   while (1)
  //   {
  //     if(cv::waitKey(0) == 'q') break;
  //   }
  // }

  
  // cv::Mat wp_image, wp_depth;
  // // RenderColorPointCloud(ref_color, depth_gt, K);
  // WarpImage(ref_color, ref_depth, K, Rcws[ref_id], Rcws[cur_id], tcws[ref_id], tcws[cur_id], wp_image, wp_depth);

  // cv::Mat wp_depth_color;
  // RenderDepthImage(wp_depth, wp_depth_color);

  // cv::imshow("dst_image", wp_image);
  // cv::imshow("dst_depth", wp_depth_color);

  // RenderColorPointCloud(ref_color, ref_depth, K);
  return 0;
}