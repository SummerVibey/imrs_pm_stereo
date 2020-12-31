#include <iostream>
#include <chrono>
#include <stdio.h>
#include <dirent.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
// #include <opencv2/core/eigen.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>


#include "patch_match_cuda.h"
#include "device_utils.h"
#include "multi_view_stereo.h"
#include "mvs_matcher_wrapper.h"
#include "viewer.h"

inline void Mat33DotVec3(const float mat[9], const float vec[3], float result[3]) 
{
  result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

inline void Vec3AddVec3(const float vec1[9], const float vec2[3], float result[3]) 
{
  result[0] = vec1[0] + vec2[0];
  result[1] = vec1[1] + vec2[1];
  result[2] = vec1[2] + vec2[2];
}

inline void Warp(const float K1[9], const float K2[9], const float R21[9], const float t21[3], const float x1, const float y1, const float idp1, float &x2, float &y2)
{
  const float &ref_fx = K1[0], &ref_fy = K1[4],
              &ref_cx = K1[2], &ref_cy = K1[5];

  const float ref_inv_fx = 1 / ref_fx, ref_inv_cx = -ref_cx / ref_fx,
              ref_inv_fy = 1 / ref_fy, ref_inv_cy = -ref_cy / ref_fy;
  
  const float depth1 = 1 / idp1;
  const float pt3d[3] = {depth1 * (ref_inv_fx * x1 + ref_inv_cx), depth1 * (ref_inv_fy * y1 + ref_inv_cy), depth1};

  float Rp[3], Rpt[3], pt2d[3];
  Mat33DotVec3(R21, pt3d, Rp);
  Vec3AddVec3(Rp, t21, Rpt);
  Mat33DotVec3(K2, Rpt, pt2d);
  x2 = pt2d[0] / pt2d[2];
  y2 = pt2d[1] / pt2d[2];
}

int main(int argc, char** argv)
{

if (argc < 3) {
		std::cout << "�������٣�������ָ������Ӱ��·����" << std::endl;
		return -1;
	}

	printf("Image Loading...");
	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ��ȡӰ��
	std::string path_left = argv[1];
	std::string path_right = argv[2];
  std::string calib_file = argv[3];

	cv::Mat img_left = cv::imread(path_left, cv::IMREAD_GRAYSCALE);
	cv::Mat img_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

  img_left.convertTo(img_left, CV_32F);
  img_right.convertTo(img_right, CV_32F);

	// if (img_left.data == nullptr || img_right.data == nullptr) {
	// 	std::cout << "��ȡӰ��ʧ�ܣ�" << std::endl;
	// 	return -1;
	// }
	// if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
	// 	std::cout << "����Ӱ��ߴ粻һ�£�" << std::endl;
	// 	return -1;
	// }

  cv::FileStorage calib(calib_file, cv::FileStorage::READ);
  cv::Mat P0 = cv::Mat_<float>(3, 4), P1 = cv::Mat_<float>(3, 4);
  calib["P0"] >> P0;
  calib["P1"] >> P1;
  int width, height;
  float bf;
  calib["width"] >> width;
  calib["height"] >> height;
  calib["bf"] >> bf;
  SelectCudaDevice();

  cv::Mat K = P0.rowRange(0, 3).colRange(0, 3).clone();
  cv::Mat R1 = cv::Mat::eye(3, 3, CV_32F), R2 = cv::Mat::eye(3, 3, CV_32F);
  cv::Mat t1 = cv::Mat::zeros(3, 1, CV_32F), t2 = cv::Mat::zeros(3, 1, CV_32F);
  t2.at<float>(0) = -bf / K.at<float>(0,0);

  std::cout << K << std::endl;
  std::cout << R1 << std::endl;
  std::cout << t1 << std::endl;
  std::cout << R2 << std::endl;
  std::cout << t2 << std::endl;

  // float x1 = 101.4, y1 = 167.3, depth = 25.3, x2, y2;
  // cv::Mat R21 = R2 * R1.inv();
  // cv::Mat t21 = -R21 * t1 + t2;
  // Warp((float*)K.data, (float*)K.data, (float*)R21.data, (float*)t21.data, x1, y1, 1/depth, x2, y2);
  // std::cout << "my_result: " << x2 << "  " << y2 << std::endl;

  // cv::Mat tfk = K * R21 * K.inv();
  // cv::Mat tfb = K * t21;
  // cv::Mat pimg1 = cv::Mat_<float>(3, 1, CV_32F);
  // pimg1.at<float>(0) = x1, pimg1.at<float>(1) = y1, pimg1.at<float>(2) = 1;
  // cv::Mat pimg2 = tfk * pimg1 * depth + tfb;
  // pimg2 /= pimg2.at<float>(2);
  // std::cout << "eigen_result: " << pimg2.at<float>(0) << "  " << pimg2.at<float>(1) << std::endl;

  // int ref_index = 6;
  // int src_size = 2;

  PatchMatchOptions *options = new PatchMatchOptions();
  options->sigma_c = 10.0f;
  options->sigma_s = 15.0f;
  options->patch_size = 15;
  options->max_disparity = 0.4f;
  MVSMatcherWrapper *mvs_matcher = new MVSMatcherWrapper(options, height, width);
  mvs_matcher->SetReferenceView(img_left, K, R1, t1);
  mvs_matcher->AddSourceView(img_right, K, R2, t2);
  mvs_matcher->Initialize();
  
  cv::Mat  depth_img, normal_img;
  mvs_matcher->Run(depth_img, normal_img);

  cv::Mat color = cv::imread(path_left, cv::IMREAD_COLOR);
  // // cv::pyrDown(color, color, cv::Size(color.cols/2, color.rows/2));
  // color.resize(depth_img.rows, depth_img.cols);
  ConvertDepthToCloud(color, depth_img, K, height, width);

  delete mvs_matcher;
  delete options;

}

// #include <iostream>
// #include <chrono>
// #include <stdio.h>
// #include <dirent.h>
// #include <cstring>
// #include <sstream>
// #include <iomanip>
// #include <time.h>
// #include <stdio.h>

// #include <opencv2/core/core.hpp>
// // #include <opencv2/core/eigen.hpp>

// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Geometry>
// #include <eigen3/Eigen/Dense>
// #include <eigen3/Eigen/Eigen>


// #include "patch_match_cuda.h"
// #include "device_utils.h"
// #include "multi_view_stereo.h"
// #include "mvs_matcher_wrapper.h"
// #include "viewer.h"

// #include <dirent.h>
// #include <cstring>
// #include <sstream>
// #include <iomanip>
// #include <time.h>
// #include <stdio.h>

// inline void ComputeRelativePose(
//   const float R1[9], const float t1[3], // ref pose
//   const float R2[9], const float t2[3], // src pose 
//   float R21[9], float t21[3])           // relative pose ref to src 
// {
//   R21[0] = R2[0] * R1[0] + R2[1] * R1[1] + R2[2] * R1[2];
//   R21[1] = R2[0] * R1[3] + R2[1] * R1[4] + R2[2] * R1[5];
//   R21[2] = R2[0] * R1[6] + R2[1] * R1[7] + R2[2] * R1[8];
//   R21[3] = R2[3] * R1[0] + R2[4] * R1[1] + R2[5] * R1[2];
//   R21[4] = R2[3] * R1[3] + R2[4] * R1[4] + R2[5] * R1[5];
//   R21[5] = R2[3] * R1[6] + R2[4] * R1[7] + R2[5] * R1[8];
//   R21[6] = R2[6] * R1[0] + R2[7] * R1[1] + R2[8] * R1[2];
//   R21[7] = R2[6] * R1[3] + R2[7] * R1[4] + R2[8] * R1[5];
//   R21[8] = R2[6] * R1[6] + R2[7] * R1[7] + R2[8] * R1[8];

//   t21[0] = - R21[0] * t1[0] - R21[1] * t1[1] - R21[2] * t1[2] + t2[0];
//   t21[1] = - R21[3] * t1[0] - R21[4] * t1[1] - R21[5] * t1[2] + t2[1];
//   t21[2] = - R21[6] * t1[0] - R21[7] * t1[1] - R21[8] * t1[2] + t2[2];
// }

// void PrintMat3f(const float mat[9])
// {
//   for(int i = 0; i < 9; ++i) {
//     std::cout << mat[i] << "  ";
//     if(i % 3 == 2)
//       std::cout << std::endl;
//   }
// }

// void PrintVec3f(const float vec[3])
// {
//   std::cout << vec[0] << "  " << vec[1] << "  " << vec[2] << std::endl;
// }

// inline void ComputeHomography(
//   const float *K_ref, 
//   const float *K_src, 
//   const float *R1,
//   const float *R2,
//   const float *t1,
//   const float *t2,
//   const float x, 
//   const float y, 
//   const float idp, 
//   const float normal[3],
//   float H[9])
// {
//   const float &ref_fx = K_ref[0], &ref_fy = K_ref[4],
//               &ref_cx = K_ref[2], &ref_cy = K_ref[5];

//   const float &src_fx = K_src[0], &src_fy = K_src[4],
//               &src_cx = K_src[2], &src_cy = K_src[5];

//   const float ref_inv_fx = 1 / ref_fx, ref_inv_cx = -ref_cx / ref_fx,
//               ref_inv_fy = 1 / ref_fy, ref_inv_cy = -ref_cy / ref_fy;
  
//   float depth = 1/idp;
//   const float pt3d[3] = {depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth};
//   std::cout << "my xyz: " << pt3d[0] << "  " << pt3d[1] << "  " << pt3d[2] << std::endl;

//   // Distance to the plane.
//   const float dist = normal[0] * pt3d[0] + normal[1] * pt3d[1] + normal[2] * pt3d[2];
//   std::cout << "my dist: " << dist << std::endl;
//   const float inv_dist = 1.0f / dist;
//   const float inv_dist_N0 = inv_dist * normal[0];
//   const float inv_dist_N1 = inv_dist * normal[1];
//   const float inv_dist_N2 = inv_dist * normal[2];

//   // Relative Pose from ref to src
//   float R21[9], t21[3];
//   ComputeRelativePose(R1, t1, R2, t2, R21, t21);

//   H[0] = ref_inv_fx * (src_fx * (R21[0] + inv_dist_N0 * t21[0]) +
//                          src_cx * (R21[6] + inv_dist_N0 * t21[2]));
//   H[1] = ref_inv_fy * (src_fx * (R21[1] + inv_dist_N1 * t21[0]) +
//                          src_cx * (R21[7] + inv_dist_N1 * t21[2]));
//   H[2] = src_fx * (R21[2] + inv_dist_N2 * t21[0]) +
//          src_cx * (R21[8] + inv_dist_N2 * t21[2]) +
//          ref_inv_cx * (src_fx * (R21[0] + inv_dist_N0 * t21[0]) +
//                          src_cx * (R21[6] + inv_dist_N0 * t21[2])) +
//          ref_inv_cy * (src_fx * (R21[1] + inv_dist_N1 * t21[0]) +
//                          src_cx * (R21[7] + inv_dist_N1 * t21[2]));
//   H[3] = ref_inv_fx * (src_fy * (R21[3] + inv_dist_N0 * t21[1]) +
//                          src_cy * (R21[6] + inv_dist_N0 * t21[2]));
//   H[4] = ref_inv_fy * (src_fy * (R21[4] + inv_dist_N1 * t21[1]) +
//                          src_cy * (R21[7] + inv_dist_N1 * t21[2]));
//   H[5] = src_fy * (R21[5] + inv_dist_N2 * t21[1]) +
//          src_cy * (R21[8] + inv_dist_N2 * t21[2]) +
//          ref_inv_cx * (src_fy * (R21[3] + inv_dist_N0 * t21[1]) +
//                          src_cy * (R21[6] + inv_dist_N0 * t21[2])) +
//          ref_inv_cy * (src_fy * (R21[4] + inv_dist_N1 * t21[1]) +
//                          src_cy * (R21[7] + inv_dist_N1 * t21[2]));
//   H[6] = ref_inv_fx * (R21[6] + inv_dist_N0 * t21[2]);
//   H[7] = ref_inv_fy * (R21[7] + inv_dist_N1 * t21[2]);
//   H[8] = R21[8] + ref_inv_cx * (R21[6] + inv_dist_N0 * t21[2]) +
//          ref_inv_cy * (R21[7] + inv_dist_N1 * t21[2]) + inv_dist_N2 * t21[2];

// }

// inline void ComputeHomographyEigen(
//   const Eigen::Matrix3f &K_ref, 
//   const Eigen::Matrix3f &K_src, 
//   const Eigen::Matrix4f &T21,
//   const float x, 
//   const float y, 
//   const float depth, 
//   const Eigen::Vector3f &normal,
//   Eigen::Matrix3f& H)
// {
//   const float &ref_fx = K_ref(0,0), &ref_fy = K_ref(1,1),
//               &ref_cx = K_ref(0,2), &ref_cy = K_ref(1,2);

//   const float ref_inv_fx = 1 / ref_fx, ref_inv_cx = -ref_cx / ref_fx,
//               ref_inv_fy = 1 / ref_fy, ref_inv_cy = -ref_cy / ref_fy;

//   // const float pt3d[3] = {depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth};
//   Eigen::Vector3f pt3d(depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth);
//   std::cout << "eigen xyz: " << pt3d.transpose() << std::endl;
//   // Distance to the plane.
//   // const float dist = pt3d.dot(normal);
//   const float dist = pt3d.transpose() * normal;
//   std::cout << "eigen dist: " << dist << std::endl;
//   Eigen::Matrix3f R21 = T21.block(0,0,3,3);
//   Eigen::Vector3f t21 = T21.block(0,3,3,1);
//   H = K_src * (R21 + t21 * normal.transpose()/dist) * K_ref.inverse(); H /= H(2,2);

// }

// void Compare(Eigen::Vector3f &uv1, Eigen::Matrix3f &K, Eigen::Matrix3f &R21, Eigen::Vector3f &t21, float depth, Eigen::Vector3f &normal)
// {
//   std::cout << "K inv: " << std::endl << K.inverse() << std::endl;
//   std::cout << "uv1: " << std::endl << uv1 << std::endl;
//   std::cout << "depth: " << std::endl << depth << std::endl;
//   Eigen::Vector3f xyz1 = K.inverse() * uv1 * depth;
//   float dist = xyz1.transpose() * normal;
//   std::cout << "cpa xyz: " << xyz1.transpose() << std::endl;
//   std::cout << "cpa dist: " << dist << std::endl;
//   Eigen::Matrix3f tf1 = depth * K * (R21 + t21 * normal.transpose()/dist) * K.inverse(); tf1/=tf1(2,2);
//   Eigen::Vector3f bias1 = Eigen::Vector3f::Zero();
//   std::cout << "tf1: " << std::endl << tf1 << std::endl;
//   std::cout << "bias1: " << std::endl << bias1 << std::endl;
//   Eigen::Matrix3f tf2 = K * R21 * K.inverse() * depth;
//   Eigen::Vector3f bias2 = K * t21;
//   std::cout << "tf2: " << std::endl << tf2 << std::endl;
//   std::cout << "bias2: " << std::endl << bias2 << std::endl;
//   Eigen::Vector3f result1 = tf1 * uv1 + bias1; result1/=result1(2);
//   Eigen::Vector3f result2 = tf2 * uv1 + bias2; result2/=result2(2);
//   std::cout << "result1: " << std::endl << result1 << std::endl;
//   std::cout << "result2: " << std::endl << result2 << std::endl;

// }

// inline void HomogeneousWarp(const float mat[9], const float px, const float py, float &qx, float &qy) 
// {
//   const float inv_z = 1.0f / (mat[6] * px + mat[7] * py + mat[8]);
//   qx = inv_z * (mat[0] * px + mat[1] * py + mat[2]);
//   qy = inv_z * (mat[3] * px + mat[4] * py + mat[5]);
// }

// Eigen::Vector3f warp(Eigen::Vector3f& uv, Eigen::Matrix3f& K, Eigen::Matrix3f& R, Eigen::Vector3f& t, float depth)
// {
//   Eigen::Vector3f pc = K.inverse() * uv * depth;
//   return K * (R * pc + t);
// }

// void ReadMVGFiles(const std::string& filename, std::vector<cv::Mat_<float>> &Rcws, std::vector<cv::Mat_<float>> &tcws)
// {
//   std::fstream in;
//   in.open(filename);
// 	if (!in.is_open()) {
// 		std::cout << "Can not find " << filename << std::endl;
// 		return ;
// 	}
// 	std::string buff;
//   std::vector<cv::Mat_<float>> poses;
// 	int i = 0;
// 	while(getline(in, buff)) {
// 		std::vector<float> data;
// 		char *s_input = (char *)buff.c_str();
// 		const char *split = " ";
// 		char *p = strtok(s_input, split);
// 		float a;
// 		while (p != NULL) {
// 			a = atof(p);
// 			data.push_back(a);
// 			p = strtok(NULL, split);
//     }
//     cv::Mat_<float> Rcw(3, 3), tcw(3, 1); 
//     memcpy((float*)Rcw.data, &data[0], sizeof(float) * 9);
//     memcpy((float*)tcw.data, &data[9], sizeof(float) * 3);
//     Rcws.push_back(Rcw);
//     tcws.push_back(tcw);
// 		i++;
// 	}
// 	in.close();
// }

// std::vector<std::string> listDirectoryFiles(std::string& folder_path) 
// {
//   if(folder_path[folder_path.length()-1] != '/') {
//     folder_path += "/";
//   }
//   std::vector<std::string> file_list; 
//   DIR *dir;
//   struct dirent *ent;
//   if ((dir = opendir (folder_path.c_str())) != NULL) {
//     /* print all the files and directories within directory */
//     while ((ent = readdir (dir)) != NULL) {
//       if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
//         continue;
//       file_list.push_back(ent->d_name);      
//     }
//     closedir(dir);
//   } else {
//     /* could not open directory */
//     std::runtime_error("unable to list files in the selected directory");
//   }
//   std::sort(file_list.begin(), file_list.end(), [&](const std::string& file1, const std::string& file2){ return file1 < file2; });
//   return file_list;
// }

// const int ref_idx = 0, src_idx = 3;

// int main(int argc, char** argv)
// {

// if (argc < 4) {
// 		std::cout << "�������٣�������ָ������Ӱ��·����" << std::endl;
// 		return -1;
// 	}

// 	printf("Image Loading...");
// 	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
// 	// ��ȡӰ��
// 	std::string img_folder = argv[1];
//   std::string calib_file = argv[2];
//   std::string pose_file = argv[3];

//   std::vector<std::string> img_filenames = listDirectoryFiles(img_folder);
//   std::string path_left = img_folder + img_filenames[ref_idx], path_right = img_folder + img_filenames[src_idx];

// 	cv::Mat img_left = cv::imread(path_left, cv::IMREAD_GRAYSCALE);
// 	cv::Mat img_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

//   img_left.convertTo(img_left, CV_32F);
//   img_right.convertTo(img_right, CV_32F);

//   cv::FileStorage calib(calib_file, cv::FileStorage::READ);
//   cv::Mat K = cv::Mat_<float>(3, 3);
//   calib["camera_matrix"] >> K;

//   std::vector<cv::Mat_<float>> Rs;
//   std::vector<cv::Mat_<float>> ts;
//   ReadMVGFiles(pose_file, Rs, ts);
//   assert(img_filenames.size() == Rs.size() && img_filenames.size() == ts.size());
//   SelectCudaDevice();

//   PatchMatchOptions *options = new PatchMatchOptions();
//   options->sigma_c = 12.0f;
//   options->sigma_s = 12.0f;
//   options->patch_size = 35;
//   options->min_disparity = 1 / 50.0f;
//   options->max_disparity = 1 / 1.0f;
//   MVSMatcherWrapper *mvs_matcher = new MVSMatcherWrapper(options, img_left.rows, img_left.cols);
//   mvs_matcher->SetReferenceView(img_left, K, Rs[ref_idx], ts[ref_idx]);
//   mvs_matcher->AddSourceView(img_right, K, Rs[src_idx], ts[src_idx]);
//   mvs_matcher->Initialize();
  
//   cv::Mat  depth_img, normal_img;
//   mvs_matcher->Run(depth_img, normal_img);

//   // cv::Mat color = cv::imread(path_left, cv::IMREAD_COLOR);
//   // // // cv::pyrDown(color, color, cv::Size(color.cols/2, color.rows/2));
//   // // color.resize(depth_img.rows, depth_img.cols);
//   // ConvertDepthToCloud(color, depth_img, K, img_left.rows, img_left.cols);

//   delete mvs_matcher;
//   delete options;

// }