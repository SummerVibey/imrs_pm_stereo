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
  cv::Mat R0 = cv::Mat::eye(3, 3, CV_32F), R1 = cv::Mat::eye(3, 3, CV_32F);
  cv::Mat t0 = cv::Mat::zeros(3, 1, CV_32F), t1 = cv::Mat::zeros(3, 1, CV_32F);
  t1.at<float>(0) = -bf / K.at<float>(0,0);

  std::cout << K << std::endl;
  std::cout << R0 << std::endl;
  std::cout << t0 << std::endl;
  std::cout << R1 << std::endl;
  std::cout << t1 << std::endl;
  // int ref_index = 6;
  // int src_size = 2;

  PatchMatchOptions *options = new PatchMatchOptions();
  MVSMatcherWrapper *mvs_matcher = new MVSMatcherWrapper(options);
  mvs_matcher->SetReferenceView(img_left, K, R0, t0);
  mvs_matcher->AddSourceView(img_right, K, R1, t1);
  mvs_matcher->Initialize();
  mvs_matcher->Run();

  delete mvs_matcher;
  delete options;

}