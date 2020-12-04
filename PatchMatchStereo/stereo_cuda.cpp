/* -*-c++-*- PatchMatchStereo - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
*			  https://github.com/ethan-li-coding
* Describe	: main
*/


#include <iostream>
#include <chrono>
#include <stdio.h>

#include "patch_match_cuda.h"
#include "device_utils.h"

using namespace std::chrono;

// opencv library
#include <opencv2/opencv.hpp>

/*��ʾ�Ӳ�ͼ*/
void ShowDisparityMap(const float* disp_map, const int& width, const int& height, const std::string& name);
/*�����Ӳ�ͼ*/
void SaveDisparityMap(const float* disp_map, const int& width, const int& height, const std::string& path);
/*�����Ӳ����*/
void SaveDisparityCloud(const unsigned char* img_bytes, const float* disp_map, const int& width, const int& height, const std::string& path);


void ConstructTexture(cv::Mat& img, cudaTextureObject_t& tex, cudaArray *arr)
{
  assert(img.type() == CV_32F);
  int rows = img.rows;
  int cols = img.cols;

  // Create channel with floating point type
  cudaChannelFormatDesc channelDesc =
  cudaCreateChannelDesc (32,
                          0,
                          0,
                          0,
                          cudaChannelFormatKindFloat);
  // Allocate array with correct size and number of channels
  checkCudaErrors(cudaMallocArray(&arr,
                                  &channelDesc,
                                  cols,
                                  rows));

  checkCudaErrors(cudaMemcpy2DToArray (arr,
                                        0,
                                        0,
                                        img.ptr<float>(),
                                        img.step[0],
                                        cols*sizeof(float),
                                        rows,
                                        cudaMemcpyHostToDevice));

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType         = cudaResourceTypeArray;
  resDesc.res.array.array = arr;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeWrap;
  texDesc.addressMode[1]   = cudaAddressModeWrap;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  checkCudaErrors(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

}

/**
* \brief
* \param argv 3
* \param argc argc[1]:��Ӱ��·�� argc[2]: ��Ӱ��·�� argc[3]: ��С�Ӳ�[��ѡ��Ĭ��0] argc[4]: ����Ӳ�[��ѡ��Ĭ��64]
* \param eg. ..\Data\cone\im2.png ..\Data\cone\im6.png 0 64
* \param eg. ..\Data\Reindeer\view1.png ..\Data\Reindeer\view5.png 0 128
* \return
*/

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

	cv::Mat img_left = cv::imread(path_left, cv::IMREAD_COLOR);
	cv::Mat img_right = cv::imread(path_right, cv::IMREAD_COLOR);

	if (img_left.data == nullptr || img_right.data == nullptr) {
		std::cout << "��ȡӰ��ʧ�ܣ�" << std::endl;
		return -1;
	}
	if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
		std::cout << "����Ӱ��ߴ粻һ�£�" << std::endl;
		return -1;
	}

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

  cv::imshow("img0", img_left);
  cv::imshow("img1", img_right);
  while(1) {
    if(cv::waitKey(0) == 'q') break;
  }


	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// ����Ӱ��Ĳ�ɫ����
	auto bytes_left = new unsigned char[width * height * 3];
	auto bytes_right = new unsigned char[width * height * 3];
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			bytes_left[i * 3 * width + 3 * j] = img_left.at<cv::Vec3b>(i, j)[0];
			bytes_left[i * 3 * width + 3 * j + 1] = img_left.at<cv::Vec3b>(i, j)[1];
			bytes_left[i * 3 * width + 3 * j + 2] = img_left.at<cv::Vec3b>(i, j)[2];
			bytes_right[i * 3 * width + 3 * j] = img_right.at<cv::Vec3b>(i, j)[0];
			bytes_right[i * 3 * width + 3 * j + 1] = img_right.at<cv::Vec3b>(i, j)[1];
			bytes_right[i * 3 * width + 3 * j + 2] = img_right.at<cv::Vec3b>(i, j)[2];
		}
	}
	printf("Done0!\n");

  unsigned char *bytes_left_device, *bytes_right_device;
  cudaMalloc((void **)&bytes_left_device, sizeof(unsigned char) * width * height * 3);
  cudaMalloc((void **)&bytes_right_device, sizeof(unsigned char) * width * height * 3);
  cudaMemcpy(bytes_left_device, bytes_left, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(bytes_right_device, bytes_right, sizeof(unsigned char) * width * height * 3, cudaMemcpyHostToDevice);

	// PMSƥ��������
	PatchMatchOptions *pms_options = new PatchMatchOptions;
	pms_options->patch_size = 35;
	pms_options->min_disparity = 0.0f;
	pms_options->max_disparity = 64.0f;
	pms_options->sigma_r = 10.0f;
	pms_options->alpha = 0.9f;
	pms_options->tau_col = 10.0f;
	pms_options->tau_grad = 2.0f;
	pms_options->num_iters = 3;
	pms_options->is_check_lr = true;
	pms_options->lrcheck_thres = 1.0f;
	pms_options->is_fill_holes = true;
	pms_options->is_fource_fpw = true;
	pms_options->is_integer_disp = true;

  CalibrationParams *pms_params = new CalibrationParams(P0, P1);

	// ����PMSƥ����ʵ��
	StereoMatcherCuda *matcher = new StereoMatcherCuda(width, height, pms_options, pms_params);
  
  auto disparity = new float[(unsigned int)(width * height)]();		
  matcher->Match(bytes_left_device, bytes_right_device, disparity);
	// // printf("PatchMatch Initializing...");
	// std::cout << "PatchMatch Initializing..." << std::endl;
	// auto start = std::chrono::steady_clock::now();
	// //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// // ��ʼ��
	// if (!pms.Initialize(width, height, pms_option)) {
	// 	std::cout << "PMS��ʼ��ʧ�ܣ�" << std::endl;
	// 	return -2;
	// }
	// auto end = std::chrono::steady_clock::now();
	// auto tt = duration_cast<std::chrono::milliseconds>(end - start);
	// // printf("Done! Timing : %lf s\n", tt.count() / 1000.0);

	// // printf("PatchMatch Matching...");
	// std::cout << "PatchMatch Matching..." << std::endl;
	// start = std::chrono::steady_clock::now();
	// //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// // ƥ��
	// // disparity���鱣�������ص��Ӳ���
	// auto disparity = new float[unsigned int(width * height)]();		
	// if (!pms.Match(bytes_left, bytes_right, disparity)) {
	// 	std::cout << "PMSƥ��ʧ�ܣ�" << std::endl;
	// 	return -2;
	// }
	// end = std::chrono::steady_clock::now();
	// tt = duration_cast<std::chrono::milliseconds>(end - start);
	// printf("Done! Timing : %lf s\n", tt.count() / 1000.0);

	// //��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// // ��ʾ�Ӳ�ͼ
	// ShowDisparityMap(pms.GetDisparityMap(0), width, height, "disp-left");
	// ShowDisparityMap(pms.GetDisparityMap(1), width, height, "disp-right");


	cv::waitKey(0);

	//��������������������������������������������������������������������������������������������������������������������������������������������������������������//
	// // �ͷ��ڴ�
	// delete[] disparity;
	// disparity = nullptr;
	// delete[] bytes_left;
	// bytes_left = nullptr;
	// delete[] bytes_right;
	// bytes_right = nullptr;

	// system("pause");
  delete matcher;
  delete pms_options;
  delete pms_params;

  cudaFree(bytes_left_device);
  cudaFree(bytes_right_device);
	return 0;
}

void ShowDisparityMap(const float* disp_map,const int& width,const int& height, const std::string& name)
{
	// ��ʾ�Ӳ�ͼ
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8U);
	float min_disp = float(width), max_disp = -float(width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const float disp = abs(disp_map[i * width + j]);
			if (disp != std::numeric_limits<float>::infinity()) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const float disp = abs(disp_map[i * width + j]);
			if (disp == std::numeric_limits<float>::infinity()) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}

	cv::imshow(name, disp_mat);
	cv::Mat disp_color;
	applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	cv::imshow(name + "-color", disp_color);
	while(1)
	{
		if(cv::waitKey(0) == 'q')
			break;
	}

}

void SaveDisparityMap(const float* disp_map, const int& width, const int& height, const std::string& path)
{
	// �����Ӳ�ͼ
	const cv::Mat disp_mat = cv::Mat(height, width, CV_8U);
	float min_disp = float(width), max_disp = -float(width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const float disp = abs(disp_map[i * width + j]);
			if (disp != std::numeric_limits<float>::infinity()) {
				min_disp = std::min(min_disp, disp);
				max_disp = std::max(max_disp, disp);
			}
		}
	}
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			const float disp = abs(disp_map[i * width + j]);
			if (disp == std::numeric_limits<float>::infinity()) {
				disp_mat.data[i * width + j] = 0;
			}
			else {
				disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
			}
		}
	}

	cv::imwrite(path + "-d.png", disp_mat);
	cv::Mat disp_color;
	applyColorMap(disp_mat, disp_color, cv::COLORMAP_JET);
	cv::imwrite(path + "-c.png", disp_color);
}

// void SaveDisparityCloud(const unsigned char* img_bytes, const float* disp_map, const int& width, const int& height, const std::string& path)
// {
// 	float B = 193.001;		// ����
// 	float f = 999.421;		// ����
// 	float x0l = 294.182;		// ����ͼ������x0
// 	float y0l = 252.932;		// ����ͼ������y0
// 	float x0r = 326.95975;	// ����ͼ������x0


// 	// �������
// 	FILE* fp_disp_cloud = nullptr;
// 	fopen_s(&fp_disp_cloud, (path + "-cloud.txt").c_str(), "w");
// 	if (fp_disp_cloud) {
// 		for (int y = 0; y < height; y++) {
// 			for (int x = 0; x < width; x++) {
// 				const float disp = abs(disp_map[y * width + x]);
// 				if (disp == Invalid_Float) {
// 					continue;
// 				}
// 				float Z = B * f / (disp + (x0r - x0l));
// 				float X = Z * (x - x0l) / f;
// 				float Y = Z * (y - y0l) / f;
// 				fprintf_s(fp_disp_cloud, "%f %f %f %d %d %d\n", X, Y,
// 					Z, img_bytes[y * width * 3 + 3 * x + 2], img_bytes[y * width * 3 + 3 * x + 1], img_bytes[y * width * 3 + 3 * x]);
// 			}
// 		}
// 		fclose(fp_disp_cloud);
// 	}
// }