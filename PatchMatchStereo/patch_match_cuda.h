#ifndef PATCH_MATCH_CUDA_H_
#define PATCH_MATCH_CUDA_H_

#include <vector>
#include <opencv2/core/core.hpp>
#include <cuda_runtime_api.h> 
#include <device_launch_parameters.h> 
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <builtin_types.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include <driver_functions.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include "global_malloc.h"
#include "math_utils.h"
#include "display_utils.h"
#include "data_io.h"
#include "data_types.h"
#include "patch_match_cost.h"

#include "cost_computer_host.hpp"

class StereoMatcherCuda : public GlobalMalloc
{
public:
	StereoMatcherCuda(const int& width, const int& height, PatchMatchOptions *options, CalibrationParams *params);
	~StereoMatcherCuda();

public:
	/**
	* \brief ��ĳ�ʼ�������һЩ�ڴ��Ԥ���䡢������Ԥ���õ�
	* \param width		���룬�������Ӱ���
	* \param height		���룬�������Ӱ���
	* \param option		���룬�㷨����
	*/
	bool Initialize(const unsigned int& width, const unsigned int& height, const PatchMatchOptions *options, const CalibrationParams *params);

	/**
	* \brief ִ��ƥ��
	* \param img_left	���룬��Ӱ������ָ�룬3ͨ��
	* \param img_right	���룬��Ӱ������ָ�룬3ͨ��
	* \param disp_left	�������Ӱ���Ӳ�ͼָ�룬Ԥ�ȷ����Ӱ��ȳߴ���ڴ�ռ�
	*/
	bool Match(const uint8_t* img_left, const uint8_t* img_right, float* disp_left);

  bool Match(const uint8_t* color_left, const uint8_t* color_right, cv::Mat &depth_img, cv::Mat &normal_img, bool texture);

	/**
	* \brief ����
	* \param width		���룬�������Ӱ���
	* \param height		���룬�������Ӱ���
	* \param option		���룬�㷨����
	*/
	bool Reset(const unsigned int& width, const unsigned int& height, PatchMatchOptions *options, CalibrationParams *params);


	/**
	 * \brief ��ȡ�Ӳ�ͼָ��
	 * \param view 0-����ͼ 1-����ͼ
	 * \return �Ӳ�ͼָ��
	 */
	float* GetDisparityMap(const int& view) const;


	/**
	 * \brief ��ȡ�ݶ�ͼָ��
	 * \param view 0-����ͼ 1-����ͼ
	 * \return �ݶ�ͼָ��
	 */
	Gradient* GetGradientMap(const int& view) const;
private:
	/** \brief �����ʼ�� */
	void RandomInitialization() const;

	/** \brief ����Ҷ����� */
	void ComputeGray() const;

	/** \brief �����ݶ����� */
	void ComputeGradient() const;

	/** \brief �������� */
	void Propagation() const;

	/** \brief һ���Լ��	 */
	void LRCheck();

	/** \brief �Ӳ�ͼ��� */
	void FillHolesInDispMap();

	/** \brief ƽ��ת�����Ӳ� */
	void PlaneToDisparity() const;

	/** \brief �ڴ��ͷ�	 */
	void Release();

public:
	/** \brief PMS����	 */
	PatchMatchOptions *options_;

  CalibrationParams *params_;

	/** \brief Ӱ���	 */ 
	int width_;

	/** \brief Ӱ���	 */
	int height_;

  cudaTextureObject_t tex_left_;
  cudaTextureObject_t tex_right_;
  cudaArray *array_left_, *array_right_;


	/** \brief ��Ӱ������	 */
	const uint8_t* img_left_;
	/** \brief ��Ӱ������	 */
	const uint8_t* img_right_;

	/** \brief ��Ӱ��Ҷ�����	 */
	uint8_t* gray_left_;
	/** \brief ��Ӱ��Ҷ�����	 */
	uint8_t* gray_right_;

	/** \brief ��Ӱ���ݶ�����	 */
	Gradient* grad_left_;
	/** \brief ��Ӱ���ݶ�����	 */
	Gradient* grad_right_;

	/** \brief ��Ӱ��ۺϴ�������	 */
	float* cost_left_;
	/** \brief ��Ӱ��ۺϴ�������	 */
	float* cost_right_;

	/** \brief ��Ӱ���Ӳ�ͼ	*/
	float* disp_left_;
	/** \brief ��Ӱ���Ӳ�ͼ	*/
	float* disp_right_;

	/** \brief ��Ӱ��ƽ�漯	*/
	DispPlane* plane_left_;
	/** \brief ��Ӱ��ƽ�漯	*/
	DispPlane* plane_right_;

  curandState* cs_left_;
  curandState* cs_right_;

	/** \brief �Ƿ��ʼ����־	*/
	bool is_initialized_;

	/** \brief ��ƥ�������ؼ�	*/
	std::vector<std::pair<int, int>> mismatches_left_;
	std::vector<std::pair<int, int>> mismatches_right_;

};


#endif