/* -*-c++-*- PatchMatchStereo - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
*			  https://github.com/ethan-li-coding
* Describe	: header of pms_propagation
*/

#ifndef PATCH_MATCH_STEREO_PROPAGATION_H_
#define PATCH_MATCH_STEREO_PROPAGATION_H_
#include "pms_types.h"

#include "cost_computor.hpp"
#include <random>
#include "display_utils.h"
/**
 * \brief ������
 */
class PMSPropagation final{
public:
	PMSPropagation(const sint32 width, const sint32 height,
		const uint8* img_left, const uint8* img_right,
		const PGradient* grad_left, const PGradient* grad_right,
		DisparityPlane* plane_left, DisparityPlane* plane_right,
		const PMSOption& option,
		float32* cost_left, float32* cost_right,
		float32* disparity_map);

	~PMSPropagation();

public:
	/** \brief ִ�д���һ�� */
	void DoPropagation();

private:
	/** \brief ����������� */
	void ComputeCostData() const;

	/**
	 * \brief �ռ䴫��
	 * \param x ����x����
	 * \param y ����y����
	 * \param direction ��������
	 */
	void SpatialPropagation(const sint32& x, const sint32& y, const sint32& direction) const;
	
	/**
	 * \brief ��ͼ����
	 * \param x ����x����
	 * \param y ����y����
	 */
	void ViewPropagation(const sint32& x, const sint32& y) const;
	
	/**
	 * \brief ƽ���Ż�
	 * \param x ����x����
	 * \param y ����y����
	 */
	void PlaneRefine(const sint32& x, const sint32& y) const;
private:
	/** \brief ���ۼ����� */
	CostComputer* cost_cpt_left_;
	CostComputer* cost_cpt_right_;

	/** \brief PMS�㷨����*/
	PMSOption option_;

	/** \brief Ӱ����� */
	sint32 width_;  
	sint32 height_;

	/** \brief ������������ */
	sint32 num_iter_;

	/** \brief Ӱ������ */
	const uint8* img_left_;
	const uint8* img_right_;

	/** \brief �ݶ����� */
	const PGradient* grad_left_;
	const PGradient* grad_right_;

	/** \brief ƽ������ */
	DisparityPlane* plane_left_;
	DisparityPlane* plane_right_;

	/** \brief ��������	 */
	float32* cost_left_;
	float32* cost_right_;

	/** \brief �Ӳ����� */
	float32* disparity_map_;

	/** \brief ����������� */
	std::uniform_real_distribution<float32>* rand_disp_;
	std::uniform_real_distribution<float32>* rand_norm_;
};

#endif
