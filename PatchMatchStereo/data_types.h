#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "global_malloc.h"
#include "math_utils.h"
#include "texture_utils.h"
#include "device_buffer.h"
#include "global_buffer.h"

// typedef uint8_t  uint8_t;
// typedef          int16_t  int8_t;
// typedef uint16_t uint16_t;
// typedef          int16_t int16_t;
// typedef          float float32_t;

#define MAX_IMAGES_SIZE 64

class PMCostComputer : public GlobalMalloc
{
public:
  PMCostComputer(): img_left_(nullptr), img_right_(nullptr), width_(0), height_(0), patch_size_(0), min_disp_(0),
	                   max_disp_(0) {}

  PMCostComputer(const uint8_t* img_left, const uint8_t* img_right, 
                 const int& width, const int& height, const int& patch_size, 
                 const int& min_disp, const int& max_disp)
  : img_left_(img_left), img_right_(img_right), width_(width), height_(height), 
    patch_size_(patch_size), min_disp_(min_disp), max_disp_(max_disp) {}

  ~PMCostComputer();

private:
  /** \brief ��Ӱ������ */
	const uint8_t* img_left_;
	/** \brief ��Ӱ������ */
	const uint8_t* img_right_;

	/** \brief Ӱ��� */
	int width_;
	/** \brief Ӱ��� */
	int height_;
	/** \brief �ֲ�����Patch��С */
	int patch_size_;

	/** \brief ��С����Ӳ� */
	int min_disp_;
	int max_disp_;
};

/** \brief PMS�����ṹ�� */
struct PatchMatchOptions : public GlobalMalloc
{
	int	patch_size;			// patch�ߴ磬�ֲ�����Ϊ patch_size*patch_size
	float min_disparity;		// ��С�Ӳ�
	float	max_disparity;		// ����Ӳ�

	float	sigma_s;			//
  float sigma_r;       
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
    patch_size(35), min_disparity(0.0f), max_disparity(1.0f), sigma_s(3.0f), sigma_r(10.0f), alpha(0.9f), tau_col(10.0f),
    tau_grad(2.0f), num_iters(3), is_check_lr(false), lrcheck_thres(0),
    is_fill_holes(false), is_fource_fpw(false), is_integer_disp(false) {}

  void Print()
  {
    printf("Patch Match Stereo Matcher Options\n");
    printf("----------------------------------\n");
    printf("   patch_size: %d\n", patch_size);
    printf("min_disparity: %f\n", min_disparity);
    printf("max_disparity: %f\n", max_disparity);
    printf("      sigma_s: %f\n", sigma_s);
    printf("      sigma_r: %f\n", sigma_r);
    printf("        alpha: %f\n", alpha);
    printf("      tau_col: %f\n", tau_col);
    printf("     tau_grad: %f\n", tau_grad);
  }
};


class CalibrationParams : public GlobalMalloc
{
public:
  CalibrationParams(const cv::Mat_<float> &Pl, const cv::Mat_<float> &Pr) 
  {
    checkCudaErrors(cudaMallocManaged(&P0, sizeof(float) * 3 * 4));
    checkCudaErrors(cudaMallocManaged(&P1, sizeof(float) * 3 * 4));
    memcpy(P0, (float*)Pl.data, 12*sizeof(float));
    memcpy(P1, (float*)Pr.data, 12*sizeof(float));
    bf = norm3f(imgatf(Pl,0,3)-imgatf(Pr,0,3), imgatf(Pl,1,3)-imgatf(Pr,1,3), imgatf(Pl,2,3)-imgatf(Pr,2,3));
  }
  ~CalibrationParams() {
    cudaFree(P0);
    cudaFree(P1);
  }
  float *P0;
  float *P1;
  float bf;
};


/**
 * \brief ��ɫ�ṹ��
 */
struct Color : public GlobalMalloc 
{
	unsigned int r, g, b;
	Color() : r(0), g(0), b(0) {}
	Color(unsigned int _b, unsigned int _g, unsigned int _r) {
		r = _r; g = _g; b = _b;
	}
};
/**
 * \brief �ݶȽṹ��
 */
struct Gradient : public GlobalMalloc  
{
	int16_t x, y;
	Gradient() : x(0), y(0) {}
	Gradient(int16_t _x, int16_t _y) {
		x = _x; y = _y;
	}
};

/**
* \brief ��άʸ���ṹ��
*/
struct Vector2f : public GlobalMalloc  
{

	float x = 0.0f, y = 0.0f;

	Vector2f() = default;
	Vector2f(const float& _x, const float& _y) {
		x = _x; y = _y;
	}
	Vector2f(const int16_t& _x, const int16_t& _y) {
		x = float(_x); y = float(_y);
	}
	Vector2f(const Vector2f& v) {
		x = v.x; y = v.y;
	}

	// ������operators
	// operator +
	Vector2f operator+(const Vector2f& v) const {
		return Vector2f(x + v.x, y + v.y);
	}
	// operator -
	Vector2f operator-(const Vector2f& v) const {
		return Vector2f(x - v.x, y - v.y);
	}
	// operator -t
	Vector2f operator-() const {
		return Vector2f(-x, -y);
	}
	// operator =
	Vector2f& operator=(const Vector2f& v) {
		if (this == &v) {
			return *this;
		}
		else {
			x = v.x; y = v.y;
			return *this;
		}
	}
};

/**
* \brief ��άʸ���ṹ��
*/
struct Vector3f : public GlobalMalloc 
{

	float x = 0.0f, y = 0.0f, z = 0.0f;

	Vector3f() = default;
	Vector3f(const float& _x, const float& _y, const float& _z) {
		x = _x; y = _y; z = _z;
	}
	Vector3f(const unsigned int& _x, const unsigned int& _y, const unsigned int& _z) {
		x = float(_x); y = float(_y); z = float(_z);
	}
	Vector3f(const Vector3f& v) {
		x = v.x; y = v.y; z = v.z;
	}

	// normalize
	void normalize() {
		if (x == 0.0f && y == 0.0f && z == 0.0f) {
			return;
		}
		else {
			float sq = x * x + y * y + z * z;
			float sqf = sqrt(sq);
			x /= sqf; y /= sqf; z /= sqf;
		}
	}

	// ������operators
	// operator +
	Vector3f operator+(const Vector3f& v) const {
		return Vector3f(x + v.x, y + v.y, z + v.z);
	}
	// operator -
	Vector3f operator-(const Vector3f& v) const {
		return Vector3f(x - v.x, y - v.y, z - v.z);
	}
	// operator -t
	Vector3f operator-() const {
		return Vector3f(-x, -y, -z);
	}
	// operator =
	Vector3f& operator=(const Vector3f& v) {
		if (this == &v) {
			return *this;
		}
		else {
			x = v.x;
			y = v.y;
			z = v.z;
			return *this;
		}
	}
	// operator ==
	bool operator==(const Vector3f& v) const {
		return (x == v.x) && (y == v.y) && (z == v.z);
	}
	// operator !=
	bool operator!=(const Vector3f& v) const {
		return (x != v.x) || (y != v.y) || (z != v.z);
	}

	// dot
	float dot(const Vector3f& v) const {
		return x * v.x + y * v.y + z * v.z;
	}
};



/**
 * \brief �Ӳ�ƽ��
 */
class DispPlane : public GlobalMalloc 
{
public:
	float a, b, c;
	DispPlane() = default;
	DispPlane(const float& x,const float& y,const float& z) {
		a = x; b = y; c = z;
	}
	DispPlane(const int& x, const int& y, const Vector3f& n, const float& d) {
		a = -n.x / n.z;
		b = -n.y / n.z;
		c = (n.x * x + n.y * y + n.z * d) / n.z;
	}

	/**
	 * \brief ��ȡ��ƽ��������(x,y)���Ӳ�
	 * \param x		����x����
	 * \param y		����y����
	 * \return ����(x,y)���Ӳ�
	 */
	float to_disparity(const int& x,const int& y) const
	{
    return a * x + b * y + c;
	}

	/** \brief ��ȡƽ��ķ��� */
	Vector3f to_normal() const
	{
		Vector3f n(a, b, -1.0f);
		n.normalize();
		return n;
	}
	/**
	 * \brief ���Ӳ�ƽ��ת������һ��ͼ
	 * ��������ͼƽ�淽��Ϊ d = a_p*xl + b_p*yl + c_p
	 * ������ͼ���㣺(1) xr = xl - d_p; (2) yr = yl; (3) �Ӳ�����෴(���������Ӳ�Ϊ��ֵ�����Ӳ�Ϊ��ֵ)
	 * ��������ͼ�Ӳ�ƽ�淽�̾Ϳɵõ�����ͼ����ϵ�µ�ƽ�淽��: d = -a_p*xr - b_p*yr - (c_p+a_p*d_p)
	 * ������ͬ��
	 * \param x		����x����
	 * \param y 	����y����
	 * \return ת�����ƽ��
	 */
	DispPlane to_another_view(const int& x, const int& y) const
	{
		const float d = to_disparity(x, y);
		return { -a, -b, -c - a * d };
	}

	// operator ==
	bool operator==(const DispPlane& v) const {
		return (a == v.a && b == v.b && c == v.c);
	}
	// operator !=
	bool operator!=(const DispPlane& v) const {
		return (a != v.a || b != v.b || c != v.c);
	}
};


class InvDepthPlane : public GlobalMalloc 
{
public:
	float a, b, c;
	InvDepthPlane() = default;
	InvDepthPlane(const float& x,const float& y,const float& z) {
		a = x; b = y; c = z;
	}
	InvDepthPlane(const int& x, const int& y, const Vector3f& n, const float& d) {
		a = -n.x / n.z;
		b = -n.y / n.z;
		c = (n.x * x + n.y * y + n.z * d) / n.z;
	}

	/**
	 * \brief ��ȡ��ƽ��������(x,y)���Ӳ�
	 * \param x		����x����
	 * \param y		����y����
	 * \return ����(x,y)���Ӳ�
	 */
	float to_inverse_depth(const int& x, const int& y) const
	{
    return a * x + b * y + c;
	}

	/** \brief ��ȡƽ��ķ��� */
	Vector3f to_normal() const
	{
		Vector3f n(a, b, -1.0f);
		n.normalize();
		return n;
	}
	/**
	 * \brief ���Ӳ�ƽ��ת������һ��ͼ
	 * ��������ͼƽ�淽��Ϊ d = a_p*xl + b_p*yl + c_p
	 * ������ͼ���㣺(1) xr = xl - d_p; (2) yr = yl; (3) �Ӳ�����෴(���������Ӳ�Ϊ��ֵ�����Ӳ�Ϊ��ֵ)
	 * ��������ͼ�Ӳ�ƽ�淽�̾Ϳɵõ�����ͼ����ϵ�µ�ƽ�淽��: d = -a_p*xr - b_p*yr - (c_p+a_p*d_p)
	 * ������ͬ��
	 * \param x		����x����
	 * \param y 	����y����
	 * \return ת�����ƽ��
	 */
	InvDepthPlane to_another_view(const int& x, const int& y) const
	{
		const float d = to_inverse_depth(x, y);
		return { -a, -b, -c - a * d };
	}

	// operator ==
	bool operator==(const InvDepthPlane& v) const {
		return (a == v.a && b == v.b && c == v.c);
	}
	// operator !=
	bool operator!=(const InvDepthPlane& v) const {
		return (a != v.a || b != v.b || c != v.c);
	}
};

struct PlaneState : public GlobalMalloc
{
  float inv_depth_;
  float normal_x_;
  float normal_y_;
  float normal_z_;
  PlaneState(float inv_depth, float normal_x, float normal_y, float normal_z)
  : inv_depth_(inv_depth), normal_x_(normal_x), normal_y_(normal_y), normal_z_(normal_z) {}
};

class MultiViewGeometryParams : public GlobalMalloc
{
public:
  MultiViewGeometryParams(const cv::Mat_<float> &K, const cv::Mat_<float> &R, const cv::Mat_<float> &t)
  {
    checkCudaErrors(cudaMallocManaged(&P_, sizeof(float) * 3 * 4));
    checkCudaErrors(cudaMallocManaged(&K_, sizeof(float) * 3 * 3));
    checkCudaErrors(cudaMallocManaged(&R_, sizeof(float) * 3 * 3));
    checkCudaErrors(cudaMallocManaged(&t_, sizeof(float) * 3 * 1));
    cv::Mat_<float> T34(3, 4);
    cv::hconcat(R, t, T34);
    cv::Mat_<float> P = K * T34;
    cv::Mat_<float> Pinv = (P.t() * P).inv() * P.t();
    memcpy(P_, (float*)P.data, 12*sizeof(float));
    memcpy(K_, (float*)K.data, 9*sizeof(float));
    memcpy(R_, (float*)R.data, 9*sizeof(float));
    memcpy(t_, (float*)t.data, 3*sizeof(float));
    printf("project matrix:\n");
    for(int i = 0; i < 3; ++i) {
      for(int j = 0; j < 4; ++j) {
        std::cout << P_[i*4+j] << "  ";
      }
      std::cout << std::endl;
    }
    printf("intrinsic matrix:\n");
    for(int i = 0; i < 3; ++i) {
      for(int j = 0; j < 3; ++j) {
        std::cout << K_[i*3+j] << "  ";
      }
      std::cout << std::endl;
    }
  }
  ~MultiViewGeometryParams()
  {
    cudaFree(P_);
    cudaFree(K_);
    cudaFree(R_);
    cudaFree(t_);
  }
  
  float *P_;
  float *K_;
  float *R_;
  float *t_;
};

class ViewSpace : public GlobalMalloc
{
public:
  ViewSpace() : params_(nullptr) {}

  ViewSpace(const cv::Mat_<float> &K, const cv::Mat_<float> &R, const cv::Mat_<float> &t, int height, int width)
  {
    width_ = width;
    height_ = height;
    params_ = new MultiViewGeometryParams(K, R, t);
  }
  ~ViewSpace()
  {
    DestroyTextureObject(tex_, arr_);
    delete params_;
  }
  
  int height_, width_;
  MultiViewGeometryParams *params_;
  cudaTextureObject_t tex_;
  cudaArray *arr_;
  
};

class ReferenceViewSpace : public ViewSpace
{
public:
  ReferenceViewSpace(const cv::Mat_<float> &K, const cv::Mat_<float> &R, const cv::Mat_<float> &t, int height, int width) 
  {
    width_ = width;
    height_ = height;

    params_ = new MultiViewGeometryParams(K, R, t);
    checkCudaErrors(cudaMallocManaged((void **)&cost_, sizeof(float) * width_ * height_));
    checkCudaErrors(cudaMallocManaged((void **)&cs_, sizeof(curandState) * width_ * height_));
    checkCudaErrors(cudaMallocManaged((void **)&plane_, sizeof(PlaneState) * width_ * height_));
  }
  ~ReferenceViewSpace()
  {
    checkCudaErrors(cudaFree(cost_));
    checkCudaErrors(cudaFree(cs_));
    checkCudaErrors(cudaFree(plane_));
    DestroyTextureObject(tex_, arr_);
    delete params_;
  }

  float* cost_;
  curandState* cs_;
  PlaneState* plane_;
};

class MultiViewStereoMatcherCuda : public GlobalMalloc
{
public:
  MultiViewStereoMatcherCuda() : options_(nullptr) {}

  MultiViewStereoMatcherCuda(PatchMatchOptions *options)
  : options_(options) {}

  ~MultiViewStereoMatcherCuda() {
    delete ref_;
    for(int i = 0; i < image_size_; ++i) {
      delete src_[i];
    }
    // src_.destroy();
    // while(!src_.empty()) {
    //   delete src_.back();
    //   src_.pop_back();
    // }
  }

  void Match();


  ReferenceViewSpace *ref_;
  // std::vector<ViewSpace*> src_;
  ViewSpace *src_[MAX_IMAGES_SIZE];
  int image_size_;
  // GlobalBuffer<ViewSpace*> src_;

  PatchMatchOptions *options_;
  
};



#endif