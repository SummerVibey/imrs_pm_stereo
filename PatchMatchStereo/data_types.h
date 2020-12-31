#ifndef DATA_TYPES_H
#define DATA_TYPES_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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

#define CallSafeDelete(p) {if(p){delete (p);(p)=nullptr;}}
#define CallSafeDeleteBuffer(p,n) {for(int i=0;i<n;++i) {if(p[i]) delete (p[i]);(p[i])=nullptr;}}

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
	
	float min_disparity;		// ��С�Ӳ�
	float	max_disparity;		// ����Ӳ�

  int	patch_size;			// patch�ߴ磬�ֲ�����Ϊ patch_size*patch_size
	float	sigma_s;			//
  float sigma_c;       
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
    patch_size(35), min_disparity(0.0f), max_disparity(0.5f), sigma_s(3.0f), sigma_c(10.0f), alpha(0.9f), tau_col(10.0f),
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
    printf("      sigma_c: %f\n", sigma_c);
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
  float dp, nx, ny, nz;
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
  float idp_;
  float nx_;
  float ny_;
  float nz_;
  PlaneState() {}
  PlaneState(float idp, float nx, float ny, float nz)
  : idp_(idp), nx_(nx), ny_(ny), nz_(nz) {}
};

struct Plane : public GlobalMalloc
{
  float a_, b_, c_;
  
  Plane(float a, float b, float c): a_(a), b_(b), c_(c) {}

  float to_inverse_depth(const int& x, const int& y) const
	{
    return a_ * x + b_ * y + c_;
	}

	/** \brief ��ȡƽ��ķ��� */
	Vector3f to_normal() const
	{
		Vector3f n(a_, b_, -1.0f);
		n.normalize();
		return n;
	}
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
    memcpy(P_, (float*)P.data, 12*sizeof(float));
    memcpy(K_, (float*)K.data, 9*sizeof(float));
    memcpy(R_, (float*)R.data, 9*sizeof(float));
    memcpy(t_, (float*)t.data, 3*sizeof(float));
    Print();
  }
  MultiViewGeometryParams(const cv::Mat_<float> &P)
  {
    cv::Mat K, R, t4, t;
    cv::decomposeProjectionMatrix(P, K, R, t4);
    t = t4(cv::Range(0, 3), cv::Range(0, 1)) / t4.at<float>(3, 0);
    checkCudaErrors(cudaMallocManaged(&P_, sizeof(float) * 3 * 4));
    checkCudaErrors(cudaMallocManaged(&K_, sizeof(float) * 3 * 3));
    checkCudaErrors(cudaMallocManaged(&R_, sizeof(float) * 3 * 3));
    checkCudaErrors(cudaMallocManaged(&t_, sizeof(float) * 3 * 1));
    cv::Mat_<float> T34(3, 4);
    cv::hconcat(R, t, T34);
    memcpy(P_, (float*)P.data, 12*sizeof(float));
    memcpy(K_, (float*)K.data, 9*sizeof(float));
    memcpy(R_, (float*)R.data, 9*sizeof(float));
    memcpy(t_, (float*)t.data, 3*sizeof(float));
    // Print();
  }
  ~MultiViewGeometryParams()
  {
    cudaFree(P_);
    cudaFree(K_);
    cudaFree(R_);
    cudaFree(t_);
  }
  void Print()
  {
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
    printf("rotation matrix:\n");
    for(int i = 0; i < 3; ++i) {
      for(int j = 0; j < 3; ++j) {
        std::cout << R_[i*3+j] << "  ";
      }
      std::cout << std::endl;
    }
    printf("translation matrix:\n");
    for(int j = 0; j < 3; ++j) {
      std::cout << t_[j] << "  ";
    }
    std::cout << std::endl;
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
  ViewSpace(const cv::Mat_<float> &P, int height, int width)
  {
    width_ = width;
    height_ = height;
    params_ = new MultiViewGeometryParams(P);
  }
  ~ViewSpace()
  {
    DestroyTextureObject(tex_, arr_);
    CallSafeDelete(params_);
  }
  
  int height_, width_;
  MultiViewGeometryParams *params_;
  cudaTextureObject_t tex_;
  cudaArray *arr_;
  
};

class RefViewSpace : public ViewSpace
{
public:
  RefViewSpace(const cv::Mat_<float> &K, const cv::Mat_<float> &R, const cv::Mat_<float> &t, int height, int width) 
  {
    width_ = width;
    height_ = height;

    params_ = new MultiViewGeometryParams(K, R, t);
    checkCudaErrors(cudaMallocManaged((void **)&cost_, sizeof(float) * width_ * height_));
    checkCudaErrors(cudaMallocManaged((void **)&cs_, sizeof(curandState) * width_ * height_));
    checkCudaErrors(cudaMallocManaged((void **)&plane_, sizeof(PlaneState) * width_ * height_));
  }
  RefViewSpace(const cv::Mat_<float> &P, int height, int width) 
  {
    width_ = width;
    height_ = height;
    params_ = new MultiViewGeometryParams(P);
    checkCudaErrors(cudaMallocManaged((void **)&cost_, sizeof(float) * width_ * height_));
    checkCudaErrors(cudaMallocManaged((void **)&cs_, sizeof(curandState) * width_ * height_));
    checkCudaErrors(cudaMallocManaged((void **)&plane_, sizeof(PlaneState) * width_ * height_));
  }
  ~RefViewSpace()
  {
    DestroyTextureObject(tex_, arr_);
    checkCudaErrors(cudaFree(cs_));
    checkCudaErrors(cudaFree(plane_));
    checkCudaErrors(cudaFree(cost_));
    CallSafeDelete(params_);
  }

  float* cost_;
  curandState* cs_;
  PlaneState* plane_;
};

class MultiViewStereoMatcherCuda : public GlobalMalloc
{
public:
  MultiViewStereoMatcherCuda() : options_(nullptr), image_size_(0), ref_(nullptr) {}

  MultiViewStereoMatcherCuda(PatchMatchOptions *options, int image_size)
  : options_(options), image_size_(image_size), ref_(nullptr)
  {
  }

  void Reset(PatchMatchOptions *options, int image_size)
  {
    CallSafeDelete(ref_);
    CallSafeDeleteBuffer(src_, image_size_);
    options_ = options;
    image_size_ = image_size;
  }

  ~MultiViewStereoMatcherCuda() {
    CallSafeDelete(ref_);
    CallSafeDeleteBuffer(src_, image_size_);
  }

  void Match(cv::Mat &depth, cv::Mat &normal);

  PatchMatchOptions *options_;
  int image_size_;

  RefViewSpace *ref_;
  ViewSpace *src_[MAX_IMAGES_SIZE];
  

  
  
};



#endif