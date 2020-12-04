#ifndef COST_COMPUTER_HOST_HPP
#define COST_COMPUTER_HOST_HPP

#include "data_types.h"
#include "patch_match_cost.h"

class CostComputerHost {
public:
	/** \brief ���ۼ�����Ĭ�Ϲ��� */
	CostComputerHost(): img_left_(nullptr), img_right_(nullptr), width_(0), height_(0), patch_size_(0), min_disp_(0),
	                max_disp_(0) {}

	/**
	 * \brief ���ۼ�������ʼ��
	 * \param img_left		��Ӱ������ 
	 * \param img_right		��Ӱ������
	 * \param width			Ӱ���
	 * \param height		Ӱ���
	 * \param patch_size	�ֲ�Patch��С
	 * \param min_disp		��С�Ӳ�
	 * \param max_disp		����Ӳ�
	 */
	CostComputerHost(const unsigned char* img_left, const unsigned char* img_right, const int& width,const int& height,const int& patch_size,const int& min_disp, const int& max_disp){
		img_left_ = img_left;
		img_right_ = img_right;
		width_ = width;
		height_ = height;
		patch_size_ = patch_size;
		min_disp_ = min_disp;
		max_disp_ = max_disp;
	}

	/** \brief ���ۼ��������� */
	virtual ~CostComputerHost() = default;

public:

	/**
	 * \brief ������Ӱ��p���Ӳ�Ϊdʱ�Ĵ���ֵ
	 * \param i		p��������
	 * \param j		p�������
	 * \param d		�Ӳ�ֵ
	 * \return ����ֵ
	 */
	virtual float Compute(const int& i, const int& j, const float& d) = 0;

public:
	/** \brief ��Ӱ������ */
	const unsigned char* img_left_;
	/** \brief ��Ӱ������ */
	const unsigned char* img_right_;

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


/**
 * \brief ���ۼ�������PatchMatchSteroԭ�Ĵ��ۼ�����
 */
class CostComputerPMSHost : public CostComputerHost {
public:

	/** \brief PMS���ۼ�����Ĭ�Ϲ��� */
	CostComputerPMSHost(): grad_left_(nullptr), grad_right_(nullptr), gamma_(0), alpha_(0), tau_col_(0), tau_grad_(0) {} ;

	/**
	 * \brief PMS���ۼ��������ι���
	 * \param img_left		��Ӱ������
	 * \param img_right		��Ӱ������
	 * \param grad_left		���ݶ�����
	 * \param grad_right	���ݶ�����
	 * \param width			Ӱ���
	 * \param height		Ӱ���
	 * \param patch_size	�ֲ�Patch��С
	 * \param min_disp		��С�Ӳ�
	 * \param max_disp		����Ӳ�
	 * \param gamma			����gammaֵ
	 * \param alpha			����alphaֵ
	 * \param t_col			����tau_colֵ
	 * \param t_grad		����tau_gradֵ
	 */
	CostComputerPMSHost(const unsigned char* img_left, const unsigned char* img_right, const Gradient* grad_left, const Gradient* grad_right, const int& width, const int& height, const int& patch_size,
		const int& min_disp, const int& max_disp,
		const float& gamma, const float& alpha, const float& t_col, const float t_grad) :
		CostComputerHost(img_left, img_right, width, height, patch_size, min_disp, max_disp) {
		grad_left_ = grad_left;
		grad_right_ = grad_right;
		gamma_ = gamma;
		alpha_ = alpha;
		tau_col_ = t_col;
		tau_grad_ = t_grad;
	}

	/**
	 * \brief ������Ӱ��p���Ӳ�Ϊdʱ�Ĵ���ֵ
	 * \param x		p��x����
	 * \param y		p��y����
	 * \param d		�Ӳ�ֵ
	 * \return ����ֵ
	 */
	inline float Compute(const int& x, const int& y, const float& d) override
	{
		const float xr = x - d;
		if (xr < 0.0f || xr >= static_cast<float>(width_)) {
			return (1 - alpha_) * tau_col_ + alpha_ * tau_grad_;
		}
		// ��ɫ�ռ����
		const auto col_p = GetColor(img_left_, x, y);
		const auto col_q = GetColor(img_right_, xr, y);
		const auto dc = std::min(abs(col_p.b - col_q.x) + abs(col_p.g - col_q.y) + abs(col_p.r - col_q.z), tau_col_);

		// �ݶȿռ����
		const auto grad_p = GetGradient(grad_left_, x, y);
		const auto grad_q = GetGradient(grad_right_, xr, y);
		const auto dg = std::min(abs(grad_p.x - grad_q.x)+ abs(grad_p.y - grad_q.y), tau_grad_);

		// ����ֵ
		return (1 - alpha_) * dc + alpha_ * dg;
	}

	/**
	 * \brief ������Ӱ��p���Ӳ�Ϊdʱ�Ĵ���ֵ
	 * \param col_p		p����ɫֵ
	 * \param grad_p	p���ݶ�ֵ
	 * \param x			p��x����
	 * \param y			p��y����
	 * \param d			�Ӳ�ֵ
	 * \return ����ֵ
	 */
	inline float Compute(const Color& col_p,const Gradient& grad_p, const int& x, const int& y, const float& d) const
	{
		const float xr = x - d;
		if (xr < 0.0f || xr >= static_cast<float>(width_)) {
			return (1 - alpha_) * tau_col_ + alpha_ * tau_grad_;
		}
		// ��ɫ�ռ����
		const Vector3f col_q = GetColor(img_right_, xr, y);
		const float dc = std::min(fabs((float)col_p.b - (float)col_q.x) + fabs((float)col_p.g - col_q.y) + fabs((float)col_p.r - col_q.z), tau_col_);
    // printf("diff color: %f\n", (float)dc);

		// �ݶȿռ����
		const Vector2f grad_q = GetGradient(grad_right_, xr, y);
		const float dg = std::min(fabs((float)grad_p.x - (float)grad_q.x) + fabs((float)grad_p.y - (float)grad_q.y), tau_grad_);
    // printf("diff grad: %f\n", (float)dg);

    // printf("cost: %f\n", (1 - alpha_) * dc + alpha_ * dg);
		// ����ֵ
		return (1 - alpha_) * dc + alpha_ * dg;
	}


	/**
	 * \brief ������Ӱ��p���Ӳ�ƽ��Ϊpʱ�ľۺϴ���ֵ
	 * \param x		p��x����
	 * \param y 	p��y����
	 * \param p		ƽ�����
	 * \return �ۺϴ���ֵ
	 */
	inline float ComputeA(const int& x, const int& y, const DispPlane& p) const
	{
		const int pat = patch_size_ / 2;
		const Color& col_p = GetColor(img_left_, x, y);
		float cost = 0.0f;
    printf("Cost Host: \n");
		for (int r = -pat; r <= pat; r++) {
			const int yr = y + r;
			for (int c = -pat; c <= pat; c++) {
				const int xc = x + c;
        printf("x, y: %d, %d\n", xc, yr);
				if (yr < 0 || yr > height_ - 1 || xc < 0 || xc > width_ - 1) {
          printf("skip\n");
					continue;
				}
				// �����Ӳ�ֵ
				const float d = p.to_disparity(xc,yr);
				if (d < min_disp_ || d > max_disp_) {
					cost += COST_PUNISH;
          printf("out of range\n");
					continue;
				}

				// ����Ȩֵ
				const Color& col_q = GetColor(img_left_, xc, yr);
				const float dc = fabs((float)col_p.r - (float)col_q.r) + fabs((float)col_p.g - (float)col_q.g) + fabs((float)col_p.b - (float)col_q.b);
#ifdef USE_FAST_EXP
				const auto w = fast_exp(double(-dc / gamma_));
#else
				const float w = expf(-dc / gamma_);
#endif
        printf("col_p: %f %f %f\n", (float)col_p.r, (float)col_p.g, (float)col_p.b);
        printf("col_q: %f %f %f\n", (float)col_q.r, (float)col_q.g, (float)col_q.b);
        printf("dc: %f\n", (float)dc);
        printf("weight: %f\n", (float)w);
				// �ۺϴ���
				const Gradient grad_q = GetGradient(grad_left_, xc, yr);
				cost += w * Compute(col_q, grad_q, xc, yr, d);
        printf("Cost: %f\n", (float)Compute(col_q, grad_q, xc, yr, d));
			}
		}
    printf("Sum cost: %f\n", cost);
		return cost;
	}

	/**
	* \brief ��ȡ���ص����ɫֵ
	* \param img_data	��ɫ����,3ͨ��
	* \param x			����x����
	* \param y			����y����
	* \return ����(x,y)����ɫֵ
	*/
	inline Color GetColor(const unsigned char* img_data, const int& x, const int& y) const
	{
		auto* pixel = img_data + y * width_ * 3 + 3 * x;
		return { pixel[0], pixel[1], pixel[2] };
	}

	/**
	* \brief ��ȡ���ص����ɫֵ
	* \param img_data	��ɫ����
	* \param x			����x���꣬ʵ���������ڲ�õ���ɫֵ
	* \param y			����y����
	* \return ����(x,y)����ɫֵ
	*/
	inline Vector3f GetColor(const uint8_t* img_data, const float& x,const int& y) const
	{
		float col[3];
		const auto x1 = static_cast<int>(x);
		const int x2 = x1 + 1;
		const float ofs = x - x1;

		for (int n = 0; n < 3; n++) {
			const auto& g1 = img_data[y * width_ * 3 + 3 * x1 + n];
			const auto& g2 = (x2 < width_) ? img_data[y * width_ * 3 + 3 * x2 + n] : g1;
			col[n] = (1 - ofs) * g1 + ofs * g2;
		}

		return { col[0], col[1], col[2] };
	}

	/**
	* \brief ��ȡ���ص���ݶ�ֵ
	* \param grad_data	�ݶ�����
	* \param x			����x����
	* \param y			����y����
	* \return ����(x,y)���ݶ�ֵ
	*/
	inline Gradient GetGradient(const Gradient* grad_data, const int& x, const int& y) const
	{
		return grad_data[y * width_ + x];
	}

	/**
	* \brief ��ȡ���ص���ݶ�ֵ
	* \param grad_data	�ݶ�����
	* \param x			����x���꣬ʵ���������ڲ�õ��ݶ�ֵ
	* \param y			����y����
	* \return ����(x,y)���ݶ�ֵ
	*/
	inline Vector2f GetGradient(const Gradient* grad_data, const float& x, const int& y) const
	{
		const auto x1 = static_cast<int>(x);
		const int x2 = x1 + 1;
		const float ofs = x - x1;

		const auto& g1 = grad_data[y * width_ + x1];
		const auto& g2 = (x2 < width_) ? grad_data[y * width_ + x2] : g1;

		return { (1 - ofs) * g1.x + ofs * g2.x, (1 - ofs) * g1.y + ofs * g2.y };
	}

private:
	/** \brief ��Ӱ���ݶ����� */
	const Gradient* grad_left_;
	/** \brief ��Ӱ���ݶ����� */
	const Gradient* grad_right_;

	/** \brief ����gamma */
	float gamma_;
	/** \brief ����alpha */
	float alpha_;
	/** \brief ����tau_col */
	float tau_col_;
	/** \brief ����tau_grad */
	float tau_grad_;
};

#endif