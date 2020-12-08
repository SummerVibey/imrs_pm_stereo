#include "patch_match_cuda.h"

__device__ __forceinline__ float CurandBetween(curandState *cs, float min, float max)
{
  return (curand_uniform(cs) * (max-min) + min);
}

__device__ __forceinline__ void Normalize3f(Vector3f *vec)
{
  const float inv_sqr2 = 1.0f / norm3f(vec->x, vec->y, vec->z);
  vec->x *= inv_sqr2;
  vec->y *= inv_sqr2;
  vec->z *= inv_sqr2;
}

__device__ __forceinline__ void ConstructPlane(int x, int y, float disp, float nx, float ny, float nz, DispPlane *plane)
{
  plane->a = -nx / nz;
  plane->b = -ny / nz;
  plane->c = (nx * x + ny * y + nz * disp) / nz;
}

__device__ __forceinline__ void PlaneToDisparity(const DispPlane *plane, const int& x, const int& y, float *disp)
{
  *disp = plane->a * x + plane->b * y + plane->c;
}

__device__ __forceinline__ void PlaneToNormal(const DispPlane *plane, Vector3f *norm)
{
  norm->x = plane->a;
  norm->y = plane->b;
  norm->z = -1.0f;
  Normalize3f(norm);
}



__device__ __forceinline__ void RandomPointPlane(
  DispPlane *local_plane, 
  curandState *local_cs, 
  int x, int y, 
  int height, int width, 
  float mind, float maxd, 
  float *P)
{
  float disp_rd = CurandBetween(local_cs, mind, maxd);

  float q1 = 1.0f;
  float q2 = 1.0f;
  float sum = 2.0f;
  while(sum >= 1.0f) {
    q1 = CurandBetween(local_cs, -1.0f, 1.0f);
    q2 = CurandBetween(local_cs, -1.0f, 1.0f);
    sum = q1 * q1 + q2 * q2;
  }
  const float sqr = sqrtf(1.0f - sum);
  Vector3f norm_rd;
  norm_rd.x = 2.0f * q1 * sqr;
  norm_rd.y = 2.0f * q2 * sqr;
  norm_rd.z = 1.0f - 2.0f * sum;

  Vector3f view_ray;
  const float& fx = P[0], &fy = P[5], &cx = P[2], &cy = P[6];

  view_ray.x = (x - cx) / fx;
  view_ray.y = (y - cy) / fy;
  view_ray.z = 1;

  float dp = norm_rd.x * view_ray.x + norm_rd.y * view_ray.y + norm_rd.z * view_ray.z;
  if(dp > 0) {
    norm_rd.x = -norm_rd.x;
    norm_rd.y = -norm_rd.y;
    norm_rd.z = -norm_rd.z;
  }
  ConstructPlane(x, y, disp_rd, norm_rd.x, norm_rd.y, norm_rd.z, local_plane);
}

// checked
__device__ __forceinline__ void GetGrayPoint(const uint8_t *color, uint8_t *gray, int x, int y, int height, int width)
{
  int center = y * width + x;

  const uint8_t b = color[center * 3];
  const uint8_t g = color[center * 3 + 1];
  const uint8_t r = color[center * 3 + 2];

  *gray = (uint8_t)(r * 0.299 + g * 0.587 + b * 0.114);
}

__device__ __forceinline__ void GetGrayFloatPoint(const uint8_t *color, float *gray, int x, int y, int height, int width)
{
  int center = y * width + x;

  const uint8_t b = color[center * 3];
  const uint8_t g = color[center * 3 + 1];
  const uint8_t r = color[center * 3 + 2];

  *gray = (float)(r * 0.299f + g * 0.587f + b * 0.114f);
}

__device__ __forceinline__ void GetGradientPoint(const uint8_t *gray, Gradient *grad, int x, int y, int height, int width)
{
  const auto grad_x = (-gray[(y - 1) * width + x - 1] + gray[(y - 1) * width + x + 1]) +
                          (-2 * gray[y * width + x - 1] + 2 * gray[y * width + x + 1]) +
                          (-gray[(y + 1) * width + x - 1] + gray[(y + 1) * width + x + 1]);
  const auto grad_y = (-gray[(y - 1) * width + x - 1] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + x + 1]) +
					                (gray[(y + 1) * width + x - 1] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + x + 1]);
  grad[y * width + x].x = grad_x / 8;
  grad[y * width + x].y = grad_y / 8;
}


__device__ __forceinline__ void GetColorCuda(const uint8_t* img_data, Vector3f *color_pt, const float& x, const int& y, int width)
{
  float col[3];
  const int x1 = static_cast<int>(x);
  const int x2 = x1 + 1;
  const float ofs = x - x1;

  for (int n = 0; n < 3; n++) {
    const uint8_t& g1 = img_data[y * width * 3 + 3 * x1 + n];
    const uint8_t& g2 = (x2 < width) ? img_data[y * width * 3 + 3 * x2 + n] : g1;
    col[n] = (1 - ofs) * (float)g1 + ofs * (float)g2;
  }
  color_pt->x = col[0];
  color_pt->y = col[1];
  color_pt->z = col[2];
}


__device__ __forceinline__ void GetGradientCuda(const Gradient* grad_data, Vector2f *grad_pt, const float& x, const int& y, int width) 
{
  const int x1 = static_cast<int>(x);
  const int x2 = x1 + 1;
  const float ofs = x - x1;

  const Gradient& g1 = grad_data[y * width + x1];
  const Gradient& g2 = (x2 < width) ? grad_data[y * width + x2] : g1;

  grad_pt->x = (1 - ofs) * (float)g1.x + ofs * (float)g2.x;
  grad_pt->y = (1 - ofs) * (float)g1.y + ofs * (float)g2.y;

}

__device__ __forceinline__ float ComputeCostPointCuda(
  const uint8_t *img_left, const uint8_t *img_right, const Gradient *grad_left, const Gradient *grad_right,
  const int& x, const int& y, const float& d, const PatchMatchOptions *options, const int& width)
{
  const float xr = x - d;
  // if (xr < 0.0f || xr >= static_cast<float>(width)) {
  //   return (1 - options->alpha) * options->tau_col + options->alpha * options->tau_grad;
  // }
  // ��ɫ�ռ����
  Vector3f col_p, col_q;
  GetColorCuda(img_left, &col_p, x, y, width);
  GetColorCuda(img_right, &col_q, xr, y, width);
  const float dc = fminf(fabs(col_p.x - col_q.x) + fabs(col_p.y - col_q.y) + fabs(col_p.z - col_q.z), options->tau_col);
  // const float dc = fminf(fabs((col_p.x * 0.299f + col_p.y * 0.587f + col_p.z * 0.114f) - (col_q.x * 0.299f + col_q.y * 0.587f + col_q.z * 0.114f)), options->tau_col);
  // �ݶȿռ����
  Vector2f grad_p, grad_q;
  GetGradientCuda(grad_left, &grad_p, x, y, width);
  GetGradientCuda(grad_right, &grad_q, xr, y, width);
  const float dg = fminf(fabs(grad_p.x - grad_q.x)+ fabs(grad_p.y - grad_q.y), options->tau_grad);

  return (1 - options->alpha) * dc + options->alpha * dg;
}

__device__ __forceinline__ float ComputeCostPointTexture(
  cudaTextureObject_t img_left, cudaTextureObject_t img_right,
  const int& x, const int& y, const float& d, const PatchMatchOptions *options, const int& width)
{
  const float xr = x - d;
  if (xr < 0.0f || xr >= static_cast<float>(width)) {
    return (1 - options->alpha) * options->tau_col + options->alpha * options->tau_grad;
  }
  // ��ɫ�ռ����
  float col_p = color(img_left, x, y);
  float col_q = color(img_right, xr, y);

  const float dc = fminf(fabs(col_p - col_q), options->tau_col);

  // �ݶȿռ����
  float grad_px = gradx(img_left, x, y);
  float grad_qx = gradx(img_right, xr, y);
  float grad_py = grady(img_left, x, y);
  float grad_qy = grady(img_right, xr, y);

  const float dg = fminf(fabs(grad_px - grad_qx)+ fabs(grad_py - grad_qy), options->tau_grad);

  return (1 - options->alpha) * dc + options->alpha * dg;
}

__device__ __forceinline__ float ComputeCostRegionCuda(
  const uint8_t *img_left, const uint8_t *img_right, const Gradient *grad_left, const Gradient *grad_right,
  const int& px, const int& py, const DispPlane *p, const PatchMatchOptions *options, int height, int width) 
{
  const int pat = options->patch_size / 2;

  Vector3f col_p;
  GetColorCuda(img_left, &col_p, px, py, width);

  float cost = 0.0f;
  for (int r = -pat; r <= pat; r+=1) {
    const int qy = py + r;
    for (int c = -pat; c <= pat; c+=1) {
      const int qx = px + c;
      if (qy < 0 || qy > height - 1 || qx < 0 || qx > width - 1) {
        continue;
      }
      // �����Ӳ�ֵ
      float dq;
      PlaneToDisparity(p, qx, qy, &dq);
      if (dq < options->min_disparity || dq > options->max_disparity) {
        cost += COST_PUNISH;
        continue;
      }

      // ����Ȩֵ
      Vector3f col_q;
      GetColorCuda(img_left, &col_q, qx, qy, width);

      const float dc = (fabs(col_p.x - col_q.x) + fabs(col_p.y - col_q.y) + fabs(col_p.z - col_q.z)) / 3.0f;
      const float ds = abs(px - qx) + abs(py - qy);
      // printf("dc=%f\n", dc);
      // const float w = expf(-dc / options->gamma);
      const float w = expf(-ds / options->sigma_s - dc / options->sigma_c);
      
      // const float dc = fabs((col_p.x * 0.299f + col_p.y * 0.587f + col_p.z * 0.114f) - (col_q.x * 0.299f + col_q.y * 0.587f + col_q.z * 0.114f));
      // const float w = expf(-(abs(px-qx) + abs(py-qy)) / options->gamma) * expf(-dc / options->gamma);
      // const float dc = fabs(col_p.z - col_q.z) + fabs(col_p.y - col_q.y) + fabs(col_p.x - col_q.x);
      // const float w = expf(-dc / options->sigma_c);
      // �ۺϴ���
      // Vector2f grad_q;
      // GetGradientCuda(grad_left, &grad_q, qx, yr, width);
      cost += w * ComputeCostPointCuda(img_left, img_right, grad_left, grad_right, qx, qy, dq, options, width);
    }
  }
  return cost;
}

__device__ __forceinline__ float ComputePMCostRegion(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right,
  const int& px, const int& py, 
  const DispPlane *p, 
  const PatchMatchOptions *options, 
  int height, int width) 
{
  const int pat = options->patch_size / 2;

  float col_p = color(img_left, px, py);

  float cost = 0.0f;
  for (int r = -pat; r <= pat; r+=1) {
    const int qy = py + r;
    for (int c = -pat; c <= pat; c+=1) {
      const int qx = px + c;
      // printf("x, y: %d, %d\n", qx, qy);
      // if (qy < 0 || qy > height - 1 || qx < 0 || qx > width - 1) {
      //   // printf("skip\n");
      //   continue;
      // }
      // �����Ӳ�ֵ
      float dq;
      PlaneToDisparity(p, qx, qy, &dq);
      if (dq < options->min_disparity || dq > options->max_disparity) {
        cost += COST_PUNISH;
        // printf("out of range\n");
        continue;
      }

      // ����Ȩֵ
      float col_q = color(img_left, qx, qy);;
      const float dc = fabs(col_p - col_q);
      const float ds = abs(px - qx) + abs(py - qy);
      // printf("dc=%f\n", dc);
      // const float w = expf(-dc / options->gamma);
      const float w = expf(-ds / options->sigma_s - dc / options->sigma_c);
      // printf("weight=%f\n", w);
      // �ۺϴ���
      // Vector2f grad_q;
      // GetGradientCuda(grad_left, &grad_q, qx, yr, width);

      cost += w * ComputeCostPointTexture(img_left, img_right, qx, qy, dq, options, width);
      // printf("cost=%f\n", ComputeCostPointTexture(img_left, img_right, qx, qy, dq, options, width));
      // printf("\n");
    }
  }
  // printf("sum cost: %f\n", cost);
  return cost;
}

__global__ void RandomInit(curandState *cs, unsigned long long seed, int height, int width)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 
  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;

  int center = y * width + x;
  curand_init(seed, y, x, &cs[center]);
}

__global__ void RandomPlane(
  DispPlane *plane, 
  curandState *cs, 
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, int width,
  bool is_right)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1)  return;

  int center = y * width + x;
  float *project_mat = (is_right) ? params->P0 : params->P1;
  RandomPointPlane(&plane[center], &cs[center], x, y, height, width, options->min_disparity, options->max_disparity, project_mat);
}

// checked
__global__ void GetGrayImage(const uint8_t *color, uint8_t *gray, int height, int width)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  int center = x + y * width;
  GetGrayPoint(color, &gray[center], x, y, height, width);
}

__global__ void GetGrayImageFloat(const uint8_t *color, float *gray, int height, int width)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  int center = x + y * width;
  GetGrayFloatPoint(color, &gray[center], x, y, height, width);
}

__global__ void GetGradientImage(const uint8_t *gray, Gradient *grad, int height, int width)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  if(x < 1 || x > width - 2 || y < 1 || y > height - 2) return;
  GetGradientPoint(gray, grad, x, y, height, width);

}

__global__ void GetInitialCostTexture(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right, 
  const DispPlane *plane, 
  const PatchMatchOptions *options,
  float *cost, 
  int height, 
  int width)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  int center = x + y * width;
  cost[center] = ComputePMCostRegion(img_left, img_right, x, y, &plane[center], options, height, width);
}

__global__ void GetInitialCostTexturePoint(
  const cudaTextureObject_t img_left, 
  const cudaTextureObject_t img_right, 
  const DispPlane *plane, 
  const PatchMatchOptions *options,
  int x, int y,
  float *cost, 
  int height, 
  int width)
{
  int center = x + y * width;
  cost[center] = ComputePMCostRegion(img_left, img_right, x, y, &plane[center], options, height, width);
}

__global__ void GetInitialCost(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  const DispPlane *plane, 
  const PatchMatchOptions *options,
  float *cost, 
  int height, 
  int width)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  int center = x + y * width;
  cost[center] = ComputeCostRegionCuda(color_left, color_right, grad_left, grad_right, x, y, &plane[center], options, height, width);
}


void GetInitialCostHost(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  const DispPlane *plane, 
  const PatchMatchOptions *options,
  float *cost, 
  int height, 
  int width)
{
  CostComputerPMSHost* cost_computer = new CostComputerPMSHost(color_left, color_right, grad_left, grad_right, width, height,
    options->patch_size, options->min_disparity, options->max_disparity, options->sigma_c,
    options->alpha, options->tau_col, options->tau_grad);

  for(int y = 0; y < height; ++y) {
    for(int x = 0; x < width; ++x) {
      const DispPlane& plane_p = plane[y * width + x];
      cost[y * width + x] = cost_computer->ComputeA(x, y, plane_p);
    }
  }
  // const DispPlane& plane_p = plane[100 * width + 100];
  // cost[100 * width + 100] = cost_computer->ComputeA(100, 100, plane_p);

  delete cost_computer;

}

void GetInitialCostHostPoint(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  const DispPlane *plane, 
  const PatchMatchOptions *options,
  int x, int y,
  float *cost, 
  int height, 
  int width)
{
  CostComputerPMSHost* cost_computer = new CostComputerPMSHost(color_left, color_right, grad_left, grad_right, width, height,
    options->patch_size, options->min_disparity, options->max_disparity, options->sigma_c,
    options->alpha, options->tau_col, options->tau_grad);


  const DispPlane& plane_p = plane[y * width + x];
  cost[y * width + x] = cost_computer->ComputeA(x, y, plane_p);

  // const DispPlane& plane_p = plane[100 * width + 100];
  // cost[100 * width + 100] = cost_computer->ComputeA(100, 100, plane_p);

  delete cost_computer;

}

__device__ __forceinline__ void SpatialPropagation(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  DispPlane *plane, 
  float *cost, 
  int x, int y, int x_bias, int y_bias,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  const int x_nb = x + x_bias;
  const int y_nb = y + y_bias;

  if(x_nb < 0 || x_nb > width - 1 || y_nb < 0 || y_nb > height - 1) return;

  // printf("Check: (%d, %d)\n", x_nb, y_nb);
  // printf("Original: (%d, %d)\n", x, y);
  const int center_local = x + y * width;
  const int center_nb = x_nb + y_nb * width;
  DispPlane &plane_local = plane[center_local];
  float &cost_local = cost[center_local];

  DispPlane &plane_nb = plane[center_nb];
  // printf("Current State: %f, %f, %f, Current Cost: %f\n", plane_local.a, plane_local.b, plane_local.c, cost_local);
  float cost_nb = ComputeCostRegionCuda(color_left, color_right, grad_left, grad_right, x, y, &plane_nb, options, height, width);
  if(cost_nb < cost_local) {
    cost_local = cost_nb;
    plane_local.a = plane_nb.a;
    plane_local.b = plane_nb.b;
    plane_local.c = plane_nb.c;
    // printf("Update, New Cost: %f\n", cost_nb);
  }
  // printf("\n");
}

__device__ __forceinline__ void SpatialPropagationTexture(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right, 
  DispPlane *plane, 
  float *cost, 
  int x, int y, int x_bias, int y_bias,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  const int x_nb = x + x_bias;
  const int y_nb = y + y_bias;

  if(x_nb < 0 || x_nb > width - 1 || y_nb < 0 || y_nb > height - 1) return;

  // printf("Check: (%d, %d)\n", x_nb, y_nb);
  // printf("Original: (%d, %d)\n", x, y);
  const int center_local = x + y * width;
  const int center_nb = x_nb + y_nb * width;
  DispPlane &plane_local = plane[center_local];
  float &cost_local = cost[center_local];

  DispPlane &plane_nb = plane[center_nb];
  // printf("Current State: %f, %f, %f, Current Cost: %f\n", plane_local.a, plane_local.b, plane_local.c, cost_local);
  float cost_nb = ComputePMCostRegion(img_left, img_right, x, y, &plane_nb, options, height, width);
  // printf("Neighbor State: %f, %f, %f, Neighbor Cost: %f\n", plane_nb.a, plane_nb.b, plane_nb.c, cost_nb);
  if(cost_nb < cost_local) {
    cost_local = cost_nb;
    plane_local.a = plane_nb.a;
    plane_local.b = plane_nb.b;
    plane_local.c = plane_nb.c;
    // printf("Update, New Cost: %f\n", cost_nb);
  }
  // printf("\n");
}

__device__ __forceinline__ void PlaneRefinement(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  int x, int y,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  const float max_disp = static_cast<float>(options->max_disparity);
	const float min_disp = static_cast<float>(options->min_disparity);


  // ����p��ƽ�桢���ۡ��Ӳ����
  int center_p = y * width + x;
	DispPlane& plane_p = plane[center_p];
	float& cost_p = cost[center_p];

  float d_p;
  Vector3f norm_p;
  PlaneToDisparity(&plane_p, x, y, &d_p);
  PlaneToNormal(&plane_p, &norm_p);


	float disp_update = (max_disp - min_disp) / 2.0f;
	float norm_update = 1.0f;
	const float stop_thres = 0.01f;

	// �����Ż�
  int trial_num = 0;
	while (disp_update > stop_thres) {
    // printf("Try number: %d, disp_update: %f, normal_update: %f\n", trial_num++, disp_update, norm_update);

		float disp_rd = CurandBetween(&cs[center_p], -1.f, 1.f) * disp_update;

		const float d_p_new = d_p + disp_rd;
		if (d_p_new < min_disp || d_p_new > max_disp) {
			disp_update /= 6.0f;
      norm_update /= 4.0f;
			continue;
		}

		// �� -norm_update ~ norm_update ��Χ���������ֵ��Ϊ������������������
		Vector3f norm_rd;

    norm_rd.x = CurandBetween(&cs[center_p], -1.f, 1.f) * norm_update;
    norm_rd.y = CurandBetween(&cs[center_p], -1.f, 1.f) * norm_update;
    float z = CurandBetween(&cs[center_p], -1.f, 1.f) * norm_update;
    while (z == 0.0f) {
      z = CurandBetween(&cs[center_p], -1.f, 1.f) * norm_update;
    }
    norm_rd.z = z;


		// ��������p�µķ���
    Vector3f norm_p_new;
    norm_p_new.x = norm_p.x + norm_rd.x;
    norm_p_new.y = norm_p.y + norm_rd.y;
    norm_p_new.z = norm_p.z + norm_rd.z;

    Normalize3f(&norm_p_new);

    Vector3f view_ray;
    const float& fx = params->P0[0], &fy = params->P0[5], &cx = params->P0[2], &cy = params->P0[6];

    view_ray.x = (x - cx) / fx;
    view_ray.y = (y - cy) / fy;
    view_ray.z = 1;

    float dp = norm_p_new.x * view_ray.x + norm_p_new.y * view_ray.y + norm_p_new.z * view_ray.z;
    if(dp > 0) {
      norm_p_new.x = -norm_p_new.x;
      norm_p_new.y = -norm_p_new.y;
      norm_p_new.z = -norm_p_new.z;
    }

    DispPlane plane_new;
    ConstructPlane(x, y, d_p_new, norm_p_new.x, norm_p_new.y, norm_p_new.z, &plane_new);
    // printf("Current State: %f, %f, %f, Current Cost: %f\n", plane_p.a, plane_p.b, plane_p.c, cost_p);
    float cost_new = ComputeCostRegionCuda(color_left, color_right, grad_left, grad_right, x, y, &plane_new, options, height, width);
    if(cost_new < cost_p) {
      cost_p = cost_new;
      plane_p.a = plane_new.a;
      plane_p.b = plane_new.b;
      plane_p.c = plane_new.c;
      // printf("Update, New Cost: %f\n", cost_new);
    }
		disp_update /= 6.0f;
		norm_update /= 4.0f;
	}
}

__device__ __forceinline__ void PlaneRefinementTexture(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  int x, int y,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  const float max_disp = static_cast<float>(options->max_disparity);
	const float min_disp = static_cast<float>(options->min_disparity);


  // ����p��ƽ�桢���ۡ��Ӳ����
  int center_p = y * width + x;
	DispPlane& plane_p = plane[center_p];
	float& cost_p = cost[center_p];

  float d_p;
  Vector3f norm_p;
  PlaneToDisparity(&plane_p, x, y, &d_p);
  PlaneToNormal(&plane_p, &norm_p);


	float disp_update = (max_disp - min_disp) / 2.0f;
	float norm_update = 1.0f;
	const float stop_thres = 0.01f;

	// �����Ż�
  int trial_num = 0;
	while (disp_update > stop_thres) {
    // printf("Try number: %d, disp_update: %f, normal_update: %f\n", trial_num++, disp_update, norm_update);

		float disp_rd = CurandBetween(&cs[center_p], -1.f, 1.f) * disp_update;

		const float d_p_new = d_p + disp_rd;
		if (d_p_new < min_disp || d_p_new > max_disp) {
			disp_update /= 2.0f;
      norm_update /= 2.0f;
			continue;
		}

		// �� -norm_update ~ norm_update ��Χ���������ֵ��Ϊ������������������
		Vector3f norm_rd;

    norm_rd.x = CurandBetween(&cs[center_p], -1.f, 1.f) * norm_update;
    norm_rd.y = CurandBetween(&cs[center_p], -1.f, 1.f) * norm_update;
    float z = CurandBetween(&cs[center_p], -1.f, 1.f) * norm_update;
    while (z == 0.0f) {
      z = CurandBetween(&cs[center_p], -1.f, 1.f) * norm_update;
    }
    norm_rd.z = z;


		// ��������p�µķ���
    Vector3f norm_p_new;
    norm_p_new.x = norm_p.x + norm_rd.x;
    norm_p_new.y = norm_p.y + norm_rd.y;
    norm_p_new.z = norm_p.z + norm_rd.z;

    Normalize3f(&norm_p_new);

    Vector3f view_ray;
    const float& fx = params->P0[0], &fy = params->P0[5], &cx = params->P0[2], &cy = params->P0[6];

    view_ray.x = (x - cx) / fx;
    view_ray.y = (y - cy) / fy;
    view_ray.z = 1;

    float dp = norm_p_new.x * view_ray.x + norm_p_new.y * view_ray.y + norm_p_new.z * view_ray.z;
    if(dp > 0) {
      norm_p_new.x = -norm_p_new.x;
      norm_p_new.y = -norm_p_new.y;
      norm_p_new.z = -norm_p_new.z;
    }

    DispPlane plane_new;
    ConstructPlane(x, y, d_p_new, norm_p_new.x, norm_p_new.y, norm_p_new.z, &plane_new);
    // printf("Current State: %f, %f, %f, Current Cost: %f\n", plane_p.a, plane_p.b, plane_p.c, cost_p);
    float cost_new = ComputePMCostRegion(img_left, img_right, x, y, &plane_new, options, height, width);
    // printf("New State: %f, %f, %f, New Cost: %f\n", plane_new.a, plane_new.b, plane_new.c, cost_new);
    if(cost_new < cost_p) {
      cost_p = cost_new;
      plane_p.a = plane_new.a;
      plane_p.b = plane_new.b;
      plane_p.c = plane_new.c;
      // printf("Update, New Cost: %f\n", cost_new);
    }
		disp_update /= 2.0f;
		norm_update /= 2.0f;
	}
}

__device__ __forceinline__ void Propagate(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  int x, int y,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  // const int x_bias[20] = {-1,0,1,0,
  //   -3,-2,-1,0,1,2,3,2,1,0,-1,-2,
  //   -5,0,5,0};
  // const int y_bias[20] = {0,1,0,-1,
  //     0,1,2,3,2,1,0,-1,-2,-3,-2,-1,
  //     0,5,0,-5};

  // for(int i = 0; i < 20; ++i) {
  //   SpatialPropagation(color_left, color_right, grad_left, grad_right, plane, cost, x, y, x_bias[i], y_bias[i], options, params, height, width);
  // }

  const int x_bias[8] = {-1,0,1,0,-5,0,5,0};
  const int y_bias[8] = {0,1,0,-1,0,5,0,-5};

  for(int i = 0; i < 8; ++i) {
    SpatialPropagation(color_left, color_right, grad_left, grad_right, plane, cost, x, y, x_bias[i], y_bias[i], options, params, height, width);
  }

  PlaneRefinement(color_left, color_right, grad_left, grad_right, plane, cost, cs, x, y, options, params, height, width);
}

__device__ __forceinline__ void PropagateTexture(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  int x, int y,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  // const int x_bias[20] = {-1,0,1,0,
  //   -3,-2,-1,0,1,2,3,2,1,0,-1,-2,
  //   -5,0,5,0};
  // const int y_bias[20] = {0,1,0,-1,
  //     0,1,2,3,2,1,0,-1,-2,-3,-2,-1,
  //     0,5,0,-5};

  // for(int i = 0; i < 20; ++i) {
  //   SpatialPropagation(color_left, color_right, grad_left, grad_right, plane, cost, x, y, x_bias[i], y_bias[i], options, params, height, width);
  // }

  const int x_bias[8] = {-1,0,1,0,-5,0,5,0};
  const int y_bias[8] = {0,1,0,-1,0,5,0,-5};

  for(int i = 0; i < 8; ++i) {
    SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, x_bias[i], y_bias[i], options, params, height, width);
  }

  PlaneRefinementTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
}

__device__ __forceinline__ void PropagateClose(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  int x, int y,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  const int x_bias[4] = {-1,0,1,0};
  const int y_bias[4] = {0,1,0,-1};

  for(int i = 0; i < 4; ++i) {
    SpatialPropagation(color_left, color_right, grad_left, grad_right, plane, cost, x, y, x_bias[i], y_bias[i], options, params, height, width);
  }

  PlaneRefinement(color_left, color_right, grad_left, grad_right, plane, cost, cs, x, y, options, params, height, width);
}

__device__ __forceinline__ void PropagateCloseTexture(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  int x, int y,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  const int x_bias[4] = {-1,0,1,0};
  const int y_bias[4] = {0,1,0,-1};

  for(int i = 0; i < 4; ++i) {
    // printf("spatial propagate direct: %d, %d\n", x_bias[i], y_bias[i]);
    SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, x_bias[i], y_bias[i], options, params, height, width);
  }

  PlaneRefinementTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
}

__device__ __forceinline__ void PropagateFar(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  int x, int y,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  const int x_bias[4] = {-5,0,5,0};
  const int y_bias[4] = {0,5,0,-5};

  for(int i = 0; i < 4; ++i) {
    SpatialPropagation(color_left, color_right, grad_left, grad_right, plane, cost, x, y, x_bias[i], y_bias[i], options, params, height, width);
  }
  PlaneRefinement(color_left, color_right, grad_left, grad_right, plane, cost, cs, x, y, options, params, height, width);
}

__device__ __forceinline__ void PropagateFarTexture(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  int x, int y,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  const int x_bias[4] = {-5,0,5,0};
  const int y_bias[4] = {0,5,0,-5};

  for(int i = 0; i < 4; ++i) {
    // printf("spatial propagate direct: %d, %d\n", x_bias[i], y_bias[i]);
    SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, x_bias[i], y_bias[i], options, params, height, width);
  }

  PlaneRefinementTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
}

__global__ void PropagateRedTexture(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (threadIdx.x % 2 == 0) y = y * 2 + 1;
  else y = y * 2;
  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;

  PropagateCloseTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
  PropagateFarTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
}

__global__ void PropagateRed(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (threadIdx.x % 2 == 0) y = y * 2 + 1;
  else y = y * 2;

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  PropagateClose(color_left, color_right, grad_left, grad_right, plane, cost, cs, x, y, options, params, height, width);
  PropagateFar(color_left, color_right, grad_left, grad_right, plane, cost, cs, x, y, options, params, height, width);
}

__global__ void PropagateBlackTexture(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  

  if (threadIdx.x % 2 == 0) y = y * 2;
  else y = y * 2 + 1;

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  PropagateCloseTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
  PropagateFarTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
}

__global__ void PropagateBlack(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (threadIdx.x % 2 == 0) y = y * 2;
  else y = y * 2 + 1;

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  PropagateClose(color_left, color_right, grad_left, grad_right, plane, cost, cs, x, y, options, params, height, width);
  PropagateFar(color_left, color_right, grad_left, grad_right, plane, cost, cs, x, y, options, params, height, width);
}



__global__ void PropagateInOrder(
  const uint8_t *color_left, 
  const uint8_t *color_right, 
  const Gradient *grad_left, 
  const Gradient *grad_right, 
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width,
  int iter
)
{
  const int dir = (iter%2==0) ? 1 : -1;
	int y = (dir == 1) ? 0 : height - 1;
	for (int i = 0; i < height; i++) {
		int x = (dir == 1) ? 0 : width - 1;
		for (int j = 0; j < width; j++) {
      printf("iter: %d, x: %d, y: %d\n", iter, x, y);
			SpatialPropagation(color_left, color_right, grad_left, grad_right, plane, cost, x, y, -dir, 0, options, params, height, width);
      SpatialPropagation(color_left, color_right, grad_left, grad_right, plane, cost, x, y, 0, -dir, options, params, height, width);
      PlaneRefinement(color_left, color_right, grad_left, grad_right, plane, cost, cs, x, y, options, params, height, width);
			x += dir;
		}
		y += dir;
	}
}

__global__ void PropagateInOrderTexture(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right,  
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width,
  int iter
)
{
  // printf("order\n");
  const int dir = (iter%2==0) ? 1 : -1;
	int y = (dir == 1) ? 0 : height - 1;
	for (int i = 0; i < height; i++) {
		int x = (dir == 1) ? 0 : width - 1;
		for (int j = 0; j < width; j++) {
      // printf("iter: %d, x: %d, y: %d\n", iter, x, y);
			SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, -dir, 0, options, params, height, width);
      SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, 0, -dir, options, params, height, width);
      PlaneRefinementTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
			x += dir;
		}
		y += dir;
	}
}

__global__ void PropagateForward(
  cudaTextureObject_t img_left, 
  cudaTextureObject_t img_right,  
  DispPlane *plane, 
  float *cost, 
  curandState *cs,
  const PatchMatchOptions *options,
  const CalibrationParams *params,
  int height, 
  int width,
  int direct)
{
  switch(direct)
  {
    // left to right
    case 0:
    {
      int y = threadIdx.y;
      if(y < 0 || y >= height) break;
      for(int i = 1; i < width; ++i) {
        int x = i;
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, -1, -1, options, params, height, width);
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, -1, 0, options, params, height, width);
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, -1, 1, options, params, height, width);
        PlaneRefinementTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
      }
    }
    // up to bottom
    case 1:
    {
      int x = threadIdx.x;
      if(x < 0 || x >= width) break;
      for(int i = 1; i < height; ++i) {
        int y = i;
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, -1, -1, options, params, height, width);
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, 0, -1, options, params, height, width);
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, 1, -1, options, params, height, width);
        PlaneRefinementTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
      }
    }


    // right to left
    case 2:
    {
      int y = threadIdx.y;
      if(y < 0 || y >= height) break;
      for(int i = width - 1; i > 0; --i) {
        int x = i;
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, 1, -1, options, params, height, width);
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, 1, 0, options, params, height, width);
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, 1, 1, options, params, height, width);
        PlaneRefinementTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
      }
    }

    // bottom to up
    case 3:
    {
      int x = threadIdx.x;
      if(x < 0 || x >= width) break;
      for(int i = height - 1; i > 0; --i) {
        int y = i;
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, -1, 1, options, params, height, width);
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, 0, 1, options, params, height, width);
        SpatialPropagationTexture(img_left, img_right, plane, cost, x, y, 1, 1, options, params, height, width);
        PlaneRefinementTexture(img_left, img_right, plane, cost, cs, x, y, options, params, height, width);
      }
    }
  }
}

StereoMatcherCuda::StereoMatcherCuda(const int& width, const int& height, PatchMatchOptions *options, CalibrationParams *params)
{
	width_ = width;
	height_ = height;
  options_ = options;
  params_ = params;

	//������ �����ڴ�ռ�
	const int img_size = width * height;
  
  checkCudaErrors(cudaMallocManaged((void **)&gray_left_, sizeof(uint8_t) * img_size));
  checkCudaErrors(cudaMallocManaged((void **)&gray_right_, sizeof(uint8_t) * img_size));

  checkCudaErrors(cudaMallocManaged((void **)&grad_left_, sizeof(Gradient) * img_size));
  checkCudaErrors(cudaMallocManaged((void **)&grad_right_, sizeof(Gradient) * img_size));

  checkCudaErrors(cudaMallocManaged((void **)&cost_left_, sizeof(float) * img_size));
  checkCudaErrors(cudaMallocManaged((void **)&cost_right_, sizeof(float) * img_size));

  checkCudaErrors(cudaMallocManaged((void **)&disp_left_, sizeof(float) * img_size));
  checkCudaErrors(cudaMallocManaged((void **)&disp_right_, sizeof(float) * img_size));

  checkCudaErrors(cudaMallocManaged((void **)&plane_left_, sizeof(DispPlane) * img_size));
  checkCudaErrors(cudaMallocManaged((void **)&plane_right_, sizeof(DispPlane) * img_size));

  checkCudaErrors(cudaMallocManaged((void **)&cs_left_, sizeof(curandState) * img_size));
  checkCudaErrors(cudaMallocManaged((void **)&cs_right_, sizeof(curandState) * img_size));

  cudaMalloc((void **)&img_left_, sizeof(float) * img_size * 3);
  cudaMalloc((void **)&img_right_, sizeof(float) * img_size * 3);

  
	is_initialized_ = grad_left_ && grad_right_ && disp_left_ && disp_right_ && plane_left_ && plane_right_;

}

StereoMatcherCuda::~StereoMatcherCuda()
{
  cudaFree(gray_left_);
  cudaFree(gray_right_);
  cudaFree(grad_left_);
  cudaFree(grad_right_);
  cudaFree(disp_left_);
  cudaFree(disp_right_);
  cudaFree(plane_left_);
  cudaFree(plane_right_);
}


void GetColor(const unsigned char* img_data, Vector3f *vec, const float& x,const int& y, int width)
{
  float col[3];
  const auto x1 = static_cast<int>(x);
  const int x2 = x1 + 1;
  const float ofs = x - x1;

  for (int n = 0; n < 3; n++) {
    const auto& g1 = img_data[y * width * 3 + 3 * x1 + n];
    const auto& g2 = (x2 < width) ? img_data[y * width * 3 + 3 * x2 + n] : g1;
    col[n] = (1 - ofs) * g1 + ofs * g2;
  } 
  vec->x = col[0];
  vec->y = col[1];
  vec->z = col[2];
}



void ComputeGrayHost(const uint8_t *color, uint8_t *gray, int width, int height)
{
	// ��ɫת�Ҷ�
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      const auto b = color[i * width * 3 + 3 * j];
      const auto g = color[i * width * 3 + 3 * j + 1];
      const auto r = color[i * width * 3 + 3 * j + 2];
      gray[i * width + j] = uint8_t(r * 0.299 + g * 0.587 + b * 0.114);
    }
  }
}

void ComputeGradientHost(const uint8_t *gray, Gradient *grad, int width, int height)
{
	// Sobel�ݶ�����

  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      const auto grad_x = (-gray[(y - 1) * width + x - 1] + gray[(y - 1) * width + x + 1]) +
        (-2 * gray[y * width + x - 1] + 2 * gray[y * width + x + 1]) +
        (-gray[(y + 1) * width + x - 1] + gray[(y + 1) * width + x + 1]);
      const auto grad_y = (-gray[(y - 1) * width + x - 1] - 2 * gray[(y - 1) * width + x] - gray[(y - 1) * width + x + 1]) +
        (gray[(y + 1) * width + x - 1] + 2 * gray[(y + 1) * width + x] + gray[(y + 1) * width + x + 1]);
      grad[y * width + x].x = grad_x / 8;
      grad[y * width + x].y = grad_y / 8;
    }
  }
}

void ShowDisparityAndNormalMap(DispPlane *plane_data, int height, int width)
{
  cv::Mat_<float> disp_img(height, width);
  cv::Mat_<cv::Vec3f> norm_img(height, width);
  ConvertPlaneToDispAndNormal(plane_data, disp_img, norm_img, height, width);
  
  cv::Mat disp_display, norm_display;
  ConvertDisparityForDisplay(disp_img, disp_display);
  ConvertNormalsForDisplay(norm_img, norm_display);
  cv::imshow("disp_display", disp_display);
  cv::imshow("norm_display", norm_display); 
	while(1)
	{
		if(cv::waitKey(0) == 'q')
			break;
	}
}

void ShowCostAndHistogramMap(float *cost, int height, int width)
{
  cv::Mat cost_img(height, width, CV_32F, cv::Scalar(0));
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      cost_img.ptr<float>(i)[j] = (float)cost[i*width+j];
    }
  }
  cv::Mat cost_display;
  ConvertCostForDisplay(cost_img, cost_display);
  cv::Mat hist_display;
  ComputeHistForDisplay(cost_img, hist_display);
  cv::imshow("cost", cost_display);
  cv::imshow("hist", hist_display);
  while(1) {
    if(cv::waitKey(0) == 'q') break;
  }

}



bool StereoMatcherCuda::Match(const uint8_t* img_left, const uint8_t* img_right, float* disp_left)
{
  printf("Start!\n");
  // cudaMemcpy(img_left_, img_left, sizeof(uint8_t) * width_ * height_ * 3, cudaMemcpyHostToDevice);
  // cudaMemcpy(img_right_, img_right, sizeof(uint8_t) * width_ * height_ * 3, cudaMemcpyHostToDevice);

  img_left_ = img_left;
  img_right_ = img_right;

  int block_size_x = 32;
  int block_size_y = 16;
  dim3 block_size(block_size_x, block_size_y);
  dim3 grid_size((width_ + block_size_x - 1)/block_size_x, (height_ + block_size_y - 1)/block_size_y);

  // UnitTestGetColor(img_left, height_, width_);

  uint8_t *img_left_host = new uint8_t[width_ * height_ * 3];
  uint8_t *img_right_host = new uint8_t[width_ * height_ * 3];
  uint8_t *gray_left_host = new uint8_t[width_ * height_];
  uint8_t *gray_right_host = new uint8_t[width_ * height_];
  Gradient *grad_left_host = new Gradient[width_ * height_];
  Gradient *grad_right_host = new Gradient[width_ * height_];
  DispPlane *plane_host = new DispPlane[width_ * height_];


  // cv::Mat grayf_left(height_, width_, CV_32F);
  // cv::Mat grayf_right(height_, width_, CV_32F);

  // GetGrayImageFloat <<<grid_size, block_size>>>(img_left, (float*)grayf_left.data, height_, width_);
  // GetGrayImageFloat <<<grid_size, block_size>>>(img_right, (float*)grayf_right.data, height_, width_);

  printf("here\n");
  cudaMemcpy(img_left_host, img_left, sizeof(uint8_t) * width_ * height_ * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(img_right_host, img_right, sizeof(uint8_t) * width_ * height_ * 3, cudaMemcpyDeviceToHost);


  RandomInit <<<grid_size, block_size>>>(cs_left_, time(nullptr), height_, width_);
  RandomPlane <<<grid_size, block_size>>>(plane_left_, cs_left_, options_, params_, height_, width_, false);
  cudaDeviceSynchronize();
  ShowDisparityAndNormalMap(plane_left_, height_, width_);


  GetGrayImage <<<grid_size, block_size>>>(img_left, gray_left_, height_, width_);
  // ComputeGrayHost(img_left_host, gray_left_host, width_, height_);
  // for(int i = 0; i < height_; ++i) {
  //   for(int j = 0; j < width_; ++j) {
  //     printf("gray host: %f, gray cuda: %f\n", (float)gray_left_host[i * width_ + j], (float)gray_left_[i * width_ + j]);
  //   }
  // }

  GetGradientImage <<<grid_size, block_size>>>(gray_left_, grad_left_, height_, width_);
  // ComputeGradientHost(gray_left_host, grad_left_host, width_, height_);
  // for(int i = 1; i < height_ - 1; ++i) {
  //   for(int j = 1; j < width_ - 1; ++j) {
  //     printf("grad host x: %f, grad cuda x: %f\n", (float)grad_left_host[i * width_ + j].x, (float)grad_left_[i * width_ + j].x);
  //     printf("grad host y: %f, grad cuda y: %f\n", (float)grad_left_host[i * width_ + j].y, (float)grad_left_[i * width_ + j].y);
  //   }
  // }

  GetInitialCost <<<grid_size, block_size>>>(img_left, img_right, grad_left_, grad_right_, plane_left_, options_, cost_left_, height_, width_);
  cudaDeviceSynchronize();
  // GetInitialCostTexture <<<grid_size, block_size>>>(tex0, tex1, plane_left_, options_, cost_left_, height_, width_);
  ShowCostAndHistogramMap(cost_left_, height_, width_);


  float *cost_host = new float[width_ * height_];
  // cudaMemcpy(grad_left_host, grad_left_, sizeof(Gradient) * width_ * height_, cudaMemcpyDeviceToHost);
  // cudaMemcpy(grad_right_host, grad_right_, sizeof(Gradient) * width_ * height_, cudaMemcpyDeviceToHost);
  cudaMemcpy(plane_host, plane_left_, sizeof(DispPlane) * width_ * height_, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  // ComputeGrayHost(img_left_host, gray_left_host, width_, height_);
  // ComputeGradientHost(gray_left_host, grad_left_host, width_, height_);
  // GetInitialCostHost(img_left_host, img_right_host, grad_left_host, grad_right_host, plane_host, options_, cost_host, height_, width_);

  // cudaMemcpy(cost_left_, cost_host, sizeof(float) * width_ * height_, cudaMemcpyHostToDevice);
  // PropagateInOrderTexture <<<grid_size, block_size>>>(img_left, img_right, grad_left_, grad_right_, plane_left_, cost_left_, cs_left_, options_, params_, height_, width_);

  for(int iter = 0; iter < 8; iter++) {
    printf("iter: %d\n", iter);
    PropagateRed <<<grid_size, block_size>>>(img_left, img_right, grad_left_, grad_right_, plane_left_, cost_left_, cs_left_, options_, params_, height_, width_);
    cudaDeviceSynchronize();
    printf("show red\n");
    ShowDisparityAndNormalMap(plane_left_, height_, width_);
    PropagateBlack <<<grid_size, block_size>>>(img_left, img_right, grad_left_, grad_right_, plane_left_, cost_left_, cs_left_, options_, params_, height_, width_);
    cudaDeviceSynchronize();
    printf("show black\n");
    ShowDisparityAndNormalMap(plane_left_, height_, width_);
  }
  // printf("Start Propagate!\n");
  // for(int iter = 0; iter < 1; ++iter) {
  //   PropagateInOrder<<<1, 1>>>(img_left, img_right, grad_left_, grad_right_, plane_left_, cost_left_, cs_left_, options_, params_, height_, width_, iter);
  //   cudaDeviceSynchronize();
  // }
  
  
  ShowCostAndHistogramMap(cost_left_, height_, width_);

  // for(int i = 100; i < 120; i++) {
  //   for(int j = 100; j < 120; j++) {
  //     printf("cost host = %f, cost device = %f\n", cost_host[i*width_+j],cost_left_[i*width_+j]);
  //   }
  // }

  delete img_left_host;
  delete img_right_host;
  delete gray_left_host;
  delete gray_right_host;
  delete grad_left_host;
  delete grad_right_host;


  return true;
}

bool StereoMatcherCuda::Match(const uint8_t* color_left, const uint8_t* color_right, cv::Mat &depth_img, cv::Mat &normal_img, bool texture)
{
  printf("Start Texture!\n");

  int block_size_x = 32;
  int block_size_y = 16;
  dim3 block_size(block_size_x, block_size_y);
  dim3 grid_size((width_ + block_size_x - 1)/block_size_x, (height_ + block_size_y - 1)/block_size_y);

  printf("GridSize is %dx%dx%d\n", grid_size.x, grid_size.y, grid_size.z);
  printf("BlockSize is %dx%dx%d\n", block_size.x, block_size.y, block_size.z);
  printf("ImagesSize is %dx%d\n", width_, height_);

  RandomInit <<<grid_size, block_size>>>(cs_left_, time(nullptr), height_, width_);
  RandomPlane <<<grid_size, block_size>>>(plane_left_, cs_left_, options_, params_, height_, width_, false);
  cudaDeviceSynchronize();
  ShowDisparityAndNormalMap(plane_left_, height_, width_);

  GetInitialCostTexture <<<grid_size, block_size>>>(tex_left_, tex_right_, plane_left_, options_, cost_left_, height_, width_);
  // GetInitialCostTexturePoint <<<1, 1>>>(img_left, img_right, plane_left_, options_, 100, 100, cost_left_, height_, width_);
  cudaDeviceSynchronize();
  ShowCostAndHistogramMap(cost_left_, height_, width_);

  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t used = total - avail;
  printf("Device memory used: %fMB\n", used/1000000.0f);

  for(int iter = 0; iter < 8; iter++) {
    printf("iter: %d\n", iter);
    PropagateRedTexture <<<grid_size, block_size>>>(tex_left_, tex_right_, plane_left_, cost_left_, cs_left_, options_, params_, height_, width_);
    cudaDeviceSynchronize();
    // printf("show red\n");
    ShowDisparityAndNormalMap(plane_left_, height_, width_);
    PropagateBlackTexture <<<grid_size, block_size>>>(tex_left_, tex_right_, plane_left_, cost_left_, cs_left_, options_, params_, height_, width_);
    cudaDeviceSynchronize();
    // printf("show black\n");
    ShowDisparityAndNormalMap(plane_left_, height_, width_);
  }

  // {
  //   dim3 block_size_iter(1, height_);
  //   dim3 grid_size_iter(1, 1);
  //   PropagateForward <<<grid_size_iter, block_size_iter>>>(tex_left_, tex_right_, plane_left_, cost_left_, cs_left_, options_, params_, height_, width_, 0);
  //   cudaDeviceSynchronize();
  //   printf("Complete l2r!\n");
  //   ShowDisparityAndNormalMap(plane_left_, height_, width_);
  // }

  // {
  //   dim3 block_size_iter(width_, 1);
  //   dim3 grid_size_iter(1, 1);
  //   PropagateForward <<<grid_size_iter, block_size_iter>>>(tex_left_, tex_right_, plane_left_, cost_left_, cs_left_, options_, params_, height_, width_, 1);
  //   cudaDeviceSynchronize();
  //   printf("Complete u2b!\n");
  //   ShowDisparityAndNormalMap(plane_left_, height_, width_);
  // }

  // {
  //   dim3 block_size_iter(1, height_);
  //   dim3 grid_size_iter(1, 1);
  //   PropagateForward <<<grid_size_iter, block_size_iter>>>(tex_left_, tex_right_, plane_left_, cost_left_, cs_left_, options_, params_, height_, width_, 2);
  //   cudaDeviceSynchronize();
  //   printf("Complete r2l!\n");
  //   ShowDisparityAndNormalMap(plane_left_, height_, width_);
  // }

  // {
  //   dim3 block_size_iter(width_, 1);
  //   dim3 grid_size_iter(1, 1);
  //   PropagateForward <<<grid_size_iter, block_size_iter>>>(tex_left_, tex_right_, plane_left_, cost_left_, cs_left_, options_, params_, height_, width_, 3);
  //   cudaDeviceSynchronize();
  //   printf("Complete b2u!\n");
  //   ShowDisparityAndNormalMap(plane_left_, height_, width_);
  // }


  ShowDisparityAndNormalMap(plane_left_, height_, width_);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("Error: %s\n", cudaGetErrorString(err));

  ShowCostAndHistogramMap(cost_left_, height_, width_);

  depth_img = cv::Mat_<float>(height_, width_);
  normal_img = cv::Mat_<cv::Vec3f>(height_, width_);
  // ConvertPlaneToDepthAndNormal(ref_->plane_, depth, normal, height, width);
  for(int i = 0; i < height_; ++i) {
    for(int j = 0; j < width_; ++j) {
      depth_img.at<float>(i,j) = params_->bf / plane_left_[i*width_+j].a * j + plane_left_[i*width_+j].b * i + plane_left_[i*width_+j].c;
      Vector3f norm_ij = plane_left_[i*width_+j].to_normal();
      normal_img.at<cv::Vec3f>(i,j) = cv::Vec3f(norm_ij.x, norm_ij.y, norm_ij.z);
    }
  }

  return true;
}

