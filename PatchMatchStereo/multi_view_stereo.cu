#include "multi_view_stereo.h"

#define COST_PUNISH 120.0f

__device__ __forceinline__ void Mat33DotVec3(const float mat[9], const float vec[3], float result[3]) 
{
  result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

__device__ __forceinline__ void ComputeRelativePose(
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

__device__ __forceinline__ void ComputeHomography(
  const float *K_ref, 
  const float *K_src, 
  const float *R1,
  const float *R2,
  const float *t1,
  const float *t2,
  const int x, 
  const int y, 
  const float depth, 
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

  const float src_inv_fx = 1 / src_fx, src_inv_cx = -src_cx / src_fx,
              src_inv_fy = 1 / src_fy, src_inv_cy = -src_cy / src_fy;
  
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

__device__ __forceinline__ void HomogeneousWarp(const float mat[9], const float vec[2], float result[2]) 
{
  const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
  result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
  result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
}

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

__device__ __forceinline__ void ConstructPlane(int x, int y, float inv_depth, float nx, float ny, float nz, InvDepthPlane *plane)
{
  plane->a = -nx / nz;
  plane->b = -ny / nz;
  plane->c = (nx * x + ny * y + nz * inv_depth) / nz;
}

__device__ __forceinline__ void PlaneToInvDepth(const InvDepthPlane *plane, const int& x, const int& y, float *inv_depth)
{
  *inv_depth = plane->a * x + plane->b * y + plane->c;
}

__device__ __forceinline__ void PlaneToNormal(const InvDepthPlane *plane, Vector3f *norm)
{
  norm->x = plane->a;
  norm->y = plane->b;
  norm->z = -1.0f;
  Normalize3f(norm);
}

__device__ __forceinline__ void RandomPointPlane(
  PlaneState *local_plane, 
  curandState *local_cs, 
  int x, int y, 
  int height, int width, 
  float mind, float maxd, 
  float *K)
{
  float inv_depth_rd = CurandBetween(local_cs, mind, maxd);

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
  const float& fx = K[0], &fy = K[4], &cx = K[2], &cy = K[5];

  view_ray.x = (x - cx) / fx;
  view_ray.y = (y - cy) / fy;
  view_ray.z = 1;

  float dp = norm_rd.x * view_ray.x + norm_rd.y * view_ray.y + norm_rd.z * view_ray.z;
  if(dp > 0) {
    norm_rd.x = -norm_rd.x;
    norm_rd.y = -norm_rd.y;
    norm_rd.z = -norm_rd.z;
  }
  local_plane->inv_depth_ = inv_depth_rd;
  local_plane->normal_x_ = norm_rd.x;
  local_plane->normal_y_ = norm_rd.y;
  local_plane->normal_z_ = norm_rd.z;
  // ConstructPlane(x, y, inv_depth_rd, norm_rd.x, norm_rd.y, norm_rd.z, local_plane);
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
  PlaneState *plane, 
  curandState *cs, 
  const PatchMatchOptions *options,
  int height, int width, float *K,
  bool is_right)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1)  return;

  int center = y * width + x;
  // printf("max inv_depth: %f, min inv_depth: %f\n", options->max_disparity, options->min_disparity);
  RandomPointPlane(&plane[center], &cs[center], x, y, height, width, options->min_disparity, options->max_disparity, K);
}

__device__ __forceinline__ float PatchMatchCostMultiView(
  const cudaTextureObject_t tex_ref, 
  const cudaTextureObject_t tex_src, 
  int rx, int ry,
  float sx, float sy,
  const PatchMatchOptions *options,
  int height, int width
)
{
  if (sx < 0.0f || sx >= static_cast<float>(width) || sy < 0.0f || sy >= static_cast<float>(height)) {
    return (1 - options->alpha) * options->tau_col + options->alpha * options->tau_grad;
  }
  const float col_p = color(tex_ref, rx, ry);
  const float col_q = color(tex_src, sx, sy);
  const float dc = fminf(fabs(col_p - col_q), options->tau_col);

  const float grad_px = gradx(tex_ref, rx, ry);
  const float grad_qx = gradx(tex_src, sx, sy);
  const float grad_py = grady(tex_ref, rx, ry);
  const float grad_qy = grady(tex_src, sx, sy);

  const float dg = fminf(fabs(grad_px - grad_qx)+ fabs(grad_py - grad_qy), options->tau_grad);
  return (1 - options->alpha) * dc + options->alpha * dg;
}

__device__ __forceinline__ float PatchMatchCostMultiViewRegion(
  ReferenceViewSpace *view_ref, 
  ViewSpace *view_src[],
  int src_size,
  const int& px, const int& py, 
  const PlaneState *local_plane, 
  const PatchMatchOptions *options, 
  int height, int width
)
{
  const int pat = options->patch_size / 2;
  float col_p = color(view_ref->tex_, px, py);
  const float& idp = local_plane->inv_depth_;
  const float& nx = local_plane->normal_x_;
  const float& ny = local_plane->normal_y_;
  const float& nz = local_plane->normal_z_;

  const float plane_a = -nx / nz;
  const float plane_b = -ny / nz;
  const float plane_c = (nx * px + ny * py + nz * idp) / nz;

  float cost = 0.0f;
  for(int idx = 0; idx < src_size; ++idx) {
    const ViewSpace *view_src_i = view_src[idx];
    float Hsr[9];
    float normal[3] = {nx, ny, nz};
    ComputeHomography(
      view_ref->params_->K_, 
      view_src_i->params_->K_, 
      view_ref->params_->R_,
      view_src_i->params_->R_,
      view_ref->params_->t_,
      view_src_i->params_->t_,
      px, py, 1/idp, normal, Hsr);

    for (int r = -pat; r <= pat; r+=1) {
      const int qy = py + r;
      for (int c = -pat; c <= pat; c+=1) {
        const int qx = px + c;
        // printf("x, y: %d, %d\n", qx, qy);
        if (qy < 0 || qy > height - 1 || qx < 0 || qx > width - 1) {
          // printf("skip\n");
          continue;
        }
        // �����Ӳ�ֵ
        float idq = plane_a * qx + plane_b * qy + plane_c;
        if (idq < options->min_disparity || idq > options->max_disparity) {
          cost += COST_PUNISH;
          // printf("out of range\n");
          continue;
        }
  
        // // ����Ȩֵ
        float col_q = color(view_ref->tex_, qx, qy);;
        const float dc = fabs(col_p - col_q);
        const float ds = abs(px - qx) + abs(py - qy);
        // // printf("dc=%f\n", dc);
        const float w = expf(-ds / options->sigma_s - dc / options->sigma_r);
        // // printf("weight=%f\n", w);
        float qr[2] = {(float)qx, (float)qy}, qs[2];
        HomogeneousWarp(Hsr, qr, qs);
        
        cost += w * PatchMatchCostMultiView(view_ref->tex_, view_src_i->tex_, qx, qy, qs[0], qs[1], options, height, width);
        // // printf("cost=%f\n", ComputeCostPointTexture(img_left, img_right, qx, qy, dq, options, width));
        // // printf("\n");
      }
    }
  }
  return cost;
}

__global__ void CostInit(
  ReferenceViewSpace *view_ref, 
  ViewSpace *view_src[],
  const int src_size,
  const PatchMatchOptions *options,
  const int height, 
  const int width
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  int center = x + y * width;
  view_ref->cost_[center] = PatchMatchCostMultiViewRegion(view_ref, view_src, src_size, x, y, &view_ref->plane_[center], options, height, width);
}

void MultiViewStereoMatcherCuda::Match()
{

  int block_size_x = 32;
  int block_size_y = 16;
  const int width = ref_->width_;
  const int height = ref_->height_;
  dim3 block_size(block_size_x, block_size_y);
  dim3 grid_size((width + block_size_x - 1)/block_size_x, (height + block_size_y - 1)/block_size_y);

  printf("GridSize is %dx%dx%d\n", grid_size.x, grid_size.y, grid_size.z);
  printf("BlockSize is %dx%dx%d\n", block_size.x, block_size.y, block_size.z);
  printf("ImagesSize is %dx%d\n", width, height);

  RandomInit <<<grid_size, block_size>>>(ref_->cs_, time(nullptr), height, width);
  cudaDeviceSynchronize();
  RandomPlane <<<grid_size, block_size>>>(ref_->plane_, ref_->cs_, options_, height, width, ref_->params_->K_, false);
  cudaDeviceSynchronize();

  CostInit <<<grid_size, block_size>>>(ref_, src_, image_size_, options_, height, width);


  ShowDepthAndNormal(ref_->plane_, height, width);
  ShowCostAndHistogram(ref_->cost_, height, width);
}

// test homography and warp!!
