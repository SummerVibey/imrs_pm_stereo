#include "multi_view_stereo.h"

#define COST_PUNISH 120.0f


__device__ __forceinline__ void Mat33DotVec3(const float mat[9], const float vec[3], float result[3]) 
{
  result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

__device__ __forceinline__ void Mat33MulMat33(const float mat1[9], const float mat2[9], float result[9]) 
{
  result[0] = mat1[0] * mat2[0] + mat1[1] * mat2[0+3] + mat1[2] * mat2[0+6];
  result[1] = mat1[0] * mat2[1] + mat1[1] * mat2[1+3] + mat1[2] * mat2[1+6]; 
  result[2] = mat1[0] * mat2[2] + mat1[1] * mat2[2+3] + mat1[2] * mat2[2+6]; 
  result[3] = mat1[3] * mat2[0] + mat1[4] * mat2[0+3] + mat1[5] * mat2[0+6]; 
  result[4] = mat1[3] * mat2[1] + mat1[4] * mat2[1+3] + mat1[5] * mat2[1+6]; 
  result[5] = mat1[3] * mat2[2] + mat1[4] * mat2[2+3] + mat1[5] * mat2[2+6]; 
  result[6] = mat1[6] * mat2[0] + mat1[7] * mat2[0+3] + mat1[8] * mat2[0+6]; 
  result[7] = mat1[6] * mat2[1] + mat1[7] * mat2[1+3] + mat1[8] * mat2[1+6]; 
  result[8] = mat1[6] * mat2[2] + mat1[7] * mat2[2+3] + mat1[8] * mat2[2+6];
}

__device__ __forceinline__ void Vec3AddVec3(const float vec1[9], const float vec2[3], float result[3]) 
{
  result[0] = vec1[0] + vec2[0];
  result[1] = vec1[1] + vec2[1];
  result[2] = vec1[2] + vec2[2];
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
  const float idp, 
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
  
  const float depth = 1 / idp;
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

__device__ __forceinline__ void Warp(const float K1[9], const float K2[9], const float R21[9], const float t21[3], const float x1, const float y1, const float idp1, float &x2, float &y2, float &idp2)
{
  const float &ref_fx = K1[0], &ref_fy = K1[4],
              &ref_cx = K1[2], &ref_cy = K1[5];

  const float ref_inv_fx = 1 / ref_fx, ref_inv_cx = -ref_cx / ref_fx,
              ref_inv_fy = 1 / ref_fy, ref_inv_cy = -ref_cy / ref_fy;
  
  const float depth1 = 1 / idp1;
  const float pt3d[3] = {depth1 * (ref_inv_fx * x1 + ref_inv_cx), depth1 * (ref_inv_fy * y1 + ref_inv_cy), depth1};

  // const float K1inv[9] = {ref_inv_fx, 0, ref_inv_cx, 0, ref_inv_fy, ref_inv_cy, 0, 0, 1};

  // const float pt3d0[3] = {x1 * depth1, y1 * depth1, depth1};
  // printf("x1, y1, depth1, disp1: %f, %f, %f, %f\n", x1, y1, depth1, 180.0f / depth1);
  // printf("pt3d:\n");
  // for(int j = 0; j < 3; ++j) {
  //   printf("%f   ", pt3d0[j]);
  // }
  // printf("\n");
  // float RKinv[9], KRKinv[9];
  // Mat33MulMat33(R21, K1inv, RKinv);
  // Mat33MulMat33(K2, RKinv, KRKinv);
  // printf("KRKinv:\n");
  // for(int i = 0; i < 3; ++i) {
  //   for(int j = 0; j < 3; ++j) {
  //     printf("%f   ", KRKinv[i*3+j]);
  //   }
  //   printf("\n");
  // }
  // float Kt[3];
  // printf("Kt:\n");
  // Mat33DotVec3(K2, t21, Kt);
  // for(int j = 0; j < 3; ++j) {
  //   printf("%f   ", Kt[j]);
  // }
  // printf("\n");
  
  // float KRKinvP[3], pt2d2[3];
  // Mat33DotVec3(KRKinv, pt3d0, KRKinvP);
  // Vec3AddVec3(KRKinvP, Kt, pt2d2);
  // printf("pt2d2:\n");
  // for(int j = 0; j < 3; ++j) {
  //   printf("%f   ", pt2d2[j]/pt2d2[2]);
  // }
  // printf("\n");

  float Rp[3], Rpt[3], pt2d[3];
  Mat33DotVec3(R21, pt3d, Rp);
  Vec3AddVec3(Rp, t21, Rpt);
  Mat33DotVec3(K2, Rpt, pt2d);
  x2 = pt2d[0] / pt2d[2];
  y2 = pt2d[1] / pt2d[2];
  idp2 = 1 / pt2d[2];
}

__device__ __forceinline__ void HomogeneousWarp(const float mat[9], const float vec[2], float result[2]) 
{
  const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
  result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
  result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
}

__device__ __forceinline__ void TransformView(const float mat[9], const float px, const float py, float &qx, float &qy) 
{
  const float inv_z = 1.0f / (mat[6] * px + mat[7] * py + mat[8]);
  qx = inv_z * (mat[0] * px + mat[1] * py + mat[2]);
  qy = inv_z * (mat[3] * px + mat[4] * py + mat[5]);
}

__global__ void TestHomographyWarp(
  const float *K, 
  const float *R1,
  const float *R2,
  const float *t1,
  const float *t2,
  int x, int y, 
  float depth, 
  float normal[3]
)
{
  float H[9];
  ComputeHomography(K, K, R1, R2, t1, t2, x, y, depth, normal, H);
  float input[2] = {(float)x, (float)y}, output[2];
  HomogeneousWarp(H, input, output);  
  printf("warped result: %f, %f\n", output[0], output[1]);
}

__device__ __forceinline__ float BilateralWeight(
  const float x_diff, 
  const float y_diff, 
  const float color_diff, 
  const float sigma_s, 
  const float sigma_c)
{
  // return expf( -(x_diff * x_diff + y_diff * y_diff) / (2 * sigma_s * sigma_s) - color_diff * color_diff / (2 * sigma_c * sigma_c));
  return expf( -sqrtf(x_diff * x_diff + y_diff * y_diff) / sigma_s - abs(color_diff) / sigma_c);
}

__device__ __forceinline__ float CurandBetween(curandState *cs, float min, float max)
{
  return (curand_uniform(cs) * (max-min) + min);
}

__device__ __forceinline__ void Normalize3f(Vector3f *vec)
{
  const float inv_sqr = 1.0f / norm3f(vec->x, vec->y, vec->z);
  vec->x *= inv_sqr;
  vec->y *= inv_sqr;
  vec->z *= inv_sqr;
}

__device__ __forceinline__ void Normalize(float *nx, float *ny, float *nz)
{
  const float inv_sqr = 1.0f / norm3f(*nx, *ny, *nz);
  *nx *= inv_sqr;
  *ny *= inv_sqr;
  *nz *= inv_sqr;
}

__device__ __forceinline__ void ConstructPlane(int x, int y, float inv_depth, float nx, float ny, float nz, Plane *plane)
{
  plane->a_ = -nx / nz;
  plane->b_ = -ny / nz;
  plane->c_ = (nx * x + ny * y + nz * inv_depth) / nz;
}

__device__ __forceinline__ void PlaneToIDepth(const Plane *plane, const int& x, const int& y, float *idp)
{
  *idp = plane->a_ * x + plane->b_ * y + plane->c_;
}

__device__ __forceinline__ void PlaneToNormal(const Plane *plane, Vector3f *norm)
{
  norm->x = plane->a_;
  norm->y = plane->b_;
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

  local_plane->idp_ = inv_depth_rd;
  local_plane->nx_ = norm_rd.x;
  local_plane->ny_ = norm_rd.y;
  local_plane->nz_ = norm_rd.z;
}

__global__ void RandomInit(curandState *cs, unsigned long long seed, int height, int width)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 
  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;

  int center = y * width + x;
  curand_init(seed, y, x, &cs[center]);
}

__global__ void GenerateRandomState(
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

__global__ void RandomPlaneInOrder(
  PlaneState *plane, 
  curandState *cs, 
  const PatchMatchOptions *options,
  int height, int width, float *K,
  bool is_right)
{

  for(int y = 0; y < height; ++y) {
    for(int x = 0; x < width; ++x) {
      int center = y * width + x;
      RandomPointPlane(&plane[center], &cs[center], x, y, height, width, options->min_disparity, options->max_disparity, K);
    }
  }
}

__device__ __forceinline__ float ComputePMCost(
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

__device__ __forceinline__ float ComputePMCostRegion(
  RefViewSpace *view_ref, 
  ViewSpace *view_src,
  const int px, const int py, 
  const float idp, 
  const float nx, const float ny, const float nz,
  const PatchMatchOptions *options, 
  int height, int width
)
{

  const int pat = options->patch_size / 2;
  float ref_color_center = color(view_ref->tex_, px, py);

  float cost = 0.0f;
  // for(int idx = 0; idx < src_size; ++idx) {
  float Hsr[9];
  float normal[3] = {nx, ny, nz};
  ComputeHomography(
    view_ref->params_->K_, 
    view_src->params_->K_, 
    view_ref->params_->R_,
    view_src->params_->R_,
    view_ref->params_->t_,
    view_src->params_->t_,
    px, py, idp, normal, Hsr);

  for (int r = -pat; r <= pat; r+=1) {
    const int qy = py + r;
    for (int c = -pat; c <= pat; c+=1) {
      const int qx = px + c;

      float ref_color = color(view_ref->tex_, qx, qy);
      const float weight = BilateralWeight(px - qx, py - qy, ref_color_center - ref_color, options->sigma_s, options->sigma_c);

      // float qsx, qsy;
      float qsx, qsy;
      TransformView(Hsr, qx, qy, qsx, qsy);
      // Warp(view_ref->params_->K_, view_src->params_->K_, R21, t21, qx, qy, idq / 180.0f, qsx, qsy);
      cost += weight * ComputePMCost(view_ref->tex_, view_src->tex_, qx, qy, qsx, qsy, options, height, width);

    }
  }

  return cost;
}

// __device__ __forceinline__ float ComputeGeometryConsistencyCost(
//   RefViewSpace *view_ref, 
//   ViewSpace *view_src,
//   const int px, const int py, 
//   const float idp, 
//   const float nx, const float ny, const float nz,
//   const PatchMatchOptions *options, 
//   int height, int width
// )


__device__ __forceinline__ float ComputePhotoConsistencyCost(
  cudaTextureObject_t tex_ref,
  cudaTextureObject_t tex_src,
  const int pcx, const int pcy, 
  const float Hsr[9],
  const float idp, 
  const float nx, const float ny, const float nz,
  const PatchMatchOptions *options
) 
{
  const int window_radius = options->patch_size / 2;
  const float ref_center_color = color(tex_ref, pcx, pcy);

  float ref_color_sum = 0.0f;
  float ref_color_squared_sum = 0.0f;
  float src_color_sum = 0.0f;
  float src_color_squared_sum = 0.0f;
  float src_ref_color_sum = 0.0f;
  float bilateral_weight_sum = 0.0f;

  int cnt = 0;
  for (int y_bias = -window_radius; y_bias <= window_radius; y_bias+=1) {
    for (int x_bias = -window_radius; x_bias <= window_radius; x_bias+=1) {

      float px, py, qx, qy;
      px = pcx + x_bias;
      py = pcy + y_bias;
      TransformView(Hsr, px, py, qx, qy);

      const float ref_color = color(tex_ref, px, py);
      const float src_color = color(tex_src, qx, qy);

      float bilateral_weight = BilateralWeight(x_bias, y_bias, ref_color - ref_center_color, options->sigma_s, options->sigma_c);

      const float bilateral_weight_ref = bilateral_weight * ref_color;
      const float bilateral_weight_src = bilateral_weight * src_color;
      ref_color_sum += bilateral_weight_ref;
      src_color_sum += bilateral_weight_src;
      ref_color_squared_sum += bilateral_weight_ref * ref_color;
      src_color_squared_sum += bilateral_weight_src * src_color;
      src_ref_color_sum += bilateral_weight_src * ref_color;
      bilateral_weight_sum += bilateral_weight;
    }
  }

  const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
  ref_color_sum *= inv_bilateral_weight_sum;
  src_color_sum *= inv_bilateral_weight_sum;
  ref_color_squared_sum *= inv_bilateral_weight_sum;
  src_color_squared_sum *= inv_bilateral_weight_sum;
  src_ref_color_sum *= inv_bilateral_weight_sum;

  const float ref_color_var =
      ref_color_squared_sum - ref_color_sum * ref_color_sum;
  const float src_color_var =
      src_color_squared_sum - src_color_sum * src_color_sum;

  // Based on Jensen's Inequality for convex functions, the variance
  // should always be larger than 0. Do not make this threshold smaller.
  const float kMinVar = 1e-5f;
  const float kMaxCost = 2.0f;
  if (ref_color_var < kMinVar || src_color_var < kMinVar) {
    return kMaxCost;
  } else {
    const float src_ref_color_covar =
        src_ref_color_sum - ref_color_sum * src_color_sum;
    const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
    return max(0.0f,
                min(kMaxCost, 1.0f - src_ref_color_covar / src_ref_color_var));
  }
}

__device__ __forceinline__ float ComputeGeometryConsistencyCost(
  RefViewSpace *view_ref,
  ViewSpace *view_src,
  const int px, const int py, 
  const float Hsr[9],
  const float idp, 
  const float nx, const float ny, const float nz,
  const PatchMatchOptions *options
) 
{
  const float max_cost = 5.0f;
  // Extract projection matrices for source image.
  const float *K_ref = view_ref->params_->K_;
  const float *K_src = view_src->params_->K_;

  const float &ref_fx = K_ref[0], &ref_fy = K_ref[4],
              &ref_cx = K_ref[2], &ref_cy = K_ref[5];

  const float &src_fx = K_src[0], &src_fy = K_src[4],
              &src_cx = K_src[2], &src_cy = K_src[5];

  const float ref_inv_fx = 1 / ref_fx, ref_inv_cx = -ref_cx / ref_fx,
              ref_inv_fy = 1 / ref_fy, ref_inv_cy = -ref_cy / ref_fy;

  const float src_inv_fx = 1 / src_fx, src_inv_cx = -src_cx / src_fx,
              src_inv_fy = 1 / src_fy, src_inv_cy = -src_cy / src_fy;
  
  const float dp = 1 / idp;
  const float forward_point[3] = {dp * (ref_inv_fx * px + ref_inv_cx), dp * (ref_inv_fy * py + ref_inv_cy), dp};

  float R21[9], t21[3];
  float qx, qy, idq;
  ComputeRelativePose(view_ref->params_->R_, view_ref->params_->t_, view_src->params_->R_, view_src->params_->t_, R21, t21);
  Warp(view_ref->params_->K_, view_src->params_->K_, R21, t21, px, py, idp, qx, qy, idq);

  const float dq = 1 / idq;
  const float backward_point[3] = {dq * (src_inv_fx * qx + src_inv_cx), dq * (src_inv_fy * qy + src_inv_cy), dq};

  float R12[9], t12[3];
  float rx, ry, idr;
  ComputeRelativePose(view_src->params_->R_, view_src->params_->t_, view_ref->params_->R_, view_ref->params_->t_, R12, t12);
  Warp(view_src->params_->K_, view_ref->params_->K_, R12, t12, qx, qy, idq, rx, ry, idr);

  // Return truncated reprojection error between original observation and
  // the forward-backward projected observation.
  const float diff_x = px - rx;
  const float diff_y = py - ry;
  return min(max_cost, sqrt(diff_x * diff_x + diff_y * diff_y));
}

__device__ __forceinline__ void PropagateSpatial(
  RefViewSpace *view_ref,
  ViewSpace *view_src,
  PlaneState *plane, 
  float *cost, 
  int x, int y, int x_bias, int y_bias,
  const PatchMatchOptions *options,
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
  PlaneState &plane_local = plane[center_local];
  float &cost_local = cost[center_local];

  PlaneState &plane_nb = plane[center_nb];
  const float& nx = plane_nb.nx_;
  const float& ny = plane_nb.ny_;
  const float& nz = plane_nb.nz_;
  const float& d1 = 1 / plane_nb.idp_;
  // printf("Current State: %f, %f, %f, Current Cost: %f\n", plane_local.a, plane_local.b, plane_local.c, cost_local);
  float Hsr[9];
  float normal[3] = {plane_nb.nx_, plane_nb.ny_, plane_nb.nz_};
  float plane_a = -plane_nb.nx_/plane_nb.nz_;
  float plane_b = -plane_nb.ny_/plane_nb.nz_;
  float plane_c = (plane_nb.nx_ * x_nb + plane_nb.ny_ * y_nb + plane_nb.nz_ * plane_nb.idp_) / plane_nb.nz_;
  // float idp = plane_a * x_nb + plane_b * y_nb + plane_c;
  float idp = plane_nb.idp_;
  // float idp = 1 / ((nx * x + ny * y + nz) * d1 / (nx * x_nb + ny * x_nb + nz));
  ComputeHomography(
    view_ref->params_->K_, 
    view_src->params_->K_, 
    view_ref->params_->R_,
    view_src->params_->R_,
    view_ref->params_->t_,
    view_src->params_->t_,
    x, y, idp, normal, Hsr);

  // float cost_nb = ComputePhotoConsistencyCost(
  //   view_ref->tex_, view_src->tex_, x, y, Hsr, idp, normal[0], normal[1], normal[2], options);
  float cost_nb = ComputePMCostRegion(view_ref, view_src, x, y, plane_nb.idp_, plane_nb.nx_, plane_nb.ny_, plane_nb.nz_, options, height, width);

  // printf("Neighbor State: %f, %f, %f, Neighbor Cost: %f\n", plane_nb.a, plane_nb.b, plane_nb.c, cost_nb);
  if(cost_nb < cost_local) {
    cost_local = cost_nb;
    plane_local.idp_ = idp;
    plane_local.nx_ = normal[0];
    plane_local.ny_ = normal[1];
    plane_local.nz_ = normal[2];
    // printf("Update, New Cost: %f\n", cost_nb);
  }
  // printf("\n");
}

__device__ __forceinline__ void RefinePlane(
  RefViewSpace *view_ref,
  ViewSpace *view_src,
  PlaneState *plane, 
  PlaneState *plane_buf,
  float *cost, 
  int x, int y,
  const PatchMatchOptions *options,
  int buf_size,
  int height, 
  int width
)
{
  const float max_disp = static_cast<float>(options->max_disparity);
	const float min_disp = static_cast<float>(options->min_disparity);

  // ����p��ƽ�桢���ۡ��Ӳ����
  int center_local = y * width + x;
	PlaneState& plane_local = plane[center_local];
  float& cost_local = cost[center_local];
  
	float disp_update = (max_disp - min_disp) / 2.0f;
	float norm_update = 1.0f;
  const float stop_thres = 0.01f;
  
  for(int idx = 0; idx < buf_size; ++idx) {
    PlaneState& plane_new = plane_buf[idx];
    float idp_new = plane_new.idp_;
    float norm_new[3];
    norm_new[0] = plane_new.nx_;
    norm_new[1] = plane_new.ny_;
    norm_new[2] = plane_new.nz_;

    // float Hsr[9];
    // ComputeHomography(
    //   view_ref->params_->K_, 
    //   view_src->params_->K_, 
    //   view_ref->params_->R_,
    //   view_src->params_->R_,
    //   view_ref->params_->t_,
    //   view_src->params_->t_,
    //   x, y, idp_new, norm_new, Hsr);

    // float cost_new = ComputePhotoConsistencyCost(
    //     view_ref->tex_, view_src->tex_, x, y, Hsr, idp_new, norm_new[0], norm_new[1], norm_new[2], options);
    // printf("New State: %f, %f, %f, New Cost: %f\n", plane_new->a, plane_new->b, plane_new->c, cost_new);
    float cost_new = ComputePMCostRegion(view_ref, view_src, x, y, idp_new, norm_new[0], norm_new[1], norm_new[2], options, height, width);
    if(cost_new < cost_local) {
      cost_local = cost_new;
      plane_local.idp_ = idp_new;
      plane_local.nx_ = norm_new[0];
      plane_local.ny_ = norm_new[1];
      plane_local.nz_ = norm_new[2];
      // printf("Update, New Cost: %f\n", cost_new);
    }
  }
}

__device__ __forceinline__ void GeneratePlaneSet(
  curandState *cs,
  PlaneState *plane_new,
  int buf_size
)
{

}

__device__ __forceinline__ void RefinePlaneBase(
  RefViewSpace *view_ref,
  ViewSpace *view_src,
  PlaneState *plane, 
  float *cost, 
  curandState *cs,
  int x, int y,
  const PatchMatchOptions *options,
  int height, 
  int width
)
{
  const float max_disp = static_cast<float>(options->max_disparity);
	const float min_disp = static_cast<float>(options->min_disparity);

  // ����p��ƽ�桢���ۡ��Ӳ����
  int center_local = y * width + x;
	PlaneState& plane_local = plane[center_local];
	float& cost_local = cost[center_local];

	float disp_update = (max_disp - min_disp) / 2.0f;
	float norm_update = 1.0f;
	const float stop_thres = 0.01f;

	// �����Ż�
  int trial_num = 0;
	while (disp_update > stop_thres) {
    // printf("Try number: %d, disp_update: %f, normal_update: %f\n", trial_num++, disp_update, norm_update);
    
    float idp_rd = CurandBetween(&cs[center_local], -1.f, 1.f) * disp_update;
		float idp_new  = plane_local.idp_ + idp_rd;
		if (idp_new  < min_disp || idp_new  > max_disp) {
			disp_update /= 4.0f;
      norm_update /= 4.0f;
			continue;
		}

		// �� -norm_update ~ norm_update ��Χ���������ֵ��Ϊ������������������
		Vector3f norm_rd;

    norm_rd.x = CurandBetween(&cs[center_local], -1.f, 1.f) * norm_update;
    norm_rd.y = CurandBetween(&cs[center_local], -1.f, 1.f) * norm_update;
    float z = CurandBetween(&cs[center_local], -1.f, 1.f) * norm_update;
    while (z == 0.0f) {
      z = CurandBetween(&cs[center_local], -1.f, 1.f) * norm_update;
    }
    norm_rd.z = z;

    // ��������p�µķ���
    Vector3f norm_new;
    norm_new.x = plane_local.nx_ + norm_rd.x;
    norm_new.y = plane_local.ny_ + norm_rd.y;
    norm_new.z = plane_local.nz_ + norm_rd.z;
    Normalize3f(&norm_new);

    Vector3f view_ray;
    const float& fx = view_ref->params_->K_[0], &fy = view_ref->params_->K_[5], &cx = view_ref->params_->K_[2], &cy = view_ref->params_->K_[6];

    view_ray.x = (x - cx) / fx;
    view_ray.y = (y - cy) / fy;
    view_ray.z = 1;

    float dp = norm_new.x * view_ray.x + norm_new.y * view_ray.y + norm_new.z * view_ray.z;
    if(dp > 0) {
      norm_new.x = -norm_new.x;
      norm_new.y = -norm_new.y;
      norm_new.z = -norm_new.z;
    }

    // DispPlane plane_new;
    // ConstructPlane(x, y, d_p_new, norm_p_new.x, norm_p_new.y, norm_p_new.z, &plane_new);
    // printf("Current State: %f, %f, %f, Current Cost: %f\n", plane_p.a, plane_p.b, plane_p.c, cost_p);

    float Hsr[9];
    float normal[3] = {norm_new.x, norm_new.y, norm_new.z};
    ComputeHomography(
      view_ref->params_->K_, 
      view_src->params_->K_, 
      view_ref->params_->R_,
      view_src->params_->R_,
      view_ref->params_->t_,
      view_src->params_->t_,
      x, y, idp_new, normal, Hsr);

    // float cost_new = ComputePhotoConsistencyCost(
      // view_ref->tex_, view_src->tex_, x, y, Hsr, idp_new, norm_new.x, norm_new.y, norm_new.z, options);
    float cost_new = ComputePMCostRegion(view_ref, view_src, x, y, idp_new, norm_new.x, norm_new.y, norm_new.z, options, height, width);
    // printf("New State: %f, %f, %f, New Cost: %f\n", plane_new->a, plane_new->b, plane_new->c, cost_new);
    if(cost_new < cost_local) {
      cost_local = cost_new;
      plane_local.idp_ = idp_new;
      plane_local.nx_ = norm_new.x;
      plane_local.ny_ = norm_new.y;
      plane_local.nz_ = norm_new.z;
      // printf("Update, New Cost: %f\n", cost_new);
    }
		disp_update /= 4.0f;
		norm_update /= 4.0f;
	}
}

__device__ __forceinline__ void Propagate(
  RefViewSpace *view_ref,
  ViewSpace *view_src,
  PlaneState *plane, 
  float *cost, 
  curandState *cs,
  int x, int y,
  const PatchMatchOptions *options,
  int height, 
  int width
)
{
  const int x_bias[8] = {-1,0,1,0,-5,0,5,0};
  const int y_bias[8] = {0,1,0,-1,0,5,0,-5};

  // const int x_bias[20] = {-1,0,1,0,-3,-2,-1,0,1,2,3,2,1,0,-1,-2,-5,0,5,0};
  // const int y_bias[20] = {0,1,0,-1,0,1,2,3,2,1,0,-1,-2,-3,-2,-1,0,5,0,-5};
  

  for(int i = 0; i < 8; ++i) {
    PropagateSpatial(view_ref, view_src, plane, cost, x, y, x_bias[i], y_bias[i], options, height, width);
    __syncthreads();
  }

  RefinePlaneBase(view_ref, view_src, plane, cost, cs, x, y, options, height, width);
  __syncthreads();
}

__global__ void PropagateBlack(
  RefViewSpace *view_ref,
  ViewSpace *view_src,
  PlaneState *plane, 
  float *cost, 
  curandState *cs,
  const PatchMatchOptions *options,
  int height, 
  int width
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (threadIdx.x % 2 == 0) y = y * 2;
  else y = y * 2 + 1;

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  Propagate(view_ref, view_src, plane, cost, cs, x, y, options, height, width);
}

__global__ void PropagateRed(
  RefViewSpace *view_ref,
  ViewSpace *view_src,
  PlaneState *plane, 
  float *cost, 
  curandState *cs,
  const PatchMatchOptions *options,
  int height, 
  int width
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (threadIdx.x % 2 == 0) y = y * 2 + 1;
  else y = y * 2;

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  Propagate(view_ref, view_src, plane, cost, cs, x, y, options, height, width);
}

__global__ void CostInit(
  RefViewSpace *view_ref, 
  ViewSpace *view_src,
  const int src_size,
  const PatchMatchOptions *options,
  const int height, 
  const int width
)
{
  // // using patch match cost
  // int x = threadIdx.x + blockIdx.x * blockDim.x; 
  // int y = threadIdx.y + blockIdx.y * blockDim.y; 

  // if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  // int center = x + y * width;
  // const PlaneState& plane_local = view_ref->plane_[center];
  // view_ref->cost_[center] = ComputePMCostRegion(view_ref, view_src, x, y, plane_local.idp_, plane_local.nx_, plane_local.ny_, plane_local.nz_, options, height, width);

  // using ncc cost
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  int center = x + y * width;
  const PlaneState& plane_local = view_ref->plane_[center];
  view_ref->cost_[center] = ComputePMCostRegion(view_ref, view_src, x, y, plane_local.idp_, plane_local.nx_, plane_local.ny_, plane_local.nz_, options, height, width);
  // float Hsr[9];
  // float normal[3] = {plane_local.nx_, plane_local.ny_, plane_local.nz_};
  // ComputeHomography(
  //   view_ref->params_->K_, 
  //   view_src->params_->K_, 
  //   view_ref->params_->R_,
  //   view_src->params_->R_,
  //   view_ref->params_->t_,
  //   view_src->params_->t_,
  //   x, y, plane_local.idp_, normal, Hsr);

  // view_ref->cost_[center] = ComputePhotoConsistencyCost(
  //   view_ref->tex_, view_src->tex_, x, y, Hsr, plane_local.idp_, plane_local.nx_, plane_local.ny_, plane_local.nz_, options);
}

__global__ void CostInitInOrder(
  RefViewSpace *view_ref, 
  ViewSpace *view_src,
  const int src_size,
  const PatchMatchOptions *options,
  const int height, 
  const int width
)
{
  for(int y = 0; y < height; ++y) {
    for(int x = 0; x < width; ++x) {
      int center = x + y * width;
      const PlaneState& plane_local = view_ref->plane_[center];
      printf("x, y: %d, %d\n", x, y);
      view_ref->cost_[center] = ComputePMCostRegion(view_ref, view_src, x, y, plane_local.idp_, plane_local.nx_, plane_local.ny_, plane_local.nz_, options, height, width);
    }
  }
}

__global__ void ComputePhotoConsistencyCostTest(
  RefViewSpace *view_ref, 
  ViewSpace *view_src,
  const int src_size,
  const PatchMatchOptions *options,
  float *cost_ncc,
  const int height, 
  const int width
)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x; 
  int y = threadIdx.y + blockIdx.y * blockDim.y; 

  if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  int center = x + y * width;
  const PlaneState& plane_local = view_ref->plane_[center];
  float Hsr[9];
  float normal[3] = {plane_local.nx_, plane_local.ny_, plane_local.nz_};
  ComputeHomography(
    view_ref->params_->K_, 
    view_src->params_->K_, 
    view_ref->params_->R_,
    view_src->params_->R_,
    view_ref->params_->t_,
    view_src->params_->t_,
    x, y, plane_local.idp_, normal, Hsr);

  cost_ncc[center] = ComputePhotoConsistencyCost(
    view_ref->tex_, view_src->tex_, x, y, Hsr, plane_local.idp_, plane_local.nx_, plane_local.ny_, plane_local.nz_, options);
}

__global__ void GetInitialPhotoCost(
  RefViewSpace *view_ref, 
  ViewSpace *view_src[],
  const int src_size,
  const PatchMatchOptions *options,
  const int height, 
  const int width
)
{
  // int x = threadIdx.x + blockIdx.x * blockDim.x; 
  // int y = threadIdx.y + blockIdx.y * blockDim.y; 

  // if(x < 0 || x > width - 1 || y < 0 || y > height - 1) return;
  // int center = x + y * width;
  // // printf("x, y: %d, %d\n", x, y);
  // const PlaneState *local_plane = &view_ref->plane_[center];
  // float depth = 1 / local_plane->idp_;
  // float normal[3] = {local_plane->nx_, local_plane->ny_, local_plane->nz_};
  // view_ref->cost_[center] = ComputePhotoConsistencyCost(
  //   view_ref->tex_, view_src[0]->tex_,
  //   view_ref->params_->K_, view_src[0]->params_->K_,
  //   view_ref->params_->R_, view_src[0]->params_->R_,
  //   view_ref->params_->t_, view_src[0]->params_->t_,
  //   x, y, depth, normal, options);
}

__global__ void GetInitialPhotoCostInOrder(
  RefViewSpace *view_ref, 
  ViewSpace *view_src,
  const int src_size,
  const PatchMatchOptions *options,
  const int height, 
  const int width
)
{
  
  for(int y = 0; y < height; ++y) {
    for(int x = 0; x < width; ++x) {
      // int center = y * width + x;
      // float depth = 1 / view_ref->plane_[center].idp_;
      // float normal[3] = {view_ref->plane_[center].nx_, view_ref->plane_[center].ny_, view_ref->plane_[center].nz_};
      // view_ref->cost_[center] = ComputePhotoConsistencyCost(
      //   view_ref->tex_, view_src->tex_,
      //   view_ref->params_->K_, view_src->params_->K_,
      //   view_ref->params_->R_, view_src->params_->R_,
      //   view_ref->params_->t_, view_src->params_->t_,
      //   x, y, depth, normal, options);
    }
  }
}

void MultiViewStereoMatcherCuda::Match(cv::Mat &depth, cv::Mat &normal)
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
  printf("here!\n");

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  RandomInit <<<grid_size, block_size>>>(ref_->cs_, ts.tv_nsec, height, width);
  cudaDeviceSynchronize();
  GenerateRandomState <<<grid_size, block_size>>>(ref_->plane_, ref_->cs_, options_, height, width, ref_->params_->K_, false);
  // RandomPlaneInOrder <<<1, 1>>> (ref_->plane_, ref_->cs_, options_, height, width, ref_->params_->K_, false);
  cudaDeviceSynchronize();

  CostInit <<<grid_size, block_size>>>(ref_, src_[0], image_size_, options_, height, width);
  // CostInitInOrder <<<1,1>>> (ref_, src_[0], image_size_, options_, height, width);
  // GetInitialPhotoCost <<<grid_size, block_size>>> (ref_, src_, image_size_, options_, height, width);
  // GetInitialPhotoCostInOrder <<<1, 1>>> (ref_, src_[0], image_size_, options_, height, width);
  cudaDeviceSynchronize();


  ShowDepthAndNormal(ref_->plane_, height, width);
  ShowCostAndHistogram(ref_->cost_, height, width);

  cudaDeviceSynchronize();

  for(int iter = 0; iter < 20; ++iter) {
    PropagateRed<<<grid_size, block_size>>>(ref_, src_[0], ref_->plane_, ref_->cost_, ref_->cs_, options_, height, width);
    cudaDeviceSynchronize();
    PropagateBlack<<<grid_size, block_size>>>(ref_, src_[0], ref_->plane_, ref_->cost_, ref_->cs_, options_, height, width);
    cudaDeviceSynchronize();
    ShowDepthAndNormal(ref_->plane_, height, width);
  }


  ShowDepthAndNormal(ref_->plane_, height, width);
  ShowCostAndHistogram(ref_->cost_, height, width);
  cudaDeviceSynchronize();

  // // check ncc result 
  // {
  //   float *cost_ncc;
  //   checkCudaErrors(cudaMallocManaged((void **)&cost_ncc, sizeof(float) * height * width));
    
  //   ComputePhotoConsistencyCostTest<<<grid_size, block_size>>>(ref_, src_[0], 1, options_, cost_ncc, height, width);
  //   cudaDeviceSynchronize();
  //   ShowCostAndHistogram(cost_ncc, height, width);

  //   checkCudaErrors(cudaFree(cost_ncc));
  // }







  depth = cv::Mat_<float>(height, width);
  normal = cv::Mat_<cv::Vec3f>(height, width);
  // ConvertPlaneToDepthAndNormal(ref_->plane_, depth, normal, height, width);
  for(int i = 0; i < height; ++i) {
    for(int j = 0; j < width; ++j) {
      depth.at<float>(i, j) = 1 / ref_->plane_[i*width+j].idp_;
      normal.at<cv::Vec3f>(i, j) = cv::Vec3f(ref_->plane_[i*width+j].nx_, ref_->plane_[i*width+j].ny_, ref_->plane_[i*width+j].nz_);
    }
  }

}

void TestHomographyWarpHost(const cv::Mat& K, const cv::Mat &R1, const cv::Mat &R2, const cv::Mat &t1, const cv::Mat &t2,
  int x, int y, float depth, float normal[3])
{
  // const float *K, 
  // const float *R1,
  // const float *R2,
  // const float *t1,
  // const float *t2,
  // int x, int y, 
  // float depth, 
  // float normal[3],
  // int height, int width

  float *K_, *R1_, *R2_, *t1_, *t2_, *norm;
  cudaMalloc(&K_, sizeof(float) * 9);
  cudaMalloc(&R1_, sizeof(float) * 9);
  cudaMalloc(&R2_, sizeof(float) * 9);
  cudaMalloc(&t1_, sizeof(float) * 3);
  cudaMalloc(&t2_, sizeof(float) * 3);
  cudaMalloc(&norm, sizeof(float) * 3);
  cudaMemcpy(K_, (float*)K.data, sizeof(float) * 9, cudaMemcpyHostToDevice);
  cudaMemcpy(R1_, (float*)R1.data, sizeof(float) * 9, cudaMemcpyHostToDevice);
  cudaMemcpy(R2_, (float*)R2.data, sizeof(float) * 9, cudaMemcpyHostToDevice);
  cudaMemcpy(t1_, (float*)t1.data, sizeof(float) * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(t2_, (float*)t2.data, sizeof(float) * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(norm, normal, sizeof(float) * 3, cudaMemcpyHostToDevice);
  TestHomographyWarp <<<1,1>>>(K_, R1_, R2_, t1_, t2_, x, y, depth, norm);
  cudaDeviceSynchronize();
}

// void TestComputNCCHost(
//   const cv::Mat &img0, const cv::Mat &img1, 
//   const cv::Mat& K, 
//   const cv::Mat &R1, const cv::Mat &R2, 
//   const cv::Mat &t1, const cv::Mat &t2,
//   int x, int y, float depth, float normal[3],
//   const PatchMatchOptions *options)
// {

//   float *K_, *R1_, *R2_, *t1_, *t2_, *norm;
//   cudaMalloc(&K_, sizeof(float) * 9);
//   cudaMalloc(&R1_, sizeof(float) * 9);
//   cudaMalloc(&R2_, sizeof(float) * 9);
//   cudaMalloc(&t1_, sizeof(float) * 3);
//   cudaMalloc(&t2_, sizeof(float) * 3);
//   cudaMalloc(&norm, sizeof(float) * 3);
//   cudaMemcpy(K_, (float*)K.data, sizeof(float) * 9, cudaMemcpyHostToDevice);
//   cudaMemcpy(R1_, (float*)R1.data, sizeof(float) * 9, cudaMemcpyHostToDevice);
//   cudaMemcpy(R2_, (float*)R2.data, sizeof(float) * 9, cudaMemcpyHostToDevice);
//   cudaMemcpy(t1_, (float*)t1.data, sizeof(float) * 3, cudaMemcpyHostToDevice);
//   cudaMemcpy(t2_, (float*)t2.data, sizeof(float) * 3, cudaMemcpyHostToDevice);
//   cudaMemcpy(norm, normal, sizeof(float) * 3, cudaMemcpyHostToDevice);

//   cudaTextureObject_t tex0, tex1;
//   cudaArray *arr0, *arr1;
//   CreateTextureObject(img0, tex0, arr0);
//   CreateTextureObject(img1, tex1, arr1);
//   printf("here\n");

//   ComputeNCCTest <<<1,1>>>(tex0, tex1, K_, K_, R1_, R2_, t1_, t2_, x, y, depth, norm, options);
//   cudaDeviceSynchronize();
// }

// test homography and warp!!
