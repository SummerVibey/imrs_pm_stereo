#ifndef PATCH_MATCH_COST_H
#define PATCH_MATCH_COST_H

#include "data_types.h"

#define COST_PUNISH 120.0f

// inline Vector3f GetColor(const unsigned int* img_data, const float& x, const int& y, int width)
// {
//   float col[3];
//   const auto x1 = static_cast<int>(x);
//   const int x2 = x1 + 1;
//   const float ofs = x - x1;

//   for (int n = 0; n < 3; n++) {
//     const auto& g1 = img_data[y * width * 3 + 3 * x1 + n];
//     const auto& g2 = (x2 < width) ? img_data[y * width * 3 + 3 * x2 + n] : g1;
//     col[n] = (1 - ofs) * g1 + ofs * g2;
//   }

//   return { col[0], col[1], col[2] };
// }


// inline Vector2f GetGradient(const Gradient* grad_data, const float& x, const int& y, int width) 
// {
//   const auto x1 = static_cast<int>(x);
//   const int x2 = x1 + 1;
//   const float ofs = x - x1;

//   const auto& g1 = grad_data[y * width + x1];
//   const auto& g2 = (x2 < width) ? grad_data[y * width + x2] : g1;

//   return { (1 - ofs) * g1.x + ofs * g2.x,(1 - ofs) * g1.y + ofs * g2.y };
// }

// inline float ComputeCostPoint(
//   const unsigned int *img_left, const unsigned int *img_right, const Gradient *grad_left, const Gradient *grad_right,
//   const int& x, const int& y, const float& d, const PatchMatchOptions *options, const int& width)
// {
//   const float xr = x - d;
//   if (xr < 0.0f || xr >= static_cast<float>(width)) {
//     return (1 - options->alpha) * options->tau_col + options->alpha * options->tau_grad;
//   }
//   // ��ɫ�ռ����
//   const Vector3f col_p = GetColor(img_left, x, y, width);
//   const Vector3f col_q = GetColor(img_right, xr, y, width);
//   const auto dc = std::min(fabs(col_p.x - col_q.x) + fabs(col_p.y - col_q.y) + fabs(col_p.z - col_q.z), options->tau_col);

//   // �ݶȿռ����
//   const Vector2f grad_p = GetGradient(grad_left, x, y, width);
//   const Vector2f grad_q = GetGradient(grad_right, xr, y, width);
//   const auto dg = std::min(fabs(grad_p.x - grad_q.x)+ fabs(grad_p.y - grad_q.y), options->tau_grad);

//   return (1 - options->alpha) * dc + options->alpha * dg;
// }

 

// float ComputeCostRegion(
//   const unsigned int *img_left, const unsigned int *img_right, const Gradient *grad_left, const Gradient *grad_right,
//   const int& x, const int& y, const DispPlane *p, const PatchMatchOptions *options, int height, int width) 
// {
//   const auto pat = options->patch_size / 2;
//   const Vector3f col_p = GetColor(img_left, x, y, width);
//   float cost = 0.0f;
//   for (int r = -pat; r <= pat; r++) {
//     const int yr = y + r;
//     for (int c = -pat; c <= pat; c++) {
//       const int xc = x + c;
//       if (yr < 0 || yr > height - 1 || xc < 0 || xc > width - 1) {
//         continue;
//       }
//       // �����Ӳ�ֵ
//       const float d = p->to_disparity(xc,yr);
//       if (d < options->min_disparity || d > options->max_disparity) {
//         cost += COST_PUNISH;
//         continue;
//       }

//       // ����Ȩֵ
//       const auto& col_q = GetColor(img_left, xc, yr, width);
//       const auto dc = abs(col_p.z - col_q.z) + abs(col_p.y - col_q.y) + abs(col_p.x - col_q.x);
//       const auto w = exp(-dc / options->gamma);

//       // �ۺϴ���
//       const Vector2f grad_q = GetGradient(grad_left, xc, yr, width);
//       cost += w * ComputeCostPoint(img_left, img_right, grad_left, grad_right, xc, yr, d, options, width);
//     }
//   }
//   return cost;
// }

#endif