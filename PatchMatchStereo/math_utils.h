#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#define pow2(x) ((x)*(x))
#define pow3(x) ((x)*(x)*(x))
#define pow4(x) ((x)*(x)*(x)*(x))
#define norm2_pow2(x,y) (pow2(x)+pow2(y))
#define norm2(x,y) (sqrtf(norm2_pow2)))
#define norm3_pow2(x,y,z) (pow2(x)+pow2(y)+pow2(z))
#define norm3f(x,y,z) (sqrtf(norm3_pow2(x,y,z)))

// #define abs(x) x>=0 ? x : -x

#define imgatf(img, i, j) \
  img.ptr<float>(i)[j]

#define texat(tex,x,y) \
 tex2D<float>(tex, x+0.5f, y+0.5f)

#define color(tex,x,y) \
  texat(tex,x,y);

#define gradx(tex,x,y) \
  ((-texat(tex, x-1, y-1) + texat(tex, x+1, y-1)) + \
  (-2 * texat(tex, x-1, y) + 2 * texat(tex, x+1, y)) + \
  (-texat(tex, x-1, y+1) + texat(tex, x+1, y+1))) / 8.0f

#define grady(tex,x,y) \
  ((-texat(tex, x-1, y-1) - 2 * texat(tex, x, y-1) - texat(tex, x+1, y-1)) + \
  (texat(tex, x-1, y+1) + 2 * texat(tex, x, y+1) + texat(tex, x+1, y+1))) / 8.0f

#endif