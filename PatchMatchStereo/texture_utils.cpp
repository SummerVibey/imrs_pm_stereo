#include "texture_utils.h"

void CreateTextureObject(const cv::Mat &img, cudaTextureObject_t& tex, cudaArray *arr)
{

  assert(img.type() == CV_32F);
  int rows = img.rows;
  int cols = img.cols;

  // Create channel with floating point type
  cudaChannelFormatDesc channelDesc =
  cudaCreateChannelDesc (32,
                          0,
                          0,
                          0,
                          cudaChannelFormatKindFloat);
  // Allocate array with correct size and number of channels
  checkCudaErrors(cudaMallocArray(&arr,
                                  &channelDesc,
                                  cols,
                                  rows));

  checkCudaErrors(cudaMemcpy2DToArray (arr,
                                        0,
                                        0,
                                        img.ptr<float>(),
                                        img.step[0],
                                        cols*sizeof(float),
                                        rows,
                                        cudaMemcpyHostToDevice));

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType         = cudaResourceTypeArray;
  resDesc.res.array.array = arr;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeWrap;
  texDesc.addressMode[1]   = cudaAddressModeWrap;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  checkCudaErrors(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));
}

void DestroyTextureObject(cudaTextureObject_t& tex, cudaArray *arr)
{
  cudaFreeArray(arr);
  cudaDestroyTextureObject(tex);
}