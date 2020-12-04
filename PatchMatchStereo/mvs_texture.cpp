#include <iostream>
#include <chrono>
#include <stdio.h>
#include <dirent.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
// #include <opencv2/core/eigen.hpp>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigen>


#include "patch_match_cuda.h"
#include "device_utils.h"
#include "multi_view_stereo.h"
#include "mvs_matcher_wrapper.h"

std::vector<std::string> listDirectoryFiles(std::string& folder_path) 
{
  if(folder_path[folder_path.length()-1] != '/') {
    folder_path += "/";
  }
  std::vector<std::string> file_list; 
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (folder_path.c_str())) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
        continue;
      file_list.push_back(ent->d_name);      
    }
    closedir(dir);
  } else {
    /* could not open directory */
    std::runtime_error("unable to list files in the selected directory");
  }
  std::sort(file_list.begin(), file_list.end(), [&](const std::string& file1, const std::string& file2){ return file1 < file2; });
  return file_list;
}


void ReadMVGFiles(const std::string& filename, std::vector<cv::Mat_<float>> &Rcws, std::vector<cv::Mat_<float>> &tcws)
{
  std::fstream in;
  in.open(filename);
	if (!in.is_open()) {
		std::cout << "Can not find " << filename << std::endl;
		return ;
	}
	std::string buff;
  std::vector<cv::Mat_<float>> poses;
	int i = 0;
	while(getline(in, buff)) {
		std::vector<float> data;
		char *s_input = (char *)buff.c_str();
		const char *split = " ";
		char *p = strtok(s_input, split);
		float a;
		while (p != NULL) {
			a = atof(p);
			data.push_back(a);
			p = strtok(NULL, split);
    }
    Eigen::Matrix<float, 4, 4> Twc = Eigen::Matrix<float, 4, 4>::Identity();
    Twc.block(0, 0, 3, 4) = Eigen::Map<Eigen::Matrix<float, 3, 4, Eigen::RowMajor>>(&data.front());
    Eigen::Matrix<float, 4, 4> Tcw = Twc.inverse();
    cv::Mat_<float> Rcw(3, 3), tcw(3, 1); 
    for(int i = 0; i < 3; ++i) {
      tcw.ptr<float>(i)[0] = Tcw(i, 3);
      for(int j = 0; j < 3; ++j) {
        Rcw.ptr<float>(i)[j] = Tcw(i, j);
      }
    }
    Rcws.push_back(Rcw);
    tcws.push_back(tcw);
		i++;
	}
	in.close();
}

inline void Mat33DotVec3(const float mat[9], const float vec[3], float result[3]) 
{
  result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

inline void ComputeRelativePose(
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

void PrintMat3f(const float mat[9])
{
  for(int i = 0; i < 9; ++i) {
    std::cout << mat[i] << "  ";
    if(i % 3 == 2)
      std::cout << std::endl;
  }
}

void PrintVec3f(const float vec[3])
{
  std::cout << vec[0] << "  " << vec[1] << "  " << vec[2] << std::endl;
}

inline void ComputeHomography(
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
  float H[9])
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

inline void ComputeHomographyEigen(
  const Eigen::Matrix3f &K_ref, 
  const Eigen::Matrix3f &K_src, 
  const Eigen::Matrix4f &T21,
  const int x, 
  const int y, 
  const float depth, 
  const Eigen::Vector3f &normal,
  Eigen::Matrix3f& H)
{
  const float &ref_fx = K_ref(0,0), &ref_fy = K_ref(1,1),
              &ref_cx = K_ref(0,2), &ref_cy = K_ref(1,2);

  const float &src_fx = K_src(0,0), &src_fy = K_src(1,1),
              &src_cx = K_src(0,2), &src_cy = K_src(1,2);

  const float ref_inv_fx = 1 / ref_fx, ref_inv_cx = -ref_cx / ref_fx,
              ref_inv_fy = 1 / ref_fy, ref_inv_cy = -ref_cy / ref_fy;

  const float src_inv_fx = 1 / src_fx, src_inv_cx = -src_cx / src_fx,
              src_inv_fy = 1 / src_fy, src_inv_cy = -src_cy / src_fy;
  
  // const float pt3d[3] = {depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth};
  Eigen::Vector3f pt3d(depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth);

  // Distance to the plane.
  const float dist = -pt3d.dot(normal);
  Eigen::Matrix3f R21 = T21.block(0,0,3,3);
  Eigen::Vector3f t21 = T21.block(0,3,3,1);
  H = K_src * (R21 - t21 * normal.transpose() / dist) * K_ref.inverse();

}

inline void HomogeneousWarp(const float mat[9], const float vec[2], float result[2]) 
{
  const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
  result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
  result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
}

int main(int argc, char** argv)
{



  clock_t start,end;
  if(argc < 5) {
    std::cout << "Usage: ./MVSTest <img_folder> <calib_file> <poes_file> <depth_output_folder>..." << std::endl;
    return -1;
  } 
  std::string img_folder = argv[1];
  std::string calib_file = argv[2];
  std::string pose_file = argv[3];
  std::string depth_output_folder = argv[4];

  std::vector<std::string> img_files = listDirectoryFiles(img_folder);
  std::vector<cv::Mat_<float>> Rcws, tcws;
  ReadMVGFiles(pose_file, Rcws, tcws);


  // std::cout << R21be * R21be.inverse() << std::endl;
  // std::cout << T21.block(0,0,3,3) * T21.block(0,0,3,3).inverse() << std::endl;
  //   std::cout << tcws[i] << std::endl;
  // }

  cv::FileStorage calib(calib_file, cv::FileStorage::READ);
  cv::Mat camera_matrix = cv::Mat_<float>(3, 3);
  calib["camera_matrix"] >> camera_matrix;

  int width, height;
  calib["width"] >> width;
  calib["height"] >> height;

  std::cout << "K: " << std::endl << camera_matrix << std::endl;
  std::cout << "width: " << width << std::endl;
  std::cout << "height: " << height << std::endl;
  std::cout << "img size: " << img_files.size() << std::endl;
  std::cout << "pose size: " << Rcws.size() << std::endl;
  SelectCudaDevice();

  // test homography
  Eigen::Vector3f n(0.1, 0.2, -0.8); n.normalize();
  cv::Mat R1cv = Rcws[0].clone(), R2cv = Rcws[4].clone(), t1cv = tcws[0].clone(), t2cv = tcws[4].clone();
  float R1b[9], R2b[9], t1b[3], t2b[3], R21b[9], t21b[3], Kb[9];
  memcpy(R1b, (float*)R1cv.data, sizeof(float)*9);
  memcpy(R2b, (float*)R2cv.data, sizeof(float)*9);
  memcpy(t1b, (float*)t1cv.data, sizeof(float)*3);
  memcpy(t2b, (float*)t2cv.data, sizeof(float)*3);
  memcpy(Kb, (float*)camera_matrix.data, sizeof(float)*9);

  Eigen::Matrix4f T1, T2; T1.setIdentity(); T2.setIdentity();
  T1.block(0,0,3,3) = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R1b); 
  T1.block(0,3,3,1) = Eigen::Map<Eigen::Vector3f>(t1b);
  T2.block(0,0,3,3) = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(R2b); 
  T2.block(0,3,3,1) = Eigen::Map<Eigen::Vector3f>(t2b);
  Eigen::Matrix3f K = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(Kb);

  Eigen::Matrix4f T21 = T2*T1.inverse();
  std::cout << "Eigen result: " << std::endl << T21.block(0,0,3,3) << std::endl << T21.block(0,3,3,1) << std::endl;

  ComputeRelativePose(R1b, t1b, R2b, t2b, R21b, t21b);
  
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> R21be(R21b);
  Eigen::Map<Eigen::Vector3f> t21be(t21b);
  std::cout << "my result: " << std::endl << R21be << std::endl << t21be << std::endl;

  // K * (R21 - t21 * n' / d) * Kref^-1.
  float Hb[9];
  int x = 100, y = 150;
  float depth = 20.0f;
  float normal[3] = {n(0), n(1), n(2)};
  ComputeHomography(Kb, Kb, R1b, R2b, t1b, t2b, x, y, depth, normal, Hb);
  Eigen::Matrix3f H;
  ComputeHomographyEigen(K, K, T21, x, y, depth, n, H);
  Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> Hbe(Hb);
  std::cout << "eigen H: " << std::endl << H << std::endl;
  std::cout << "my H: " << std::endl << Hbe << std::endl;

  Eigen::Vector3f a(x, y, 1), b;
  float ar[2] = {x, y}, br[2];
  b = H*a; b /= b(2);
  std::cout << "eigen result: " << b.transpose() << std::endl;
  HomogeneousWarp(Hb, ar, br);
  std::cout << "my H: " << br[0] << "  " << br[1] << std::endl;
  // int ref_index = 6;
  // int src_size = 2;

  // PatchMatchOptions *options = new PatchMatchOptions();

  // MVSMatcherWrapper *mvs_matcher = new MVSMatcherWrapper(options);

  // for(int idx = ref_index - src_size/2; idx <= ref_index + src_size/2; ++idx) {
  //   if(idx < 0) continue;
  //   else if(idx > img_files.size() - 1) continue;
  //   if(idx != ref_index) {
  //     cv::Mat img = cv::imread(img_folder + "/" + img_files[idx], cv::IMREAD_GRAYSCALE);
  //     img.convertTo(img, CV_32F);
  //     printf("Prepare to add source view!\n");
  //     mvs_matcher->AddSourceView(img, camera_matrix, Rcws[idx], tcws[idx]);
  //     printf("Have added source view!\n");
  //   }
  //   else {
  //     cv::Mat img = cv::imread(img_folder + "/" + img_files[idx], cv::IMREAD_GRAYSCALE);
  //     img.convertTo(img, CV_32F);
  //     printf("Prepare to add reference view!\n");
  //     mvs_matcher->SetReferenceView(img, camera_matrix, Rcws[idx], tcws[idx]);
  //     printf("Have added reference view!\n");
  //   }
  // }

  // mvs_matcher->Initialize();
  // mvs_matcher->Run();

  // delete mvs_matcher;
  // delete options;

}