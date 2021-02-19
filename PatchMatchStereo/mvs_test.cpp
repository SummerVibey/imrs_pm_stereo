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

std::vector<Eigen::Matrix<double, 4, 4>> readPoseGroundTruth(const std::string& filename)
{
  std::cout << "here" << std::endl;

  std::fstream in;
  in.open(filename);
	if (!in.is_open()) {
		std::cout << "Can not find " << filename << std::endl;
		return std::vector<Eigen::Matrix<double, 4, 4>>();
	}
  std::cout << "here" << std::endl;
	std::string buff;
  std::vector<Eigen::Matrix<double, 4, 4>> poses;
	int i = 0;
  std::cout << "here" << std::endl;
	while(getline(in, buff)) {
		std::vector<double> nums;
		char *s_input = (char *)buff.c_str();
		const char *split = " ";
		char *p = strtok(s_input, split);
		double a;
		while (p != NULL) {
			a = atof(p);
			nums.push_back(a);
			p = strtok(NULL, split);
    }
    std::cout << "here0" << std::endl;
    Eigen::Matrix<double, 4, 4> Twc = Eigen::Matrix<double, 4, 4>::Identity();
    Twc.block(0, 0, 3, 4) = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(&nums.front());
    poses.push_back(Twc);
		i++;
	}
	in.close();
  return poses;
}

void ComputeRelativePose(
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

 void GetHomography(
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
  
  const float point3d[3] = {depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth};

  // Distance to the plane.
  const float dist = normal[0] * point3d[0] + normal[1] * point3d[1] + normal[2] * point3d[2];
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

int main(int argc, char** argv)
{

  clock_t start,end;
  if(argc < 4) {
    std::cout << "Usage: ./mvs_test <img_folder> <calib_file> <poes_file>..." << std::endl;
    return -1;
  } 
  std::string img_folder = argv[1];
  std::string calib_file = argv[2];
  std::string pose_file = argv[3];

  std::vector<std::string> img_files = listDirectoryFiles(img_folder);
  // std::vector<Eigen::Matrix4d> poses = readPoseGroundTruth(pose_file);
  std::vector<cv::Mat_<float>> Rcws, tcws;
  // std::vector<cv::Mat> Rcws, tcws;
  // for(int i = 0; i < (int)poses.size(); ++i) {
  //   Eigen::Matrix4d Tcw = poses[i].inverse();
  //   Eigen::Matrix3f Rcwe = Tcw.block<3, 3>(0, 0).cast<float>();
  //   Eigen::Vector3f tcwe = Tcw.block<3, 1>(0, 3).cast<float>();
  //   cv::Mat Rcw = (cv::Mat_<float>(3, 3) << Rcwe(0, 0), Rcwe(0, 1), Rcwe(0, 2), 
  //                                           Rcwe(1, 0), Rcwe(1, 1), Rcwe(1, 2), 
  //                                           Rcwe(2, 0), Rcwe(2, 1), Rcwe(2, 2));

  //   cv::Mat tcw = (cv::Mat_<float>(3, 1) << tcwe(0),tcwe(1),tcwe(2));
    
  //   Rcws.push_back(Rcw);
  //   tcws.push_back(tcw);
  // }
  ReadMVGFiles(pose_file, Rcws, tcws);

  cv::FileStorage calib(calib_file, cv::FileStorage::READ);
  cv::Mat P0 = cv::Mat_<float>(3, 4), P1 = cv::Mat_<float>(3, 4);
  calib["P0"] >> P0;
  calib["P1"] >> P1;
  int width, height;
  float bf;
  calib["width"] >> width;
  calib["height"] >> height;
  calib["bf"] >> bf;
  cv::Mat K = P0.rowRange(0, 3).colRange(0, 3).clone();
  SelectCudaDevice();


  PatchMatchOptions *options = new PatchMatchOptions(0.1f, 50.0f);
  MVSMatcherWrapper *mvs_matcher = new MVSMatcherWrapper(options, height, width);

  const int idx_offset = 740;
  const int ref_idx = 742;
  for(int i = idx_offset; i < idx_offset + 5; ++i) {
    if(i == ref_idx) {
      cv::Mat imgref = cv::imread(img_folder + "/" + img_files[i], cv::IMREAD_GRAYSCALE);
      imgref.convertTo(imgref, CV_32F);
      printf("Prepare to add reference view!\n");
      mvs_matcher->SetReferenceView(imgref, K, Rcws[i], tcws[i]);
    }
    else {
      cv::Mat imgsrc = cv::imread(img_folder + "/" + img_files[i], cv::IMREAD_GRAYSCALE);
      imgsrc.convertTo(imgsrc, CV_32F);
      printf("Prepare to add source view!\n");
      mvs_matcher->AddSourceView(imgsrc, K, Rcws[i], tcws[i]);
    }
  }

  // cv::Mat imgref = cv::imread(img_folder + "/" + img_files[ref_idx], cv::IMREAD_GRAYSCALE);
  // imgref.convertTo(imgref, CV_32F);
  // mvs_matcher->SetReferenceView(imgref, K, Rcws[ref_idx], tcws[ref_idx]);
  // cv::Mat imgsrc = cv::imread(img_folder + "/" + img_files[ref_idx+1], cv::IMREAD_GRAYSCALE);
  // imgsrc.convertTo(imgsrc, CV_32F);
  // mvs_matcher->AddSourceView(imgsrc, K, Rcws[ref_idx+1], tcws[ref_idx+1]);


  mvs_matcher->Initialize();
  mvs_matcher->RunDebug();

  Eigen::Vector3f normal;
  // normal << 0.1f, -0.2f, -1.0f;
  normal << 0.0f, 0.0f, -1.0f;
  normal.normalize();
  float norm[3] = {normal(0), normal(1), normal(2)};


  cv::Mat H(3, 3, CV_32F, cv::Scalar(0));
  GetHomography((float*)K.data,
                 (float*)K.data, 
                (float*)Rcws[ref_idx].data,
                (float*)Rcws[ref_idx].data,
                (float*)tcws[ref_idx].data,
                (float*)tcws[ref_idx].data,
                100, 200, 10.0f, norm, (float*)H.data);
  std::cout << H << std::endl;

  delete mvs_matcher;
  delete options;

}