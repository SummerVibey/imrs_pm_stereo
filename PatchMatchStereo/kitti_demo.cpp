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


int main(int argc, char** argv)
{

  clock_t start,end;
  if(argc < 5) {
    std::cout << "Usage: ./mvs_test <img_left_folder> <img_right_folder> <calib_file> <poes_file>..." << std::endl;
    return -1;
  } 
  std::string img_left_folder = argv[1];
  std::string img_right_folder = argv[2];
  std::string calib_file = argv[3];
  std::string pose_file = argv[4];

  std::vector<std::string> img_left_files = listDirectoryFiles(img_left_folder);
  std::vector<std::string> img_right_files = listDirectoryFiles(img_right_folder);

  std::vector<cv::Mat_<float>> Rcws, tcws, tcw2s;
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
  cv::Mat R10 = cv::Mat::eye(3, 3, CV_32F);
  cv::Mat t10 = cv::Mat::zeros(3, 1, CV_32F);
  t10.at<float>(0) = -bf / K.at<float>(0,0);
  tcw2s.resize(tcws.size());
  for(int i = 0; i < tcws.size(); ++i) {
    tcw2s[i] = tcws[i] + t10;
  }
  SelectCudaDevice();


  PatchMatchOptions *options = new PatchMatchOptions(0.1f, 50.0f);
  options->sigma_color = 10.0f;
  options->sigma_spatial = 15.0f;
  options->patch_size = 35;
  options->step_size = 1;
  MVSMatcherWrapper *mvs_matcher = new MVSMatcherWrapper(options, height, width);


  const int ref_idx = 2;
  for(int i = 0; i < img_left_files.size(); i++) {
    if(i == ref_idx) {
      cv::Mat img_left = cv::imread(img_left_folder + "/" + img_left_files[i], cv::IMREAD_GRAYSCALE);
      cv::Mat img_right = cv::imread(img_right_folder + "/" + img_right_files[i], cv::IMREAD_GRAYSCALE);

      img_left.convertTo(img_left, CV_32F);
      img_right.convertTo(img_right, CV_32F);
      printf("Prepare to add reference view!\n");
      mvs_matcher->SetReferenceView(img_left, K, Rcws[i], tcws[i]);
      mvs_matcher->AddSourceView(img_right, K, Rcws[i], tcw2s[i]);
    }
    else if(i == 1) {
      cv::Mat img_left = cv::imread(img_left_folder + "/" + img_left_files[i], cv::IMREAD_GRAYSCALE);
      cv::Mat img_right = cv::imread(img_right_folder + "/" + img_right_files[i], cv::IMREAD_GRAYSCALE);

      img_left.convertTo(img_left, CV_32F);
      img_right.convertTo(img_right, CV_32F);
      printf("Prepare to add source view!\n");
      // mvs_matcher->AddSourceView(img_left, K, Rcws[i], tcws[i]);
      mvs_matcher->AddSourceView(img_right, K, Rcws[i], tcw2s[i]);
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

  delete mvs_matcher;
  delete options;

}