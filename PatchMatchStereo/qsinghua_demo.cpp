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

std::vector<cv::Mat> ReadProjectMat(const std::string& filename)
{

  std::fstream in;
  in.open(filename);
	if (!in.is_open()) {
		std::cout << "Can not find " << filename << std::endl;
		return std::vector<cv::Mat>();
	}
	std::string buff;
  std::vector<cv::Mat> project_mats;
	int i = 0;

	while(getline(in, buff)) {
		std::vector<float> nums;
		char *s_input = (char *)buff.c_str();
		const char *split = " ";
		char *p = strtok(s_input, split);
		float a;
		while (p != NULL) {
			a = atof(p);
			nums.push_back(a);
			p = strtok(NULL, split);
    }
    cv::Mat project_mat(3, 4, CV_32F);
    memcpy((float*)project_mat.data, &nums.front(), sizeof(float) * 12);
    project_mats.push_back(project_mat);
		i++;
	}
	in.close();
  return project_mats;
}


int main(int argc, char** argv)
{

  clock_t start,end;
  if(argc < 3) {
    std::cout << "Usage: ./mvs_test <img_folder> <calib_file>..." << std::endl;
    return -1;
  } 
  std::string img_folder = argv[1];
  std::string calib_file = argv[2];

  std::vector<std::string> img_files = listDirectoryFiles(img_folder);
  std::vector<cv::Mat> Ps = ReadProjectMat(calib_file);
  assert(img_files.size() == Ps.size());
  std::vector<cv::Mat> Ks(Ps.size()), Rs(Ps.size()), Ts(Ps.size()), Cs(Ps.size()), ts(Ps.size());
  for(int i = 0; i < Ps.size(); ++i) {
    cv::decomposeProjectionMatrix(Ps[i], Ks[i], Rs[i], Ts[i]);
    Cs[i] = Ts[i].rowRange(0, 3) / Ts[i].at<float>(3, 0);
    ts[i] = -Rs[i] * Cs[i];

    std::cout << "View " << i << std::endl;
    std::cout << "P: " << std::endl << Ps[i] << std::endl;
    std::cout << "K: " << std::endl << Ks[i] << std::endl;
    std::cout << "Rcw: " << std::endl << Rs[i] << std::endl;
    std::cout << "tcw: " << std::endl << ts[i] << std::endl;
    std::cout << std::endl;
  }

  SelectCudaDevice();

  const int ref_idx = 3;
  PatchMatchOptions *options = new PatchMatchOptions(0.1f, 100.0f);
  options->sigma_color = 10.0f;
  options->sigma_spatial = 15.0f;
  options->patch_size = 35;
  options->step_size = 2;
  MVSMatcherWrapper *mvs_matcher = new MVSMatcherWrapper(options, 728, 1092);

  for(int i = 0; i < img_files.size(); ++i) {
    if(i == ref_idx) {
      cv::Mat imgref = cv::imread(img_folder + "/" + img_files[i], cv::IMREAD_GRAYSCALE);
      cv::pyrDown(imgref, imgref);
      cv::pyrDown(imgref, imgref);
      imgref.convertTo(imgref, CV_32F);
      Ks[i] /= 4.0f;
      Ks[i].at<float>(2, 2) = 1.0f;
      printf("Prepare to add reference view!\n");
      mvs_matcher->SetReferenceView(imgref, Ks[i], Rs[i], ts[i]);
    }
    else {
      cv::Mat imgsrc = cv::imread(img_folder + "/" + img_files[i], cv::IMREAD_GRAYSCALE);
      cv::pyrDown(imgsrc, imgsrc);
      cv::pyrDown(imgsrc, imgsrc);
      imgsrc.convertTo(imgsrc, CV_32F);
      Ks[i] /= 4.0f;
      Ks[i].at<float>(2, 2) = 1.0f;
      printf("Prepare to add source view!\n");
      mvs_matcher->AddSourceView(imgsrc, Ks[i], Rs[i], ts[i]);
    }
  }

  mvs_matcher->Initialize();
  mvs_matcher->RunDebug();

  delete mvs_matcher;
  delete options;

}