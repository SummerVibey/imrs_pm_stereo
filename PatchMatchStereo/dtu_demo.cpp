// #include <iostream>
// #include <chrono>
// #include <stdio.h>
// #include <dirent.h>
// #include <cstring>
// #include <sstream>
// #include <iomanip>
// #include <time.h>
// #include <stdio.h>

// #include <opencv2/core/core.hpp>
// // #include <opencv2/core/eigen.hpp>

// #include <eigen3/Eigen/Core>
// #include <eigen3/Eigen/Geometry>
// #include <eigen3/Eigen/Dense>
// #include <eigen3/Eigen/Eigen>


// #include "patch_match_cuda.h"
// #include "device_utils.h"
// #include "multi_view_stereo.h"
// #include "mvs_matcher_wrapper.h"

// std::vector<std::string> listDirectoryFiles(std::string& folder_path) 
// {
//   if(folder_path[folder_path.length()-1] != '/') {
//     folder_path += "/";
//   }
//   std::vector<std::string> file_list; 
//   DIR *dir;
//   struct dirent *ent;
//   if ((dir = opendir (folder_path.c_str())) != NULL) {
//     /* print all the files and directories within directory */
//     while ((ent = readdir (dir)) != NULL) {
//       if(!strcmp(ent->d_name, ".") || !strcmp(ent->d_name, ".."))
//         continue;
//       file_list.push_back(ent->d_name);      
//     }
//     closedir(dir);
//   } else {
//     /* could not open directory */
//     std::runtime_error("unable to list files in the selected directory");
//   }
//   std::sort(file_list.begin(), file_list.end(), [&](const std::string& file1, const std::string& file2){ return file1 < file2; });
//   return file_list;
// }


// void ReadProjectionMat(const std::string& filename, std::vector<cv::Mat_<float>> &P)
// {
//   std::fstream in;
//   in.open(filename);
// 	if (!in.is_open()) {
// 		std::cout << "Can not find " << filename << std::endl;
// 		return ;
// 	}
// 	std::string buff;
//   std::vector<cv::Mat_<float>> poses;
// 	int i = 0;
// 	while(getline(in, buff)) {
// 		std::vector<float> data;
// 		char *s_input = (char *)buff.c_str();
// 		const char *split = " ";
// 		char *p = strtok(s_input, split);
// 		float a;
// 		while (p != NULL) {
// 			a = atof(p);
// 			data.push_back(a);
// 			p = strtok(NULL, split);
//     }
//     cv::Mat project_mat(3, 4, CV_32F);
//     memcpy((float*)project_mat.data, &data.front(), sizeof(float) * 12);
//     P.push_back(project_mat);
// 		i++;
// 	}
// 	in.close();
// }

// inline void Mat33DotVec3(const float mat[9], const float vec[3], float result[3]) 
// {
//   result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
//   result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
//   result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
// }

// inline void ComputeRelativePose(
//   const float R1[9], const float t1[3], // ref pose
//   const float R2[9], const float t2[3], // src pose 
//   float R21[9], float t21[3])           // relative pose ref to src 
// {
//   R21[0] = R2[0] * R1[0] + R2[1] * R1[1] + R2[2] * R1[2];
//   R21[1] = R2[0] * R1[3] + R2[1] * R1[4] + R2[2] * R1[5];
//   R21[2] = R2[0] * R1[6] + R2[1] * R1[7] + R2[2] * R1[8];
//   R21[3] = R2[3] * R1[0] + R2[4] * R1[1] + R2[5] * R1[2];
//   R21[4] = R2[3] * R1[3] + R2[4] * R1[4] + R2[5] * R1[5];
//   R21[5] = R2[3] * R1[6] + R2[4] * R1[7] + R2[5] * R1[8];
//   R21[6] = R2[6] * R1[0] + R2[7] * R1[1] + R2[8] * R1[2];
//   R21[7] = R2[6] * R1[3] + R2[7] * R1[4] + R2[8] * R1[5];
//   R21[8] = R2[6] * R1[6] + R2[7] * R1[7] + R2[8] * R1[8];

//   t21[0] = - R21[0] * t1[0] - R21[1] * t1[1] - R21[2] * t1[2] + t2[0];
//   t21[1] = - R21[3] * t1[0] - R21[4] * t1[1] - R21[5] * t1[2] + t2[1];
//   t21[2] = - R21[6] * t1[0] - R21[7] * t1[1] - R21[8] * t1[2] + t2[2];
// }

// void PrintMat3f(const float mat[9])
// {
//   for(int i = 0; i < 9; ++i) {
//     std::cout << mat[i] << "  ";
//     if(i % 3 == 2)
//       std::cout << std::endl;
//   }
// }

// void PrintVec3f(const float vec[3])
// {
//   std::cout << vec[0] << "  " << vec[1] << "  " << vec[2] << std::endl;
// }

// inline void ComputeHomography(
//   const float *K_ref, 
//   const float *K_src, 
//   const float *R1,
//   const float *R2,
//   const float *t1,
//   const float *t2,
//   const int x, 
//   const int y, 
//   const float depth, 
//   const float normal[3],
//   float H[9])
// {
//   const float &ref_fx = K_ref[0], &ref_fy = K_ref[4],
//               &ref_cx = K_ref[2], &ref_cy = K_ref[5];

//   const float &src_fx = K_src[0], &src_fy = K_src[4],
//               &src_cx = K_src[2], &src_cy = K_src[5];

//   const float ref_inv_fx = 1 / ref_fx, ref_inv_cx = -ref_cx / ref_fx,
//               ref_inv_fy = 1 / ref_fy, ref_inv_cy = -ref_cy / ref_fy;
  
//   const float pt3d[3] = {depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth};

//   // Distance to the plane.
//   const float dist = normal[0] * pt3d[0] + normal[1] * pt3d[1] + normal[2] * pt3d[2];
//   const float inv_dist = 1.0f / dist;
//   const float inv_dist_N0 = inv_dist * normal[0];
//   const float inv_dist_N1 = inv_dist * normal[1];
//   const float inv_dist_N2 = inv_dist * normal[2];

//   // Relative Pose from ref to src
//   float R21[9], t21[3];
//   ComputeRelativePose(R1, t1, R2, t2, R21, t21);

//   H[0] = ref_inv_fx * (src_fx * (R21[0] + inv_dist_N0 * t21[0]) +
//                          src_cx * (R21[6] + inv_dist_N0 * t21[2]));
//   H[1] = ref_inv_fy * (src_fx * (R21[1] + inv_dist_N1 * t21[0]) +
//                          src_cx * (R21[7] + inv_dist_N1 * t21[2]));
//   H[2] = src_fx * (R21[2] + inv_dist_N2 * t21[0]) +
//          src_cx * (R21[8] + inv_dist_N2 * t21[2]) +
//          ref_inv_cx * (src_fx * (R21[0] + inv_dist_N0 * t21[0]) +
//                          src_cx * (R21[6] + inv_dist_N0 * t21[2])) +
//          ref_inv_cy * (src_fx * (R21[1] + inv_dist_N1 * t21[0]) +
//                          src_cx * (R21[7] + inv_dist_N1 * t21[2]));
//   H[3] = ref_inv_fx * (src_fy * (R21[3] + inv_dist_N0 * t21[1]) +
//                          src_cy * (R21[6] + inv_dist_N0 * t21[2]));
//   H[4] = ref_inv_fy * (src_fy * (R21[4] + inv_dist_N1 * t21[1]) +
//                          src_cy * (R21[7] + inv_dist_N1 * t21[2]));
//   H[5] = src_fy * (R21[5] + inv_dist_N2 * t21[1]) +
//          src_cy * (R21[8] + inv_dist_N2 * t21[2]) +
//          ref_inv_cx * (src_fy * (R21[3] + inv_dist_N0 * t21[1]) +
//                          src_cy * (R21[6] + inv_dist_N0 * t21[2])) +
//          ref_inv_cy * (src_fy * (R21[4] + inv_dist_N1 * t21[1]) +
//                          src_cy * (R21[7] + inv_dist_N1 * t21[2]));
//   H[6] = ref_inv_fx * (R21[6] + inv_dist_N0 * t21[2]);
//   H[7] = ref_inv_fy * (R21[7] + inv_dist_N1 * t21[2]);
//   H[8] = R21[8] + ref_inv_cx * (R21[6] + inv_dist_N0 * t21[2]) +
//          ref_inv_cy * (R21[7] + inv_dist_N1 * t21[2]) + inv_dist_N2 * t21[2];

// }

// inline void ComputeHomographyEigen(
//   const Eigen::Matrix3f &K_ref, 
//   const Eigen::Matrix3f &K_src, 
//   const Eigen::Matrix4f &T21,
//   const int x, 
//   const int y, 
//   const float depth, 
//   const Eigen::Vector3f &normal,
//   Eigen::Matrix3f& H)
// {
//   const float &ref_fx = K_ref(0,0), &ref_fy = K_ref(1,1),
//               &ref_cx = K_ref(0,2), &ref_cy = K_ref(1,2);

//   const float ref_inv_fx = 1 / ref_fx, ref_inv_cx = -ref_cx / ref_fx,
//               ref_inv_fy = 1 / ref_fy, ref_inv_cy = -ref_cy / ref_fy;

//   // const float pt3d[3] = {depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth};
//   Eigen::Vector3f pt3d(depth * (ref_inv_fx * x + ref_inv_cx), depth * (ref_inv_fy * y + ref_inv_cy), depth);

//   // Distance to the plane.
//   const float dist = -pt3d.dot(normal);
//   Eigen::Matrix3f R21 = T21.block(0,0,3,3);
//   Eigen::Vector3f t21 = T21.block(0,3,3,1);
//   H = K_src * (R21 - t21 * normal.transpose() / dist) * K_ref.inverse();

// }

// inline void HomogeneousWarp(const float mat[9], const float px, const float py, float &qx, float &qy) 
// {
//   const float inv_z = 1.0f / (mat[6] * px + mat[7] * py + mat[8]);
//   qx = inv_z * (mat[0] * px + mat[1] * py + mat[2]);
//   qy = inv_z * (mat[3] * px + mat[4] * py + mat[5]);
// }

// int main(int argc, char** argv)
// {

//   clock_t start,end;
//   if(argc < 3) {
//     std::cout << "Usage: ./MVSTest <img_folder> <poes_file> ..." << std::endl;
//     return -1;
//   } 
//   std::string img_folder = argv[1];
//   std::string pose_file = argv[2];

//   std::vector<std::string> img_files = listDirectoryFiles(img_folder);
//   std::vector<cv::Mat_<float>> project_mats;
//   printf("read pro!\n");
//   ReadProjectionMat(pose_file, project_mats);
//   printf("read pro complete!\n");


//   // std::cout << R21be * R21be.inverse() << std::endl;
//   // std::cout << T21.block(0,0,3,3) * T21.block(0,0,3,3).inverse() << std::endl;
//   //   std::cout << tcws[i] << std::endl;
//   // }

//   SelectCudaDevice();


//   PatchMatchOptions *options = new PatchMatchOptions();

//   MVSMatcherWrapper *mvs_matcher = new MVSMatcherWrapper(optionsm, height);

//   cv::Mat imgref = cv::imread(img_folder + "/" + img_files[0], cv::IMREAD_GRAYSCALE);
//   imgref.convertTo(imgref, CV_32F);
//   mvs_matcher->SetReferenceView(imgref, project_mats[0]);

//   cv::Mat imgsrc = cv::imread(img_folder + "/" + img_files[1], cv::IMREAD_GRAYSCALE);
//   imgsrc.convertTo(imgsrc, CV_32F);
//   mvs_matcher->AddSourceView(imgsrc, project_mats[1]);

//   mvs_matcher->InitializeP();

//   int xc = 600, yc = 600;
//   float depth_c = 20000.0f;
//   Eigen::Vector3f norm_center(0, 0, -1); norm_center.normalize();
//   cv::Mat img_color1 = cv::imread(img_folder + "/" + img_files[0], cv::IMREAD_COLOR);
//   cv::Mat img_color2 = cv::imread(img_folder + "/" + img_files[1], cv::IMREAD_COLOR);

//   cv::Mat K1, R1, t1, t14;
//   cv::decomposeProjectionMatrix(project_mats[0], K1, R1, t14);
//   t1 = t14.rowRange(0,3).colRange(0,1) / t14.at<float>(3,0);

//   cv::Mat K2, R2, t2, t24;
//   cv::decomposeProjectionMatrix(project_mats[1], K2, R2, t24);
//   t2 = t24.rowRange(0,3).colRange(0,1) / t24.at<float>(3,0);

//   cv::Mat R21 = R2 * R1.t();
//   cv::Mat t21 = -R21 * t1 + t2;
//   cv::Mat pi = cv::Mat(3, 1, CV_32F), norm = cv::Mat(3, 1, CV_32F);

//   pi.at<float>(0) = xc, pi.at<float>(1) = yc, pi.at<float>(2) = 1.0f;
//   norm.at<float>(0) = norm_center(0), norm.at<float>(1) = norm_center(1), norm.at<float>(2) = norm_center(2);
//   norm = R1.inv() * norm;
//   cv::Mat pc = K1.inv() * pi * depth_c;
//   float dist = pc.at<float>(0) * norm.at<float>(0) + pc.at<float>(1) * norm.at<float>(1) + pc.at<float>(2) * norm.at<float>(2);

//   cv::Mat H = K2*(R21 + t21*norm.t()/dist) * K1.inv();
//   std::cout << pi.t() << std::endl;
//   std::cout << pc.t() << std::endl;
//   std::cout << norm.t() << std::endl;
//   std::cout << depth_c << std::endl;
//   std::cout << dist << std::endl;

//   Eigen::Matrix3f Ke1;
//   Ke1 << 2892.3306, -0.00024652568, 823.20532,
//         0, 2883.1753, 619.07092,
//         0, 0, 1;
  
//   Eigen::Matrix3f Ke2;
//   Ke2 << 2892.3308, 0.00059499487, 823.20459,
//         0, 2883.1758, 619.06879,
//         0, 0, 1.0000002;

//   Eigen::Matrix3f Re1;
//   Re1 << 0.97026271, 0.0074799112, 0.24193878,
//       -0.014742871, 0.99949294, 0.028223416,
//       -0.24160498, -0.030950999, 0.96988094;

//   Eigen::Matrix3f Re2;
//   Re2 << 0.80225545, -0.43934759, 0.40417805,
//           0.427993, 0.89528227, 0.12365936,
//           -0.41618288, 0.073778979, 0.90628278;

//   Eigen::Vector3f te1;
//   te1 << 190.83347,
//         -1.1601868,
//         24.261007;

//   Eigen::Vector3f te2;
//   te2 << 296.43283,
//         -64.311806,
//         62.716511;

//   Eigen::Matrix3f Re21 = Re2 * Re1.transpose();
//   Eigen::Vector3f te21 = -Re21 * te1 + te2;
//   Eigen::Matrix4f Te21 = Eigen::Matrix4f::Identity();
//   Te21.block(0,0,3,3) = Re21;
//   Te21.block(0,3,3,1) = te21;
//   Eigen::Matrix3f He;
//   ComputeHomographyEigen(Ke1, Ke2, Te21, xc, yc, depth_c, norm_center, He);
//   std::cout << "eigne H: " << std::endl << He << std::endl;

//   // for(int i = -15; i < 15; ++i) {
//   //   for(int j = -15; j < 15; ++j) {
//   //     float qx = (float)(xc + i*3), qy = (float)(yc + j*3);
//   //     cv::Mat qref = cv::Mat_<float>(3, 1);
//   //     qref.at<float>(0) = qx;
//   //     qref.at<float>(1) = qy;
//   //     qref.at<float>(2) = 1.0f;
//   //     cv::Mat qsrc = H * qref;
//   //     qsrc /= qsrc.at<float>(2);
//   //     float qsx = qsrc.at<float>(0), qsy = qsrc.at<float>(1);
//   //     cv::Point2f pref(qx, qy), psrc(qsx, qsy);
//   //     std::cout << qref.t() << std::endl;
//   //     std::cout << qsrc.t() << std::endl;
//   //     cv::circle(img_color1, pref, 0, cv::Scalar(0,255,0), -1);
//   //     cv::circle(img_color2, psrc, 0, cv::Scalar(0,255,0), -1);
//   //     cv::imshow("color1", img_color1);
//   //     cv::imshow("color2", img_color2);
//   //     while (1)
//   //     {
//   //       if(cv::waitKey(0) == 'q') break;
//   //       /* code */
//   //     }
//   //   }
//   // }

//   for(float depth = 0; depth < 100000; depth += 10.0f) {
//     pi.at<float>(0) = xc, pi.at<float>(1) = yc, pi.at<float>(2) = 1.0f;
//     norm.at<float>(0) = norm_center(0), norm.at<float>(1) = norm_center(1), norm.at<float>(2) = norm_center(2);
//     norm = R1.inv() * norm;
//     cv::Mat pc = K1.inv() * pi * depth;
//     cv::Mat pcw = K2 * (R21 * pc + t21);pcw/=pcw.at<float>(2);
//     float dist = pc.at<float>(0) * norm.at<float>(0) + pc.at<float>(1) * norm.at<float>(1) + pc.at<float>(2) * norm.at<float>(2);

//     cv::Mat H = K2*(R21 + t21*norm.t()/dist) * K1.inv();
//     cv::Mat qsrc = H * pi; qsrc /= qsrc.at<float>(2);
//     cv::Point2f pref(xc, yc), psrc(pcw.at<float>(0), pcw.at<float>(1));
//     cv::circle(img_color1, pref, 3, cv::Scalar(0,255,0), -1);
//     cv::circle(img_color2, psrc, 0, cv::Scalar(0,255,0), -1);
//     cv::imshow("color1", img_color1);
//     cv::imshow("color2", img_color2);
//     cv::waitKey(1);

//   }
  
  
//   // mvs_matcher->Run();

//   delete mvs_matcher;
//   delete options;

// }