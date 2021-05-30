#include "opencv2/aruco.hpp"
#include "opencv2/aruco/dictionary.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <stdio.h>
#include <string>

#include "checkercalib.hpp"
#include "checkerboard_marker_detection.hpp"

#ifndef SEPERATOR
#define SEPERATOR ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
#endif


/* SEE LICENSE DECLARATION IN ./LICENSE and ./checkercalib.hpp
 * */

using namespace cv;
using namespace std;

std::string CHECKER_PATHS = "../dataset/";

std::vector<std::string>
SINGLE_CAM_PATHS(std::string _path, int _camidx, int _n, std::string _type)
{
  std::vector<std::string> img_names;
  std::string current_name;
  int idx;
  for (idx = 0; idx < _n; idx++) {
    current_name = _path + to_string(_camidx) + "-" + to_string(idx) + _type;
    img_names.push_back(current_name);
  }
  return img_names;
}

std::vector<std::string>
ALL_CAM_PATHS(std::string _path, int _cam_amount, int _n, std::string _type)
{
  std::vector<std::string> img_names;
  std::string current_name;
  int idx;
  int jdx;
  for (jdx = 0; jdx < _cam_amount; jdx++) {
    for (idx = 0; idx < _n; idx++) {
      current_name = _path + to_string(jdx) + "-" + to_string(idx) + _type;
      img_names.push_back(current_name);
    }
  }
  return img_names;
}

void 
test_single_checker_detection(void)
{
  std::vector<std::string> img_names = SINGLE_CAM_PATHS(CHECKER_PATHS, 2, 49, ".jpg");
  std::vector<checkerboard_calib_data> all_calib_data;
  unsigned int i;
  for (i = 0; i < img_names.size(); i++) {
    cv::Mat img = cv::imread(img_names[i]);
    cout << ">> " << img_names[i] << "\n";
    checkerboard_calib_data calib_data = 
      single_checkerboard_detection_no_calibration(img, MY_CHECKERBOARD_DATA());
//    print_calibration_data(&calib_data);
    all_calib_data.push_back(calib_data);
  }
  full_checkerboard_calib full_calib = calibrate_checkerboard(all_calib_data);  

  printf("\ncalibration reprojection error = %f\n", full_calib.repError);
//  cout << full_calib.cameraMatrix;
}

void
test_full_calibration(void)
{
  const char* outputFilename = "output_results.xml";
  int nCamera = 4;
  std::vector<std::string> inputFilenames = ALL_CAM_PATHS(CHECKER_PATHS, nCamera, 49, ".jpg");
  int verbose = 0;
  TermCriteria ba_stop_criteria;
  /*1 = Max iterations
   *2 = epsilon change (if change <= epsilon -> stop)
   *3 = epsilon change (if change >= epsilon -> stop)*/
  ba_stop_criteria.type = 2; 
  ba_stop_criteria.epsilon = 1;
  ba_stop_criteria.maxCount = 2000;

  // do multi-camera calibration
  cm_info checkerboard_info = MY_CHECKERBOARD_DATA();
  cv::chessboardcalib::chessboardcalib chessboardcalib(nCamera, inputFilenames, checkerboard_info, ba_stop_criteria, verbose);

  SEPERATOR;
  printf("LOAD IMAGES\n");
  SEPERATOR;
  chessboardcalib.loadImages();

  SEPERATOR;
  printf("INITIALIZE");
  SEPERATOR;
  chessboardcalib.initialize();

  SEPERATOR;
  printf("OPTIMIZE");
  SEPERATOR;
  chessboardcalib.optimizeExtrinsics();

  SEPERATOR;
  printf("WRITE PARAMETERS");
  SEPERATOR;
  chessboardcalib.writeParameters(outputFilename);
}

void
test_error_estimate_based_on_data_size(int _stepsize)
{
  int iteration = 1;
  while (iteration * _stepsize < 49)
  {
    SEPERATOR;
    printf("NEW ITERATION %d\n", iteration);
    SEPERATOR;

    int nCamera = 4;
    std::vector<std::string> inputFilenames = 
      ALL_CAM_PATHS(CHECKER_PATHS, nCamera, iteration * _stepsize, ".jpg");
    int verbose = 0;
    TermCriteria ba_stop_criteria;
    /*1 = Max iterations
     *2 = epsilon change (if change <= epsilon -> stop)
     *3 = epsilon change (if change >= epsilon -> stop)*/
    ba_stop_criteria.type = 2; 
    ba_stop_criteria.epsilon = 1;
    ba_stop_criteria.maxCount = 2000;

    // do multi-camera calibration
    cm_info checkerboard_info = MY_CHECKERBOARD_DATA();
    cv::chessboardcalib::chessboardcalib chessboardcalib(nCamera, inputFilenames, checkerboard_info, ba_stop_criteria, verbose);
    chessboardcalib.loadImages();

    iteration++;
  }
}

/******************************************************************************/
int 
main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  
//  test_single_checker_detection();
  test_full_calibration(); 
//  test_error_estimate_based_on_data_size(2);
  PRINTEOF;
  return 0;
}
