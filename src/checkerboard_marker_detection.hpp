#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <stdio.h>
#include <string>

#ifndef PRINTEOF
#define PRINTEOF printf("EOF!\n"); fflush(stdout);
#endif
#ifndef FLUSHO
#define FLUSHO fflush(stdout);
#endif


#ifndef CHECKERBOARD_MARKER_CALIB_HPP
#define CHECKERBOARD_MARKER_CALIB_HPP


/* SEE LICENSE DECLARATION IN ./LICENSE and ./checkercalib.hpp
 * */

typedef struct {
  cv::Size board_size;
  float    checker_size_m;
} cm_info;

typedef struct {
  cm_info                  board_info;
  cv::Mat                  original_img;
  cv::Mat                  visualization_img;
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
}checkerboard_calib_data;

typedef struct {
  int                                   complete;
  double                                repError; 
  cv::Mat                               cameraMatrix, distCoeffs;
  std::vector<cv::Mat>                  rvecs, tvecs;
  std::vector<std::vector<cv::Point3f>> all_objectpoints;
  std::vector<std::vector<cv::Point2f>> all_imagepoints;
}full_checkerboard_calib;


cm_info
checkerboard_info_create(cv::Size _board_size, float _ml)
{
  cm_info _new;
  _new.board_size = _board_size;
  _new.checker_size_m = _ml;
  return _new;
}

cm_info
MY_CHECKERBOARD_DATA(void)
{
   cm_info _new;
  _new.board_size = cv::Size(12-1, 9 -1);
  _new.checker_size_m = 0.060; /*60 mm*/
  return _new;
}

std::vector<cv::Point3f>
get_object_points(cm_info _data)
{
  std::vector<cv::Point3f> res;
//  printf("board size = %d,%d\n", _data.board_size.height, _data.board_size.width);
  for (int i = 0; i < _data.board_size.height; i++) {
    for (int j = 0; j < _data.board_size.width; j++) {
      res.push_back(cv::Point3f(i * _data.checker_size_m, 
                                j * _data.checker_size_m, 
                                0.0f));
    }
  }
  return res;
}

checkerboard_calib_data
single_checkerboard_detection_no_calibration(cv::Mat _img, cm_info _info)
{
  bool sucess = false;
  checkerboard_calib_data _calib;

//  printf(">> SINGLE CHECKER DETECTION\n");
  _calib.board_info = _info;
  _img.copyTo(_calib.original_img);
  _img.copyTo(_calib.visualization_img);
  _calib.object_points = get_object_points(_info);

  sucess = cv::findChessboardCornersSB(_calib.original_img, 
                                       _calib.board_info.board_size, 
                                       _calib.image_points, 
                                       cv::CALIB_CB_MARKER);
//  if (sucess)
//    printf(">> FOUND THE BOARD\n");

  return _calib;
}

void
print_calibration_data(checkerboard_calib_data *_data)
{
  printf("checker data:\n");
  for (size_t i = 0; i < _data->image_points.size(); i++)
    printf(">> point %d = [%f,%f]\n", (int)i, 
                                      _data->image_points[i].x,
                                      _data->image_points[i].y);
}

full_checkerboard_calib
calibrate_checkerboard(std::vector<checkerboard_calib_data> _checker_data)
{
  full_checkerboard_calib res;
  if (_checker_data.size() == 0) {
//    printf("no data was present, calibration can not complete\n");
    res.complete = 0;
    return res;
  }
  int board_with_calib_data = -1;
  for (size_t i = 0; i < _checker_data.size(); i++) {
    /*a minimum of 4 correct corners are needed for calibration to work*/
    if (_checker_data[i].image_points.size() > 4) {
      board_with_calib_data = i;
      res.all_imagepoints.push_back(_checker_data[i].image_points);
      res.all_objectpoints.push_back(_checker_data[i].object_points);
    }
  }

//  printf("size of object points = %d\n",(int)res.all_objectpoints.size()); FLUSHO;
//  printf("size of image points = %d\n",(int)res.all_imagepoints.size()); FLUSHO;
   
  if (board_with_calib_data != -1) {
//    printf("CALIBRATION STARTS\n"); FLUSHO;
    res.repError = cv::calibrateCamera(res.all_objectpoints, 
                                       res.all_imagepoints,
                                       _checker_data[0].original_img.size(),
                                       res.cameraMatrix,
                                       res.distCoeffs,
                                       res.rvecs,
                                       res.tvecs);
  res.complete = 1;
  }
//  else
//  printf("CALIBRATION DID NOT HAPPEN\n"); FLUSHO;
//  if (_checker_data[0].board_info.board.empty())
//  printf("BOARD DOES NOT EXIST!\n"); FLUSHO;
  return res;
}


#endif /*CHECKERBOARD_MARKER_CALIB_HPP*/
