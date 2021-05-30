/* 
 * LICENCE DECLARATION
 *
 * As of p4 (redistribution) in ./LICENSE this is a MODIFIED version of the
 * opencv_contrib ccalib module used for educational work.
 *
 * */


#ifndef __CHESSCALIB_HPP__
#define __CHESSCALIB_HPP__

/*{{{FOLD INCLUDES*/
#include "opencv2/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgcodecs.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>
#include <queue>
#include <iostream>

#include "checkerboard_marker_detection.hpp"

/*}}}FOLD*/

namespace cv { 
namespace chessboardcalib {

#define HEAD -1
#define INVALID -2

class chessboardcalib{
public:
  /*{{{FOLD PUBLIC structs and enums*/
  enum {
    PINHOLE,
    OMNIDIRECTIONAL
  };

  // an edge connects a camera and pattern
  struct edge
  {
    int cameraVertex;   // vertex index for camera in this edge
    int photoVertex;  // vertex index for pattern in this edge
    int photoIndex;   // photo index among photos for this camera
    Mat transform;    // transform from pattern to camera

    edge(int cv, int pv, int pi, Mat trans)
    {
      cameraVertex = cv;
      photoVertex = pv;
      photoIndex = pi;
      transform = trans;
    }
  };

  struct vertex
  {
    Mat pose;       // relative pose to the first camera. For camera vertex, it is the
                    // transform from the first camera to this camera, for pattern vertex,
                    // it is the transform from pattern to the first camera
    int timestamp;  // timestamp of photo, only available for photo vertex

    vertex(Mat po, int ts)
    {
      pose = po;
      timestamp = ts;
    }

    vertex()
    {
      pose = Mat::eye(4, 4, CV_32F);
      timestamp = -1;
    }
  };
/*}}}FOLD*/
private:
/*{{{FOLD PRIVATE variables*/
  /*VARIABLES*/
  int                               _camType; //PINHOLE or OMNIDIRECTIONAL
  int                               _nCamera;
  int                               _verbose;
  double                            _error;
  cm_info                           _checkerboard_info;
  TermCriteria                      _criteria;
 std::vector<std::string>           _filename;
  Ptr<FeatureDetector>              _detector;
  Ptr<DescriptorExtractor>          _descriptor;
  Ptr<DescriptorMatcher>            _matcher;

  std::vector<edge>                 _edgeList;
  std::vector<vertex>               _vertexList;
  std::vector<std::vector<cv::Mat>> _objectPointsForEachCamera;
  std::vector<std::vector<cv::Mat>> _imagePointsForEachCamera;
  std::vector<cv::Mat>              _cameraMatrix;
  std::vector<cv::Mat>              _distortCoeffs;
  std::vector<cv::Mat>              _xi;
  std::vector<std::vector<Mat>>     _omEachCamera, _tEachCamera;
/*}}}FOLD*/
/*{{{FOLD PRIVATE functions*/

int 
getPhotoVertex(int timestamp)
{
    int photoVertex = INVALID;
    // find in existing photo vertex
    for (int i = 0; i < (int)_vertexList.size(); ++i)
    {
        if (_vertexList[i].timestamp == timestamp)
        {
            photoVertex = i;
            break;
        }
    }
    // add a new photo vertex
    if (photoVertex == INVALID)
    {
        _vertexList.push_back(vertex(Mat::eye(4, 4, CV_32F), timestamp));
        photoVertex = (int)_vertexList.size() - 1;
    }
    return photoVertex;
}

void 
graphTraverse(const Mat& G, 
              int begin, 
              std::vector<int>& order, 
              std::vector<int>& pre)
{
//    printf("graphTraverse() start\n");
    CV_Assert(!G.empty() && G.rows == G.cols);
    int nVertex = G.rows;
    order.resize(0);
    pre.resize(nVertex, INVALID);
    pre[begin] = -1;
    std::vector<bool> visited(nVertex, false);
    std::queue<int> q;
    visited[begin] = true;
    q.push(begin);
    order.push_back(begin);

    while(!q.empty())
    {
        int v = q.front();
        q.pop();
        Mat idx;
        // use my findNonZero maybe
        findRowNonZero(G.row(v), idx);
        for(int i = 0; i < (int)idx.total(); ++i)
        {
            int neighbor = idx.at<int>(i);
            if (!visited[neighbor])
            {
                visited[neighbor] = true;
                q.push(neighbor);
                order.push_back(neighbor);
                pre[neighbor] = v;
            }
        }
    }
}

void 
findRowNonZero(const Mat& row, Mat& idx)
{
    CV_Assert(!row.empty() && row.rows == 1 && row.channels() == 1);
    Mat _row;
    std::vector<int> _idx;
    row.convertTo(_row, CV_32F);
    for (int i = 0; i < (int)row.total(); ++i)
    {
        if (_row.at<float>(i) != 0)
        {
            _idx.push_back(i);
        }
    }
    idx.release();
    idx.create(1, (int)_idx.size(), CV_32S);
    for (int i = 0; i < (int)_idx.size(); ++i)
    {
        idx.at<int>(i) = _idx[i];
    }
}


void 
computeJacobianExtrinsic(const Mat& extrinsicParams, 
                         Mat& JTJ_inv, 
                         Mat& JTE)
{
    int nParam = (int)extrinsicParams.total();
    int nEdge = (int)_edgeList.size();
    std::vector<int> pointsLocation(nEdge+1, 0);

    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        int nPoints = (int)_objectPointsForEachCamera[_edgeList[edgeIdx].cameraVertex][_edgeList[edgeIdx].photoIndex].total();
        pointsLocation[edgeIdx+1] = pointsLocation[edgeIdx] + nPoints*2;
    }

    JTJ_inv = Mat(nParam, nParam, CV_64F);
    JTE = Mat(nParam, 1, CV_64F);

    Mat J = Mat::zeros(pointsLocation[nEdge], nParam, CV_64F);
    Mat E = Mat::zeros(pointsLocation[nEdge], 1, CV_64F);

    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        int photoVertex = _edgeList[edgeIdx].photoVertex;
        int photoIndex = _edgeList[edgeIdx].photoIndex;
        int cameraVertex = _edgeList[edgeIdx].cameraVertex;

        Mat objectPoints = _objectPointsForEachCamera[cameraVertex][photoIndex];
        Mat imagePoints = _imagePointsForEachCamera[cameraVertex][photoIndex];

        Mat rvecTran, tvecTran;
        Mat R = _edgeList[edgeIdx].transform.rowRange(0, 3).colRange(0, 3);
        tvecTran = _edgeList[edgeIdx].transform.rowRange(0, 3).col(3);
        cv::Rodrigues(R, rvecTran);

        Mat rvecPhoto = extrinsicParams.colRange((photoVertex-1)*6, (photoVertex-1)*6 + 3);
        Mat tvecPhoto = extrinsicParams.colRange((photoVertex-1)*6 + 3, (photoVertex-1)*6 + 6);

        Mat rvecCamera, tvecCamera;
        if (cameraVertex > 0)
        {
            rvecCamera = extrinsicParams.colRange((cameraVertex-1)*6, (cameraVertex-1)*6 + 3);
            tvecCamera = extrinsicParams.colRange((cameraVertex-1)*6 + 3, (cameraVertex-1)*6 + 6);
        }
        else
        {
            rvecCamera = Mat::zeros(3, 1, CV_32F);
            tvecCamera = Mat::zeros(3, 1, CV_32F);
        }

        Mat jacobianPhoto, jacobianCamera, error;
//        printf("I AM IN computeJacobianExtrinsic()\n"); FLUSHO;
//        printf("computePhotoCameraJacobian() executed %d times \n", edgeIdx); FLUSHO;
        computePhotoCameraJacobian(rvecPhoto, tvecPhoto, rvecCamera, tvecCamera, rvecTran, tvecTran,
            objectPoints, imagePoints, this->_cameraMatrix[cameraVertex], this->_distortCoeffs[cameraVertex],
            this->_xi[cameraVertex], jacobianPhoto, jacobianCamera, error);

        if (cameraVertex > 0)
        {
            jacobianCamera.copyTo(J.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]).
                colRange((cameraVertex-1)*6, cameraVertex*6));
        }
        jacobianPhoto.copyTo(J.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]).
            colRange((photoVertex-1)*6, photoVertex*6));
        error.copyTo(E.rowRange(pointsLocation[edgeIdx], pointsLocation[edgeIdx+1]));
    }
    JTJ_inv = (J.t() * J + 1e-10).inv();
    JTE = J.t() * E;
//    printf("EXITING computeJacobianExtrinsic()\n"); FLUSHO;
}

void 
computePhotoCameraJacobian(const Mat& rvecPhoto, 
                           const Mat& tvecPhoto, 
                           const Mat& rvecCamera,
                           const Mat& tvecCamera, 
                           Mat& rvecTran, 
                           Mat& tvecTran, 
                           const Mat& objectPoints, 
                           const Mat& imagePoints, 
                           const Mat& K,
                           const Mat& distort, 
                           const Mat& xi, 
                           Mat& jacobianPhoto, 
                           Mat& jacobianCamera, 
                           Mat& E)
{

    Mat drvecTran_drecvPhoto, drvecTran_dtvecPhoto,
        drvecTran_drvecCamera, drvecTran_dtvecCamera,
        dtvecTran_drvecPhoto, dtvecTran_dtvecPhoto,
        dtvecTran_drvecCamera, dtvecTran_dtvecCamera;

    compose_motion(rvecPhoto, tvecPhoto, rvecCamera, tvecCamera, rvecTran, tvecTran,
        drvecTran_drecvPhoto, drvecTran_dtvecPhoto, drvecTran_drvecCamera, drvecTran_dtvecCamera,
        dtvecTran_drvecPhoto, dtvecTran_dtvecPhoto, dtvecTran_drvecCamera, dtvecTran_dtvecCamera);


    if (rvecTran.depth() == CV_64F)
    {
        rvecTran.convertTo(rvecTran, CV_32F);
    }
    if (tvecTran.depth() == CV_64F)
    {
        tvecTran.convertTo(tvecTran, CV_32F);
    }

    Mat imagePoints2, jacobian, dx_drvecCamera, dx_dtvecCamera, dx_drvecPhoto, dx_dtvecPhoto;
    cv::projectPoints(objectPoints, rvecTran, tvecTran, K, distort, imagePoints2, jacobian);

//    cout << "Things going into cv::projectPoints()\n"
//         << "objectPoints\n" << objectPoints << "\n"
//         << "rvecTran\n" << rvecTran << "\n"
//         << "tvecTran\n" << tvecTran << "\n"
//         << "K (camera matrix)\n" << K << "\n"
//         << "distortion parameters\n" << distort << "\n\n";

    if (objectPoints.depth() == CV_32F)
    {
        Mat(imagePoints - imagePoints2).convertTo(E, CV_64FC2);
    }
    else
    {
        E = imagePoints - imagePoints2;
    }
    E = E.reshape(1, (int)imagePoints.total()*2);

//        printf("I AM IN computePhotoCameraJacobian()\n"); FLUSHO;
//        printf("imgpoints size = %d,%d\n imgpoints2 size = %d,%d\n",imagePoints.rows, imagePoints.cols, imagePoints2.rows, imagePoints2.cols);

//        cout << "Imagepoints: \n"<< imagePoints << "\n\nImagepoints2: \n" << imagePoints2;


    dx_drvecCamera = jacobian.colRange(0, 3) * drvecTran_drvecCamera + jacobian.colRange(3, 6) * dtvecTran_drvecCamera;
    dx_dtvecCamera = jacobian.colRange(0, 3) * drvecTran_dtvecCamera + jacobian.colRange(3, 6) * dtvecTran_dtvecCamera;
    dx_drvecPhoto = jacobian.colRange(0, 3) * drvecTran_drecvPhoto + jacobian.colRange(3, 6) * dtvecTran_drvecPhoto;
    dx_dtvecPhoto = jacobian.colRange(0, 3) * drvecTran_dtvecPhoto + jacobian.colRange(3, 6) * dtvecTran_dtvecPhoto;

    jacobianCamera = cv::Mat(dx_drvecCamera.rows, 6, CV_64F);
    jacobianPhoto = cv::Mat(dx_drvecPhoto.rows, 6, CV_64F);

    dx_drvecCamera.copyTo(jacobianCamera.colRange(0, 3));
    dx_dtvecCamera.copyTo(jacobianCamera.colRange(3, 6));
    dx_drvecPhoto.copyTo(jacobianPhoto.colRange(0, 3));
    dx_dtvecPhoto.copyTo(jacobianPhoto.colRange(3, 6));
}

  void compose_motion(InputArray _om1, 
                      InputArray _T1, 
                      InputArray _om2, 
                      InputArray _T2, 
                      Mat& om3, 
                      Mat& T3, 
                      Mat& dom3dom1,
                      Mat& dom3dT1, 
                      Mat& dom3dom2, 
                      Mat& dom3dT2, 
                      Mat& dT3dom1, 
                      Mat& dT3dT1, 
                      Mat& dT3dom2, 
                      Mat& dT3dT2)
{
    Mat om1, om2, T1, T2;
    _om1.getMat().convertTo(om1, CV_64F);
    _om2.getMat().convertTo(om2, CV_64F);
    _T1.getMat().reshape(1, 3).convertTo(T1, CV_64F);
    _T2.getMat().reshape(1, 3).convertTo(T2, CV_64F);
    /*Mat om2 = _om2.getMat();
    Mat T1 = _T1.getMat().reshape(1, 3);
    Mat T2 = _T2.getMat().reshape(1, 3);*/

    //% Rotations:
    Mat R1, R2, R3, dR1dom1(9, 3, CV_64FC1), dR2dom2;
    cv::Rodrigues(om1, R1, dR1dom1);
    cv::Rodrigues(om2, R2, dR2dom2);
    /*JRodriguesMatlab(dR1dom1, dR1dom1);
    JRodriguesMatlab(dR2dom2, dR2dom2);*/
    dR1dom1 = dR1dom1.t();
    dR2dom2 = dR2dom2.t();

    R3 = R2 * R1;
    Mat dR3dR2, dR3dR1;
    //dAB(R2, R1, dR3dR2, dR3dR1);
    matMulDeriv(R2, R1, dR3dR2, dR3dR1);
    Mat dom3dR3;
    cv::Rodrigues(R3, om3, dom3dR3);
    //JRodriguesMatlab(dom3dR3, dom3dR3);
    dom3dR3 = dom3dR3.t();

    dom3dom1 = dom3dR3 * dR3dR1 * dR1dom1;
    dom3dom2 = dom3dR3 * dR3dR2 * dR2dom2;
    dom3dT1 = Mat::zeros(3, 3, CV_64FC1);
    dom3dT2 = Mat::zeros(3, 3, CV_64FC1);

    //% Translations:
    Mat T3t = R2 * T1;
    Mat dT3tdR2, dT3tdT1;
    //dAB(R2, T1, dT3tdR2, dT3tdT1);
    matMulDeriv(R2, T1, dT3tdR2, dT3tdT1);

    Mat dT3tdom2 = dT3tdR2 * dR2dom2;
    T3 = T3t + T2;
    dT3dT1 = dT3tdT1;
    dT3dT2 = Mat::eye(3, 3, CV_64FC1);
    dT3dom2 = dT3tdom2;
    dT3dom1 = Mat::zeros(3, 3, CV_64FC1);
}

double 
computeProjectError(Mat& parameters)
{
    int nVertex = (int)_vertexList.size();
    CV_Assert((int)parameters.total() == (nVertex-1) * 6 && parameters.depth() == CV_32F);
    int nEdge = (int)_edgeList.size();

    // recompute the transform between photos and cameras

    std::vector<edge> edgeList = this->_edgeList;
    std::vector<Vec3f> rvecVertex, tvecVertex;
    vector2parameters(parameters, rvecVertex, tvecVertex);

    float totalError = 0;
    int totalNPoints = 0;
    for (int edgeIdx = 0; edgeIdx < nEdge; ++edgeIdx)
    {
        Mat RPhoto, RCamera, TPhoto, TCamera, transform;
        int cameraVertex = edgeList[edgeIdx].cameraVertex;
        int photoVertex = edgeList[edgeIdx].photoVertex;
        int PhotoIndex = edgeList[edgeIdx].photoIndex;
        TPhoto = Mat(tvecVertex[photoVertex - 1]).reshape(1, 3);

        //edgeList[edgeIdx].transform = Mat::ones(4, 4, CV_32F);
        transform = Mat::eye(4, 4, CV_32F);
        cv::Rodrigues(rvecVertex[photoVertex-1], RPhoto);
        if (cameraVertex == 0)
        {
            RPhoto.copyTo(transform.rowRange(0, 3).colRange(0, 3));
            TPhoto.copyTo(transform.rowRange(0, 3).col(3));
        }
        else
        {
            TCamera = Mat(tvecVertex[cameraVertex - 1]).reshape(1, 3);
            cv::Rodrigues(rvecVertex[cameraVertex - 1], RCamera);
            Mat(RCamera*RPhoto).copyTo(transform.rowRange(0, 3).colRange(0, 3));
            Mat(RCamera * TPhoto + TCamera).copyTo(transform.rowRange(0, 3).col(3));
        }

        transform.copyTo(edgeList[edgeIdx].transform);
        Mat rvec, tvec;
        cv::Rodrigues(transform.rowRange(0, 3).colRange(0, 3), rvec);
        transform.rowRange(0, 3).col(3).copyTo(tvec);

        Mat objectPoints, imagePoints, proImagePoints;
        objectPoints = this->_objectPointsForEachCamera[cameraVertex][PhotoIndex];
        imagePoints = this->_imagePointsForEachCamera[cameraVertex][PhotoIndex];

        if (this->_camType == PINHOLE)
        {
            cv::projectPoints(objectPoints, rvec, tvec, _cameraMatrix[cameraVertex], _distortCoeffs[cameraVertex],
                proImagePoints);
        }
        else if (this->_camType == OMNIDIRECTIONAL)
        {
//            float xi = _xi[cameraVertex].at<float>(0);
            printf("NO OMNIDIRECTIONAL!\n"); FLUSHO;
//            cv::omnidir::projectPoints(objectPoints, proImagePoints, rvec, tvec, _cameraMatrix[cameraVertex],
//                xi, _distortCoeffs[cameraVertex]);
        }
        Mat error = imagePoints - proImagePoints;
        Vec2f* ptr_err = error.ptr<Vec2f>();
        for (int i = 0; i < (int)error.total(); ++i)
        {
            totalError += sqrt(ptr_err[i][0]*ptr_err[i][0] + ptr_err[i][1]*ptr_err[i][1]);
        }
        totalNPoints += (int)error.total();
    }
    double meanReProjError = totalError / totalNPoints;
    _error = meanReProjError;
    return meanReProjError;
}
  void vector2parameters(const Mat& parameters, 
                         std::vector<Vec3f>& rvecVertex, 
                         std::vector<Vec3f>& tvecVertexs)
{
    int nVertex = (int)_vertexList.size();
    CV_Assert((int)parameters.channels() == 1 && (int)parameters.total() == 6*(nVertex - 1));
    CV_Assert(parameters.depth() == CV_32F);
    parameters.reshape(1, 1);

    rvecVertex.reserve(0);
    tvecVertexs.resize(0);

    for (int i = 0; i < nVertex - 1; ++i)
    {
        rvecVertex.push_back(Vec3f(parameters.colRange(i*6, i*6 + 3)));
        tvecVertexs.push_back(Vec3f(parameters.colRange(i*6 + 3, i*6 + 6)));
    }
}

void 
parameters2vector(const std::vector<Vec3f>& rvecVertex, 
                  const std::vector<Vec3f>& tvecVertex, 
                  Mat& parameters)
{
    CV_Assert(rvecVertex.size() == tvecVertex.size());
    int nVertex = (int)rvecVertex.size();
    // the pose of the first camera is known
    parameters.create(1, 6*(nVertex-1), CV_32F);

    for (int i = 0; i < nVertex-1; ++i)
    {
        Mat(rvecVertex[i]).reshape(1, 1).copyTo(parameters.colRange(i*6, i*6 + 3));
        Mat(tvecVertex[i]).reshape(1, 1).copyTo(parameters.colRange(i*6 + 3, i*6 + 6));
    }
}
/*}}}FOLD*/
public:
/*{{{FOLD PUBLIC functions*/
chessboardcalib(int nCameras, 
                std::vector<std::string> fileNames,
                cm_info checkerboard_info,
                TermCriteria criteria,
                int verbose)
{
  _camType = PINHOLE;
  _nCamera = nCameras;
  _filename = fileNames;
  _checkerboard_info = checkerboard_info;
  _criteria = criteria;
  _objectPointsForEachCamera.resize(_nCamera);
  _imagePointsForEachCamera.resize(_nCamera);
  _cameraMatrix.resize(_nCamera);
  _distortCoeffs.resize(_nCamera);
  _xi.resize(_nCamera);
  _omEachCamera.resize(_nCamera);
  _tEachCamera.resize(_nCamera);
  _verbose = verbose;
  for (int i = 0; i < _nCamera; ++i)
  {
    _vertexList.push_back(vertex());
  }
}

void 
loadImages(int verbose = 0)
{
  std::vector<std::string>              file_list = _filename;
  std::vector<std::vector<std::string>> filesEachCameraFull(_nCamera);
  std::vector<std::vector<int>>         timestampFull(_nCamera);
  std::vector<std::vector<int>>         timestampAvailable(_nCamera);

  if (verbose)
  printf("%d files in calibration\n",(int)file_list.size()); FLUSHO;

  printf("%d files per camera\n",(int)file_list.size()/_nCamera); FLUSHO;

  for (int i = 1; i < (int)file_list.size(); ++i)
  {
    int cameraVertex, timestamp;
    std::string filename  = file_list[i].substr(0, file_list[i].rfind('.'));
    size_t spritPosition1 = filename.rfind('/');
    if (spritPosition1!=std::string::npos)
    {
      filename = filename.substr(spritPosition1+1, filename.size() - 1);
    }
    sscanf(filename.c_str(), "%d-%d", &cameraVertex, &timestamp);
    filesEachCameraFull[cameraVertex].push_back(file_list[i]);
    timestampFull[cameraVertex].push_back(timestamp);
  }

  if (verbose)
    printf("Files parsed sucessfully\n");

  /*calibrate each camera individually*/
  for (int camera = 0; camera < _nCamera; ++camera)
  {
    Mat image, cameraMatrix, distortCoeffs;
    std::vector<checkerboard_calib_data> all_chessboard_data;

    /*for each image -> find image and object points */
    for (int imgIdx = 0; imgIdx < (int)filesEachCameraFull[camera].size(); ++imgIdx)
    {
      if (_verbose)
        cout << "Current file: \n" << filesEachCameraFull[camera][imgIdx] << "\n";
      /*load image*/
      image = imread(filesEachCameraFull[camera][imgIdx], IMREAD_GRAYSCALE);
      if (!image.empty() && _verbose)
      {
        std::cout << "open image " 
                  << filesEachCameraFull[camera][imgIdx] 
                  << " successfully" 
                  << std::endl;
      }
      else if (image.empty() && _verbose)
      {
        std::cout << "open image " 
                  << filesEachCameraFull[camera][imgIdx] 
                  << " failed!" 
                  << std::endl;
      }

      checkerboard_calib_data curr_data = 
        single_checkerboard_detection_no_calibration(image, _checkerboard_info);
      all_chessboard_data.push_back(curr_data);

      if (curr_data.image_points.size() > 0 && curr_data.object_points.size() > 0) {
        if (verbose)
            printf("checkerboard_detection() board is sucessfuly found in image ->  %d-%d\n"
                ,camera, imgIdx); FLUSHO;
          _imagePointsForEachCamera[camera].push_back(cv::Mat(curr_data.image_points));
          _objectPointsForEachCamera[camera].push_back(cv::Mat(curr_data.object_points));
          timestampAvailable[camera].push_back(timestampFull[camera][imgIdx]);
      }
    }

     /*Calibrate*/
     cv::Mat idx;
     full_checkerboard_calib calib = calibrate_checkerboard(all_chessboard_data);

     calib.cameraMatrix.copyTo(_cameraMatrix[camera]);
     calib.distCoeffs.copyTo(_distortCoeffs[camera]);
     if (calib.rvecs.size() > 0)
       _omEachCamera[camera] = calib.rvecs;

     if (calib.tvecs.size() > 0)
       _tEachCamera[camera] = calib.tvecs;

     calib.distCoeffs.copyTo(_xi[camera]);

     idx = Mat(1, (int)_omEachCamera[camera].size(), CV_32S);
     for (int i = 0; i < (int)idx.total(); ++i)
     {
       idx.at<int>(i) = i;
     }

     _cameraMatrix[camera].convertTo(_cameraMatrix[camera], CV_32F);
     _distortCoeffs[camera].convertTo(_distortCoeffs[camera], CV_32F);
     _xi[camera].convertTo(_xi[camera], CV_32F);

      for (int i = 0; i < (int)_omEachCamera[camera].size(); ++i)
      {
         int cameraVertex, timestamp, photoVertex;
         cameraVertex = camera;

         timestamp = timestampAvailable[camera][idx.at<int>(i)];
         
         photoVertex = this->getPhotoVertex(timestamp);

         if (_omEachCamera[camera][i].type()!=CV_32F)
         {
           _omEachCamera[camera][i].convertTo(_omEachCamera[camera][i], CV_32F);
         }
         if (_tEachCamera[camera][i].type()!=CV_32F)
         {
           _tEachCamera[camera][i].convertTo(_tEachCamera[camera][i], CV_32F);
         }

         Mat transform = Mat::eye(4, 4, CV_32F);
         Mat R, T;
         Rodrigues(_omEachCamera[camera][i], R);
         T = (_tEachCamera[camera][i]).reshape(1, 3);
         R.copyTo(transform.rowRange(0, 3).colRange(0, 3));
         T.copyTo(transform.rowRange(0, 3).col(3));

         this->_edgeList.push_back(edge(cameraVertex, photoVertex, idx.at<int>(i), transform));
      }
      cout << "-----\n";
      std::cout << "initialized for camera " << camera << " rms = " << calib.repError << std::endl;
      std::cout << "initialized camera matrix for camera " << camera << " is" << std::endl;
      std::cout << _cameraMatrix[camera] << std::endl;
      std::cout << "xi for camera " << camera << " is " << _xi[camera] << std::endl;
      cout << "-----\n";
//      printf("CAMERA CALIBRATES SUCESSFULLY\n");
   } /* Individual Camera Calibration end */
}

void 
initialize()
{

//  for (int camera = 0; camera < _nCamera; camera++) {
//    for (int q = 0; q < _imagePointsForEachCamera[camera].size(); q++) {
//    cout << _imagePointsForEachCamera[camera][q] << "\n";
//    cout << _objectPointsForEachCamera[camera][q] << "\n";
//    } 
//  }

//   for (int camera = 0; camera < _nCamera; camera++) {
//    for (int q = 0; q < _imagePointsForEachCamera[camera].size(); q++) {
//    cout << _omEachCamera[camera][q] << "\n";
//    cout << _tEachCamera[camera][q] << "\n";
//    } 
//  }

    int nVertices = (int)_vertexList.size();
    int nEdges = (int) _edgeList.size();
    printf("verticies = %d, edges = %d\n", nVertices, nEdges);

    // build graph
//    printf("build graph start\n");
    Mat G = Mat::zeros(nVertices, nVertices, CV_32S);

    for (int edgeIdx = 0; edgeIdx < nEdges; ++edgeIdx)
    {
        G.at<int>(this->_edgeList[edgeIdx].cameraVertex, this->_edgeList[edgeIdx].photoVertex) = edgeIdx + 1;
//        printf("added %d\n", edgeIdx + 1);
    }
    G = G + G.t();

    // traverse the graph
    std::vector<int> pre, order;
    graphTraverse(G, 0, order, pre);

    for (int i = 0; i < _nCamera; ++i)
    {
        if (pre[i] == INVALID) {
            std::cout << "camera " << i << " is not connected" << std::endl;
        }
        else {
            std::cout << "camera " << i << " is connected" << std::endl;
        }
    }

    for (int i = 1; i < (int)order.size(); ++i)
    {
        int vertexIdx = order[i];
        Mat prePose = this->_vertexList[pre[vertexIdx]].pose;
        int edgeIdx = G.at<int>(vertexIdx, pre[vertexIdx]) - 1;
        Mat transform = this->_edgeList[edgeIdx].transform;

        if (vertexIdx < _nCamera)
        {
            this->_vertexList[vertexIdx].pose = transform * prePose.inv();
            this->_vertexList[vertexIdx].pose.convertTo(this->_vertexList[vertexIdx].pose, CV_32F);
//            if (_verbose)
//            {
              std::cout << "initial pose for camera " << vertexIdx << " is " << std::endl;
              std::cout << this->_vertexList[vertexIdx].pose << std::endl;
//            }
        }
        else
        {
            this->_vertexList[vertexIdx].pose = prePose.inv() * transform;
            this->_vertexList[vertexIdx].pose.convertTo(this->_vertexList[vertexIdx].pose, CV_32F);
        }
    }
}

double 
optimizeExtrinsics()
{
    // get om, t vector
    int nVertex = (int)this->_vertexList.size();
//    printf("nVertex = %d\n",nVertex);
    
    Mat extrinParam(1, (nVertex-1)*6, CV_32F);
    int offset = 0;
    for (int i = 1; i < nVertex; ++i)
    {
        Mat rvec, tvec;
        cv::Rodrigues(this->_vertexList[i].pose.rowRange(0,3).colRange(0, 3), rvec);
        this->_vertexList[i].pose.rowRange(0,3).col(3).copyTo(tvec);
        
        rvec.reshape(1, 1).copyTo(extrinParam.colRange(offset, offset + 3));
        tvec.reshape(1, 1).copyTo(extrinParam.colRange(offset+3, offset +6));
        offset += 6;
    }
    
//    cout << "extrinParam looks like this: \n" << extrinParam << "\n"; 

    // optimization
    const double alpha_smooth = 0.01;
    double change = 1;

    for(int iter = 0; ; ++iter)
    {
        if ((_criteria.type == 1 && iter >= _criteria.maxCount)  ||
            (_criteria.type == 2 && change <= _criteria.epsilon) ||
            (_criteria.type == 3 && (change <= _criteria.epsilon || 
             iter >= _criteria.maxCount)))
            break;
        double alpha_smooth2 = 1 - std::pow(1 - alpha_smooth, (double)iter + 1.0);
        Mat JTJ_inv, JTError;
        this->computeJacobianExtrinsic(extrinParam, JTJ_inv, JTError);
        Mat G = alpha_smooth2*JTJ_inv * JTError;
        if (G.depth() == CV_64F)
        {
            G.convertTo(G, CV_32F);
        }

        extrinParam = extrinParam + G.reshape(1, 1);

        change = norm(G) / norm(extrinParam);
    }
    printf("Bundle adjustment finished\n");

    double error = computeProjectError(extrinParam);
    printf("total error in system = %f\n", error);

    std::vector<Vec3f> rvecVertex, tvecVertex;
    vector2parameters(extrinParam, rvecVertex, tvecVertex);
    for (int verIdx = 1; verIdx < (int)_vertexList.size(); ++verIdx)
    {
        Mat R;
        Mat pose = Mat::eye(4, 4, CV_32F);
        Rodrigues(rvecVertex[verIdx-1], R);
        R.copyTo(pose.colRange(0, 3).rowRange(0, 3));
        Mat(tvecVertex[verIdx-1]).reshape(1, 3).copyTo(pose.rowRange(0, 3).col(3));
        _vertexList[verIdx].pose = pose;

      if (verIdx < _nCamera) {
        std::cout << "final camera pose of camera " << verIdx << " is" << std::endl;
        std::cout << pose << std::endl;
      }
    }
    return error;
}

double 
run()
{
    loadImages();
    initialize();
    double error = optimizeExtrinsics();
    return error;
}

void writeParameters(const std::string& filename)
{
  printf("writing parameters to file\n");
    FileStorage fs( filename, FileStorage::WRITE );

    fs << "nCameras" << _nCamera;

    for (int camIdx = 0; camIdx < _nCamera; ++camIdx)
    {
        std::stringstream tmpStr;
        tmpStr << camIdx;
        std::string cameraMatrix = "camera_matrix_" + tmpStr.str();
        std::string cameraPose = "camera_pose_" + tmpStr.str();
        std::string cameraDistortion = "camera_distortion_" + tmpStr.str();
        std::string cameraXi = "xi_" + tmpStr.str();

        fs << cameraMatrix << _cameraMatrix[camIdx];
        fs << cameraDistortion << _distortCoeffs[camIdx];
        if (_camType == OMNIDIRECTIONAL)
        {
            fs << cameraXi << _xi[camIdx].at<float>(0);
        }

        fs << cameraPose << _vertexList[camIdx].pose;
    }

    fs << "meanReprojectError" <<_error;

    for (int photoIdx = _nCamera; photoIdx < (int)_vertexList.size(); ++photoIdx)
    {
        std::stringstream tmpStr;
        tmpStr << _vertexList[photoIdx].timestamp;
        std::string photoTimestamp = "pose_timestamp_" + tmpStr.str();

        fs << photoTimestamp << _vertexList[photoIdx].pose;
    }
}

/*}}}FOLD*/
}; /*end class chessboardcalib*/
}; /*end namespace cv*/
}; /*end namespace chessboardcalib*/

#endif /*__CHESSCALIB_HPP__*/
