#pragma once
#include "opencv2/opencv.hpp"
#include <iostream>
#include "string"
#include "vector"
#include "cmath"
#include "fstream"
#include "pupil_detector.h"

using namespace std;
using namespace cv;

void saveFramesFromVideo(string videoPath, string savePath, int sbjIdx, int interval);
vector<string> readFiles(string filePath);
Pupil detectPupil(Mat roi, int sbjIdx, bool isSavingImg, bool isShowingImg, string savePath);
Point2f detectP1Center(int sbjIdx, Mat proiBin, bool isShowingImg, bool isSavingImg, string savePath);
Point2f detectP4Center(int sbjIdx, Mat proi, int colCenter, int rowCenter, string tmplName, bool isShowingImg, bool isSavingImg, string savePath);
double calculateDpiDistance(Point p1, Point p4);

void drawCenters(Mat img, Point pupil, Point p1, Point p4);
void writeText(Mat img, Point pupil, Point p1, Point p4, int diameter, double distance);
void createResultImg(Mat img, Point pupil, Point p1, Point p4, int diameter, double distance, Rect boundingBox);
void drawBoundingBox(Mat img, Rect boundingBox);

void setP1Position(string filepath, int cnt, int startLine, Point origP4, Point2f& newP1Center, float& newDpiDist);
void setP4Position(string filepath, int cnt, int startLine, Point origP1, Point2f& newP4Center, float& newDpiDist);