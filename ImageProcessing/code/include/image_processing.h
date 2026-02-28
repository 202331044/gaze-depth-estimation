#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include "pupil_detector.h"

void saveFramesFromVideo(std::string videoPath, std::string savePath, int sbjIdx, int interval);
std::vector<std::string> readFiles(std::string filePath);
Pupil detectPupil(cv::Mat roi, int sbjIdx, bool isSavingImg, bool isShowingImg, std::string savePath);
cv::Point2f detectP1Center(int sbjIdx, cv::Mat proiBin, bool isShowingImg, bool isSavingImg, std::string savePath);
cv::Point2f detectP4Center(int sbjIdx, cv::Mat proi, int colCenter, int rowCenter, std::string tmplName, bool isShowingImg, bool isSavingImg, std::string savePath);
double calculateDpiDistance(cv::Point p1, cv::Point p4);

void drawCenters(cv::Mat img, cv::Point pupil, cv::Point p1, cv::Point p4);
void writeText(cv::Mat img, cv::Point pupil, cv::Point p1, cv::Point p4, int diameter, double distance);
void createResultImg(cv::Mat img, cv::Point pupil, cv::Point p1, cv::Point p4, int diameter, double distance, cv::Rect boundingBox);
void drawBoundingBox(cv::Mat img, cv::Rect boundingBox);

void setP1Position(std::string filepath, int cnt, int startLine, cv::Point origP4, cv::Point2f& newP1Center, float& newDpiDist);
void setP4Position(std::string filepath, int cnt, int startLine, cv::Point origP1, cv::Point2f& newP4Center, float& newDpiDist);