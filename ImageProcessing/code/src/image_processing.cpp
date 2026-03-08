#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include "pupil_detector.h"
#include <sstream>
#include <algorithm>

using namespace std;
using namespace cv;

void saveFramesFromVideo(string videoPath, string savePath, int sbjIdx, int interval)
{
	VideoCapture cap(videoPath);

	if (!cap.isOpened())
	{
		cerr << "Failed to open video!" << endl;
		return;
	}

	string depth = videoPath.substr(videoPath.size() - 8, 4);
	string saveFramePath = savePath + "sbj" + to_string(sbjIdx) + "/" + depth;
	
	double fps = cap.get(CAP_PROP_FPS);
	cout << "FPS: " << fps << "\n";
	cout << "Press ESC to exit the program, or press S to save the frame." << '\n';

	int cnt = 1;
	int intervalCnt = 0;

	while (true)
	{
		Mat frame;
		cap >> frame;
		if (frame.empty()) break;

		if (++intervalCnt == interval)
		{
			intervalCnt = 0;

			imshow("frame", frame);
			int key = waitKey(0);
			if (frame.empty() || key == 27) break;

			if (key == 's')
			{
				imwrite(saveFramePath + to_string(cnt) + ".jpg", frame);
				cnt++;
			}
		}
	}

	destroyAllWindows();
}

vector<string> readFiles(string filePath)
{
	vector<string> filenames;
	glob(filePath, filenames, false);
	cout << "Number of files: " << filenames.size() << endl;

	return filenames;
}

Pupil detectPupil(Mat roiBin, int sbjIdx, bool isSavingImg, bool isShowingImg, string savePath)
{
	vector<vector<Point>> pupilContours;
	vector<Vec4i> pupilHierarchy;
	findContours(roiBin, pupilContours, pupilHierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < pupilContours.size(); i++)
	{
		int parentIdx = pupilHierarchy[i][3];
		if (parentIdx == -1) continue;

		double area = contourArea(pupilContours[i]);
		double perimeter = arcLength(pupilContours[i], true);

		double areaTh = 4000, perimeterTh = 200;
		int pointTh = 30;

		if (sbjIdx == 2 || sbjIdx == 4 || sbjIdx == 8) 
			areaTh = 200, perimeterTh = 50; pointTh = 20;
	
		if (area < areaTh && perimeter < perimeterTh)
			continue;

		vector<Point> hull;
		convexHull(pupilContours[i], hull);
		if (hull.size() < pointTh) continue;

		RotatedRect ellipse = fitEllipse(hull);

		Point2f pupilCenter = ellipse.center;
		float diameter = max(ellipse.size.width, ellipse.size.height);

		float height = ellipse.size.width;
		float width = ellipse.size.height;

		if (abs(ellipse.angle) <= 45 || abs(ellipse.angle) >= 135)
			swap(height, width);

		Rect boundingBox;
		boundingBox.x = ellipse.center.x - width / 2;
		boundingBox.y = ellipse.center.y - height / 2;

		boundingBox.height = height;
		boundingBox.width = width;

		Pupil pupil(pupilCenter, diameter, boundingBox);

		Mat roiPupilContour;
		cvtColor(roiBin, roiPupilContour, COLOR_GRAY2BGR);
		drawContours(roiPupilContour, vector<vector<Point>>{hull}, 0, Scalar(0, 255, 0), 2);//Green

		Mat roiPupilEllipse;
		cvtColor(roiBin, roiPupilEllipse, COLOR_GRAY2BGR);
		cv::ellipse(roiPupilEllipse, ellipse, Scalar(255, 200, 0), 2);//Yellow

	 
		Mat roiBoundingBox;
		cvtColor(roiBin, roiBoundingBox, COLOR_GRAY2BGR);
		rectangle(roiBoundingBox, boundingBox, Scalar(0, 0, 255), 2);//Red

		if (isShowingImg)
		{
			imshow("roiPupilContour", roiPupilContour);
			waitKey(0);
			imshow("roiPupilEllipse", roiPupilEllipse);
			waitKey(0);
			imshow("roiBoundingBox", roiBoundingBox);
			waitKey(0);
		}
				
		if (isSavingImg)
		{
			imwrite(savePath + "roiPupilContour.jpg", roiPupilContour);
			imwrite(savePath + "roiPupilEllipse.jpg", roiPupilEllipse);
			imwrite(savePath + "roiBoundingBox.jpg", roiBoundingBox);
		}

		return pupil;
	}
}

Point2f detectP1Center(int sbjIdx, Mat proiBin, bool isShowingImg, bool isSavingImg, string savePath)
{
	vector<vector<Point>> p1Contours;
	vector<Vec4i> p1Hierarchy;
	findContours(proiBin, p1Contours, p1Hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	for (size_t i = 0; i < p1Contours.size(); i++)
	{
		double area = contourArea(p1Contours[i]);
		double perimeter = arcLength(p1Contours[i], true);

		if (perimeter <= 0) continue;

		double circularity = (4 * CV_PI * area) / (perimeter * perimeter);

		double circularityTh = 0.3;
		double areaLowTh = 15, areaHighTh = 100, perimeterLowTh = 15, perimeterHighTh = 50;

		if (sbjIdx == 2 || sbjIdx == 4 || sbjIdx == 8) 
			circularityTh = 0.0, areaHighTh = 10000, perimeterHighTh = 10000;
		
		if (circularity > circularityTh && area >= areaLowTh && area <= areaHighTh &&
			perimeter >= perimeterLowTh && perimeter <= perimeterHighTh)
		{
			vector<Point> p1Hull;
			convexHull(p1Contours[i], p1Hull);
			if (p1Hull.size() < 5)
				continue;

			RotatedRect p1Ellipse = fitEllipse(p1Hull);
			Point2f p1Center = p1Ellipse.center;

			Mat proiP1Contour;
			cvtColor(proiBin, proiP1Contour, COLOR_GRAY2BGR);
			drawContours(proiP1Contour, vector<vector<Point>>{p1Hull}, 0, Scalar(0, 255, 0), 2);//Green

			Mat proiP1Ellipse;
			cvtColor(proiBin, proiP1Ellipse, COLOR_GRAY2BGR);
			cv::ellipse(proiP1Ellipse, p1Ellipse, Scalar(255, 200, 0), 2);//Yellow
			
			Mat proiP1Center;
			cvtColor(proiBin, proiP1Center, COLOR_GRAY2BGR);
			circle(proiP1Center, p1Center, 2, Scalar(0, 0, 255), -1); //Red

			if (isShowingImg)
			{
				imshow("proiP1Contour", proiP1Contour);
				waitKey(0);
				imshow("proiP1Ellipse", proiP1Ellipse);
				waitKey(0);
				imshow("proiP1Center", proiP1Center);
				waitKey(0);
			}

			if (isSavingImg)
			{
				imwrite(savePath + "proiP1Contour.jpg", proiP1Contour);
				imwrite(savePath + "proiP1Ellipse.jpg", proiP1Ellipse);
				imwrite(savePath + "proiP1Center.jpg", proiP1Center);
			}
			
			return p1Center;
		}
	}
}

Point2f detectP4Center(int sbjIdx, Mat proi, int colCenter, int rowCenter, string tmplName, bool isShowingImg, bool isSavingImg, string savePath)
{
	Mat p4Tmpl = imread(tmplName, IMREAD_GRAYSCALE);
	uchar* p = p4Tmpl.ptr(rowCenter);
	p[colCenter] = 255;

	Mat p4Result;
	matchTemplate(proi, p4Tmpl, p4Result, TM_CCORR_NORMED);

	double maxv;
	Point maxloc;
	minMaxLoc(p4Result, 0, &maxv, 0, &maxloc);
	Point2f p4Center(maxloc.x + colCenter, maxloc.y + rowCenter);

	Mat p4MarkedProi;
	cvtColor(proi, p4MarkedProi, COLOR_GRAY2BGR);
	circle(p4MarkedProi, p4Center, 1, Scalar(0, 0, 255), -1);

	if (isShowingImg)
	{
		imshow("p4Center", p4MarkedProi);
		waitKey(0);
	}
	if (isSavingImg)
		imwrite(savePath + "p4Center.jpg", p4MarkedProi);

	return p4Center;
}

double calculateDpiDistance(Point p1, Point p4)
{
	double distance = sqrt(pow(double(p1.x - p4.x), 2) + pow(double(p1.y - p4.y), 2));
	return distance;
}

void drawCenters(Mat img, Point pupil, Point p1, Point p4)
{
	circle(img, pupil, 2, Scalar(255, 255, 255), -1);//White
	circle(img, p1, 2, Scalar(0, 255, 0), -1);//Green
	circle(img, p4, 1, Scalar(0, 0, 255), -1);//Red
}

void writeText(Mat img, Point pupil, Point p1, Point p4, int diameter, double distance)
{
	int font = FONT_HERSHEY_SIMPLEX;
	double fontSize = 0.7;
	int fontWidth = 2;
	int x = 100;
	int y = img.rows - 100;

	string diameter_txt = "pupil diameter: " + to_string(diameter);
	string pupil_txt = "pupil center : " + to_string(pupil.x) + ", " + to_string(pupil.y);
	string p1_txt = "p1 center: " + to_string(p1.x) + ", " + to_string(p1.y);
	string p4_txt = "p4 center: " + to_string(p4.x) + ", " + to_string(p4.y);
	string distance_txt = "DPI distance: " + to_string(distance);

	putText(img, diameter_txt, Point(x, y), font, fontSize, Scalar(0, 255, 0), fontWidth);
	putText(img, pupil_txt, Point(x, y - 50), font, fontSize, Scalar(0, 255, 0), fontWidth);
	putText(img, p1_txt, Point(x, y - 100), font, fontSize, Scalar(0, 255, 0), fontWidth);
	putText(img, p4_txt, Point(x, y - 150), font, fontSize, Scalar(0, 255, 0), fontWidth);
	putText(img, distance_txt, Point(x, y - 200), font, fontSize, Scalar(0, 255, 0), fontWidth);
}

void drawPupil(Mat img, Point pupil, int diameter)
{
	circle(img, pupil, diameter / 2, Scalar(0, 0, 255), 1);
	return;
}

void drawDiameter(Mat img, Point pupil, int diameter)
{
	line(img, Point(pupil.x - diameter / 2 - 2, pupil.y),
		Point(pupil.x + diameter / 2 + 2, pupil.y), Scalar(255, 0, 0), 1, 8, 0);
	return;
}

void drawBoundingBox(Mat img, Rect boundingBox)
{
	rectangle(img, boundingBox, Scalar(255, 0, 0), 2);//Blue
	return;
}

void createResultImg(Mat img, Point pupil, Point p1, Point p4, int diameter, double distance, Rect boundingBox)
{
	drawCenters(img, pupil, p1, p4);
	drawBoundingBox(img, boundingBox);
	writeText(img, pupil, p1, p4, diameter, distance);
}

void setP1Position(string filepath, int cnt, int startLine, Point origP4, Point2f& newP1Center, float& newDpiDist)
{
	ifstream file(filepath);
	if (!file.is_open()) 
		throw runtime_error("Cannot open file: " + filepath);

	string line;
	int currentLine = startLine;
	float x = 0, y = 0;

	while (getline(file, line)) {

		if (currentLine != cnt)
		{
			++currentLine;
			continue;
		}

		stringstream ss(line);
		string x_str, y_str;

		if (getline(ss, x_str, ',') && getline(ss, y_str, ',')) 
		{
			try {
				x = stof(x_str);
				y = stof(y_str);
			}
			catch (const exception& e) {
				cerr << "Coordinate transformation failed: " << e.what() << endl;
			}
		}
		else 
			cerr << "Coordinate format error: " << line << endl;

		break;
	}

	newDpiDist = calculateDpiDistance(Point(x, y), origP4);
	newP1Center = Point2f(x, y);
}

void setP4Position(string filepath, int cnt, int startLine, Point origP1, Point2f& newP4Center, float& newDpiDist)
{
	ifstream file(filepath);
	if (!file.is_open())
		throw runtime_error("Cannot open file: " + filepath);

	string line;
	int currentLine = startLine;
	float x = 0,  y = 0;

	while (getline(file, line)) 
	{
		if (currentLine != cnt)
		{
			++currentLine;
			continue;
		}

		stringstream ss(line);
		string strX, strY;

		if (getline(ss, strX, ',') && getline(ss, strY, ','))
		{
			try 
			{
				x = stof(strX);
				y = stof(strY);
			}
			catch (const exception& e) 
			{
				cerr << "Coordinate transformation failed: " << e.what() << endl;
			}
		}
		else 
			cerr << "Coordinate format error: " << line << endl;

		break;
	}

	newDpiDist = calculateDpiDistance(origP1, Point(x,y));
	newP4Center = Point2f(x, y);
}