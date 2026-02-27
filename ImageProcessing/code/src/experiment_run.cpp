#include "opencv2/opencv.hpp"
#include <iostream>
#include "string"
#include "vector"
#include "cmath"
#include "fstream"
#include "../include/image_processing.h"
#include "../include/pupil_detector.h"

using namespace std;
using namespace cv;

int run(int sbjIdx, bool isDefaultSbj, bool isShowingImg, bool isSavingImg, bool isShowingRes, bool isSavingRes)
{
	string processedImgPath = "data/processed_frames/";
	string extraOutputPath = "results/extra/sbj" + to_string(sbjIdx) + "/";
	string extraInputPath = "data/frames/extra/sbj" + to_string(sbjIdx) + "/*.jpg";

	string defOutputPath = "results/main/sbj" + to_string(sbjIdx) + "/";
	string defInputPath = "data/frames/main/sbj" + to_string(sbjIdx) + "/*.jpg";
	
	string defTmplPath = "data/templates/main/";
	string extraTmplPath = "data/templates/extra/";

	int idx = 0;

	if (sbjIdx == 1 || sbjIdx == 2) idx = 0;
	else if (sbjIdx == 5 || sbjIdx == 4) idx = 1;
	else if (sbjIdx == 7 || sbjIdx == 8) idx = 2;
	else if (sbjIdx == 10) idx = 3;
	else if (sbjIdx == 11) idx = 4;

	vector<Rect> defRoiPos = { Rect(390, 80, 400, 260),
							Rect(400, 40, 400, 260),
							Rect(380, 20, 400, 260),
							Rect(380, 0, 400, 260),
							Rect(300, 160, 400, 260) };

	vector<Rect> extraRoiPos = { Rect(360, 0, 400, 260),
						Rect(350, 0, 400, 260),
						Rect(350, 0, 400, 260) };

	vector<Size> defProiSize = { Size(150, 120),
							  Size(150, 100),
							  Size(150, 100),
							  Size(150, 100),
							  Size(150, 120) };

	vector<Size> extraProiSize = { Size(150, 170),
							  Size(150, 120),
							  Size(150, 160) };

	vector<int> defPupilTh = { 60, 80, 60, 60, 60 };
	vector<int> defPurkinje1Th = { 220, 220, 220, 220, 220 };

	vector<int> extraPupilTh = { 55, 80, 80 };
	vector<int> extraPurkinje1Th = { 220, 220, 220 };


	vector<vector<Point>> defTmplCenter = { { {4, 5}, {4, 5}, {4, 5}, {4, 5}, {4, 5},
										   {5, 6}, {4, 5}, {4, 5}, {4, 5},{4, 5} },
										 { {5, 5}, {5, 5}, {5, 5}, {5, 5}, {7, 7},
										   {5, 5}, {7, 7}, {8, 8}, {5, 5}, {5, 5} },
										 { {6, 6}, {6, 6}, {6, 6}, {6, 6}, {6, 6},
										   {6, 6}, {6, 6}, {7, 6}, {7, 6}, {6, 6} },
										 { {6, 6}, {6, 6}, {6, 6}, {6, 6}, {6, 6},
										   {6, 6}, {6, 6}, {6, 6}, {6, 6}, {6, 6}},
										 { {3, 5}, {3, 5}, {3, 5}, {3, 5}, {3, 5},
										   {3, 5}, {3, 5}, {3, 5}, {3, 5}, {3, 5} } };
	
	vector<vector<Point>> extraTmplCenter = { { {6, 5}, {6, 5}, {6, 5}, {6, 5}, {9, 9},
										   {6, 5}, {9, 9}, {9, 9}, {17, 8}, {17, 8} },
										 { {3, 4}, {3, 4}, {3, 4}, {3, 4}, {3, 4},
										   {3, 4}, {3, 4}, {3, 4}, {3, 4}, {3, 4} },
										 { {5, 5}, {5, 5}, {5, 5}, {5, 5}, {5, 5},
										   {5, 5}, {5, 5}, {5, 5}, {5, 5}, {4, 6} } };

	vector<vector<string>> defTmplIdx =  { {"1.jpg", "1.jpg", "1.jpg", "1.jpg", "1.jpg" ,
										 "1_2.jpg" , "1.jpg" , "1.jpg" , "1.jpg" ,"1.jpg" },
										{"5.jpg", "5.jpg", "5.jpg", "5.jpg", "5_2.jpg",
										 "5.jpg" , "5_2.jpg" , "5_3.jpg" , "5.jpg" , "5.jpg" },
										{"7.jpg", "7.jpg", "7.jpg", "7.jpg", "7.jpg",
										 "7.jpg" , "7.jpg" , "7_2.jpg" , "7_2.jpg" , "7.jpg" },
										{"10.jpg", "10.jpg", "10.jpg", "10.jpg", "10.jpg",
										 "10.jpg", "10.jpg", "10.jpg", "10.jpg", "10.jpg" },
										{"11.jpg", "11.jpg", "11.jpg", "11.jpg", "11.jpg",
										 "11.jpg", "11.jpg", "11.jpg", "11.jpg", "11.jpg" } };

	vector<vector<string>> extraTmplIdx = { {"2_1.jpg", "2_1.jpg", "2_1.jpg", "2_1.jpg", "2_2.jpg" ,
										"2_1.jpg" , "2_2.jpg" , "2_2.jpg" , "2_3.jpg" ,"2_3.jpg"},
									   {"4_1.jpg", "4_1.jpg", "4_1.jpg", "4_1.jpg", "4_1.jpg",
										"4_1.jpg", "4_1.jpg", "4_1.jpg", "4_1.jpg","4_1.jpg"},
									   {"8_1.jpg", "8_1.jpg", "8_1.jpg", "8_1.jpg", "8_1.jpg",
										"8_1.jpg", "8_1.jpg", "8_1.jpg", "8_1.jpg","8_2.jpg"} };

	vector<Rect> roiPos;
	vector<Size> proiSize;
	vector<int> pupilTh, purkinje1Th;
	vector<vector<Point>> tmplCenter;
	vector<vector<string>> tmplIdx;
	string outputPath, inputPath, tmplPath;

	if (isDefaultSbj)
	{
		roiPos = defRoiPos;
		proiSize = defProiSize;
		pupilTh = defPupilTh;
		purkinje1Th = defPurkinje1Th;
		tmplCenter = defTmplCenter;
		tmplIdx = defTmplIdx;
		outputPath = defOutputPath;
		inputPath = defInputPath;
		tmplPath = defTmplPath;
	}
	else
	{
		roiPos = extraRoiPos;
		proiSize = extraProiSize;
		pupilTh = extraPupilTh;
		purkinje1Th = extraPurkinje1Th;
		tmplCenter = extraTmplCenter;
		tmplIdx = extraTmplIdx;
		outputPath = extraOutputPath;
		inputPath = extraInputPath;
		tmplPath = extraTmplPath;
	}

	vector<string> inputs = readFiles(inputPath);
	if (inputs.size() == 0)
	{
		cerr << "Not found file!" << "\n";
		return -1;
	}

	vector<string> depths;
	vector<int> diameters;
	vector<double> dpiDistances;

	for (int cnt = 0; cnt < inputs.size(); cnt++)
	{
		string input = inputs[cnt];
		input.replace(input.find("\\"), 1, "/");

		Mat image = imread(input, IMREAD_GRAYSCALE);
		Mat img = image.clone();
		Mat roi = img(roiPos[idx]).clone();

		//1
		Mat roiBlur;
		blur(roi, roiBlur, Size(5, 5));//mean

		Mat roiBin;
		threshold(roiBlur, roiBin, pupilTh[idx], 255, THRESH_BINARY);

		Pupil pupil = detectPupil(roiBin, sbjIdx, isSavingImg, isShowingImg, processedImgPath);


		//2
		Point proiPos( pupil.center.x - proiSize[idx].width / 2,
					  pupil.center.y - proiSize[idx].height / 2 );

		if (sbjIdx == 2 || sbjIdx == 8) proiPos.y = 0;

		Mat proi = roi(Rect(proiPos, proiSize[idx])).clone();
		Mat proiBin;
		threshold(proi, proiBin, purkinje1Th[idx], 255, THRESH_BINARY);

		Point2f p1Center = detectP1Center(sbjIdx, proiBin, isShowingImg, isSavingImg, processedImgPath);


		//3
		int colCenter = tmplCenter[idx][cnt / 5].x;
		int rowCenter = tmplCenter[idx][cnt / 5].y;
		string tmplName = tmplPath + tmplIdx[idx][cnt / 5];

		Point2f p4Center = detectP4Center(sbjIdx, proi, colCenter, rowCenter, tmplName, isShowingImg, isSavingImg, processedImgPath);


		int pupilX = pupil.center.x + roiPos[idx].x;
		int pupilY = pupil.center.y + roiPos[idx].y;
		int p1X = p1Center.x + roiPos[idx].x + proiPos.x;
		int p1Y = p1Center.y + roiPos[idx].y + proiPos.y;
		int p4X = p4Center.x + roiPos[idx].x + proiPos.x;
		int p4Y = p4Center.y + roiPos[idx].y + proiPos.y;

		Point origPupilCenter(pupilX, pupilY);
		Point origP1Center(p1X, p1Y);
		Point origP4Center(p4X, p4Y);

		int boxX = pupil.boundingBox.x + roiPos[idx].x;
		int boxY = pupil.boundingBox.y + roiPos[idx].y;
		Rect pupilBoundingBox(boxX, boxY, pupil.boundingBox.width, pupil.boundingBox.height);

		double dpiDist = calculateDpiDistance(origP1Center, origP4Center);

		if (sbjIdx == 4)
		{
			Point2f newP4Center;
			float newDpiDist;
			setP4Position("data/position/sbj4_p4_position.txt", cnt, 0, origP1Center, newP4Center, newDpiDist);
			origP4Center = newP4Center;
			dpiDist = newDpiDist;
		}
		if (sbjIdx == 8 && cnt >= 35)
		{
			Point2f newP1Center;
			float newDpiDist;
			setP1Position("data/position/sbj8_p1_position.txt", cnt, 35, origP4Center, newP1Center, newDpiDist);
			origP1Center = newP1Center;
			dpiDist = newDpiDist;
		}
		if (sbjIdx == 2)
		{
			if (cnt >= 15)
			{
				Point2f newP1Center;
				float newDpiDist;
				setP1Position("data/position/sbj2_p1_position.txt", cnt, 0, origP4Center, newP1Center, newDpiDist);
				origP1Center = newP1Center;
				dpiDist = newDpiDist;
			}
			if (cnt >= 40)
			{
				Point2f newP4Center;
				float newDpiDist;
				setP4Position("data/position/sbj2_p4_position.txt", cnt, 0, origP1Center, newP4Center, newDpiDist);
				origP4Center = newP4Center;
				dpiDist = newDpiDist;
			}
		}

		string depth = input.substr(input.size() - 9, 5);

		depths.push_back(depth);
		dpiDistances.push_back(dpiDist);
		diameters.push_back(pupil.diameter);

		if (isShowingImg)
		{
			imshow("roi", roi);
			waitKey(0);

			imshow("roiBlur", roiBlur);
			waitKey(0);

			imshow("roiBin", roiBin);
			waitKey(0);

			imshow("proi", proi);
			waitKey(0);

			imshow("proiBin", proiBin);
			waitKey(0);
		}

		if (isSavingImg)
		{
			imwrite(processedImgPath + "roi.jpg", roi);
			imwrite(processedImgPath + "roiBlur.jpg", roiBlur);
			imwrite(processedImgPath + "roiBin.jpg", roiBin);
			imwrite(processedImgPath + "proi.jpg", proi);
			imwrite(processedImgPath + "proiBin.jpg", proiBin);
		}

		Mat resultImg;
		cvtColor(image, resultImg, COLOR_GRAY2BGR);
		createResultImg(resultImg, origPupilCenter, origP1Center, origP4Center, pupil.diameter, dpiDist, pupilBoundingBox);

		if (isShowingRes)
		{
			imshow("resultImg", resultImg);
			waitKey(0);
		}
		if (isSavingRes)
		{
			string imgFilename = depth + ".jpg";
			imwrite(outputPath + imgFilename, resultImg);
		}

		destroyAllWindows();
	}

	if (isSavingRes)
	{
		string txtFilename = "result_sbj" + to_string(sbjIdx) + ".txt";
		ofstream fout(outputPath + txtFilename);
		fout << "피험자,시선깊이(cm),DPI거리(p1-p4),동공크기" << "\n";

		for (int idx = 0; idx < dpiDistances.size(); ++idx)
			fout << sbjIdx << ", " << depths[idx] << ", " << dpiDistances[idx] << ", " << diameters[idx] << "\n";
		fout.close();
	}

	return 0;
}