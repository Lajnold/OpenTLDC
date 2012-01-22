/**
 * OpenTLDC is an algorithm for tracking of unknown objects
 * in unconstrained video streams. It is based on TLD,
 * published by Zdenek Kalal
 * (see http://info.ee.surrey.ac.uk/Personal/Z.Kalal/tld.html).
 *
 * Copyright (C) 2011 Sascha Schrader, Stefan Brending, Adrian Block
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <iostream>
#include <limits>
#ifdef _CHAR16T
#define CHAR16_T
#endif
#include "mex.h"

using namespace tld;

static const int WIN_SIZE = 4;

void euclideanDistance(cv::Point2f *point1, cv::Point2f *point2,
		float *match, int nPts) {

	for (int i = 0; i < nPts; i++) {

		match[i] = sqrt((point1[i].x - point2[i].x) * (point1[i].x
				- point2[i].x) + (point1[i].y - point2[i].y) * (point1[i].y
				- point2[i].y));

	}
}

void normCrossCorrelation(const cv::Mat &prevImg, const cv::Mat &currImg,
		cv::Point2f *points0, cv::Point2f *points1, int nPts, unsigned char *status,
		float *match, int winsize, int method) {

	cv::Mat rec0;
	cv::Mat rec1;
	cv::Mat res;

	for (int i = 0; i < nPts; i++) {
		if (status[i] == 1) {
			cv::getRectSubPix(prevImg, cv::Size(winsize, winsize), points0[i], rec0);
			cv::getRectSubPix(currImg, cv::Size(winsize, winsize), points1[i], rec1);
			cv::matchTemplate(rec0, rec1, res, method);
			match[i] = ((float *) (res.data))[0];

		} else {
			match[i] = 0.0;
		}
	}
}

// Lucas-Kanade
Eigen::Matrix<double, 4, 150> lk2(TldStruct &tld, const cv::Mat &prevImg, const cv::Mat &currImg,
		Eigen::Matrix<double, 2, 150> const & pointsI, Eigen::Matrix<double, 2, 150> const & pointsJ,
		unsigned int sizeI, unsigned int sizeJ, unsigned int level) {

	double nan = std::numeric_limits<double>::quiet_NaN();

	int Level;
	if (level != 0) {
		Level = (int) level;
	} else {
		Level = 5;
	}

	int I = 0;
	int J = 1;
	int Winsize = 10;

	// Points
	int nPts = sizeI;

	if (nPts != sizeJ) {
		std::cout << "Inconsistent input!" << std::endl;
		return Eigen::MatrixXd::Zero(1, 1);
	}

	std::vector<std::vector<cv::Point2f> > points;
	points.push_back(std::vector<cv::Point2f>(nPts));
	points.push_back(std::vector<cv::Point2f>(nPts));
	points.push_back(std::vector<cv::Point2f>(nPts));

	for (int i = 0; i < nPts; i++) {
		points[0][i].x = pointsI(0, i);
		points[0][i].y = pointsI(1, i);
		points[1][i].x = pointsJ(0, i);
		points[1][i].y = pointsJ(1, i);
		points[2][i].x = pointsI(0, i);
		points[2][i].y = pointsI(1, i);
	}

	std::vector<float> ncc(nPts);
	std::vector<float> fb(nPts);
	std::vector<unsigned char> status1;
	std::vector<unsigned char> status2;
	std::vector<float> err1;
	std::vector<float> err2;

	cv::calcOpticalFlowPyrLK(
		prevImg, currImg, points[0], points[1],
		status1, err1, cv::Size(WIN_SIZE, WIN_SIZE), Level,
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.03),
		0.0, 0);

	cv::calcOpticalFlowPyrLK(
		currImg, prevImg, points[1], points[0],
		status2, err2, cv::Size(WIN_SIZE, WIN_SIZE), Level,
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.03),
		0.0, cv::OPTFLOW_USE_INITIAL_FLOW);

	normCrossCorrelation(prevImg, currImg, points[0].data(), points[1].data(), nPts,
			status1.data(), ncc.data(), Winsize, CV_TM_CCOEFF_NORMED);

	euclideanDistance(points[0].data(), points[2].data(), fb.data(), nPts);

	// Output
	int M = 4;
	Eigen::MatrixXd output(M, 150);
	for (int i = 0; i < nPts; i++) {
		if (status1[i] == 1) {
			output(0, i) = (double) points[1][i].x;
			output(1, i) = (double) points[1][i].y;
			output(2, i) = (double) fb[i];
			output(3, i) = (double) ncc[i];
		} else {
			output(0, i) = nan;
			output(1, i) = nan;
			output(2, i) = nan;
			output(3, i) = nan;
		}
	}

	return output;
}
