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

const int MAX_COUNT = 500;
const int MAX_IMG = 2;
int win_size = 4;
CvPoint2D32f* points[3] = { 0, 0, 0 };
static IplImage **IMG = 0;
static IplImage **PYR = 0;

void euclideanDistance(CvPoint2D32f *point1, CvPoint2D32f *point2,
		float *match, int nPts) {

	for (int i = 0; i < nPts; i++) {

		match[i] = sqrt((point1[i].x - point2[i].x) * (point1[i].x
				- point2[i].x) + (point1[i].y - point2[i].y) * (point1[i].y
				- point2[i].y));

	}
}

void normCrossCorrelation(IplImage *imgI, IplImage *imgJ,
		CvPoint2D32f *points0, CvPoint2D32f *points1, int nPts, char *status,
		float *match, int winsize, int method) {

	IplImage *rec0 = cvCreateImage(cvSize(winsize, winsize), 8, 1);
	IplImage *rec1 = cvCreateImage(cvSize(winsize, winsize), 8, 1);
	IplImage *res = cvCreateImage(cvSize(1, 1), IPL_DEPTH_32F, 1);

	for (int i = 0; i < nPts; i++) {
		if (status[i] == 1) {
			cvGetRectSubPix(imgI, rec0, points0[i]);
			cvGetRectSubPix(imgJ, rec1, points1[i]);
			cvMatchTemplate(rec0, rec1, res, method);
			match[i] = ((float *) (res->imageData))[0];

		} else {
			match[i] = 0.0;
		}
	}
	cvReleaseImage(&rec0);
	cvReleaseImage(&rec1);
	cvReleaseImage(&res);

}

// Lucas-Kanade
void lkInit() {
	if (IMG != 0 && PYR != 0) {

		for (int i = 0; i < MAX_IMG; i++) {
			cvReleaseImage(&(IMG[i]));
			IMG[i] = 0;
			cvReleaseImage(&(PYR[i]));
			PYR[i] = 0;
		}
		free(IMG);
		IMG = 0;
		free(PYR);
		PYR = 0;
	}

	IMG = (IplImage**) calloc(MAX_IMG, sizeof(IplImage*));
	PYR = (IplImage**) calloc(MAX_IMG, sizeof(IplImage*));

	return;
}

// Lucas-Kanade
Eigen::Matrix<double, 4, 150> lk2(IplImage* imgI, IplImage* imgJ, Eigen::Matrix<double, 2,
		150> const & pointsI, Eigen::Matrix<double, 2, 150> const & pointsJ,
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

	// Images
	if (IMG[I] != 0) {
		IMG[I] = imgI;
	} else {
		CvSize imageSize = cvGetSize(imgI);
		IMG[I] = cvCreateImage(imageSize, 8, 1);
		PYR[I] = cvCreateImage(imageSize, 8, 1);
		IMG[I] = imgI;
	}

	if (IMG[J] != 0) {
		IMG[J] = imgJ;
	} else {
		CvSize imageSize = cvGetSize(imgJ);
		IMG[J] = cvCreateImage(imageSize, 8, 1);
		PYR[J] = cvCreateImage(imageSize, 8, 1);
		IMG[J] = imgJ;
	}

	// Points
	int nPts = sizeI;

	if (nPts != sizeJ) {
		std::cout << "Inconsistent input!" << std::endl;
		return Eigen::MatrixXd::Zero(1, 1);
	}

	points[0] = (CvPoint2D32f*) cvAlloc(nPts * sizeof(CvPoint2D32f)); // template
	points[1] = (CvPoint2D32f*) cvAlloc(nPts * sizeof(CvPoint2D32f)); // target
	points[2] = (CvPoint2D32f*) cvAlloc(nPts * sizeof(CvPoint2D32f)); // forward-backward

	for (int i = 0; i < nPts; i++) {
		points[0][i].x = pointsI(0, i);
		points[0][i].y = pointsI(1, i);
		points[1][i].x = pointsJ(0, i);
		points[1][i].y = pointsJ(1, i);
		points[2][i].x = pointsI(0, i);
		points[2][i].y = pointsI(1, i);
	}

	float *ncc = (float*) cvAlloc(nPts * sizeof(float));
	float *fb = (float*) cvAlloc(nPts * sizeof(float));
	char *status = (char*) cvAlloc(nPts);

	cvCalcOpticalFlowPyrLK(IMG[I], IMG[J], PYR[I], PYR[J], points[0],
			points[1], nPts, cvSize(win_size, win_size), Level, status, 0,
			cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03),
			CV_LKFLOW_INITIAL_GUESSES);
	cvCalcOpticalFlowPyrLK(IMG[J], IMG[I], PYR[J], PYR[I], points[1],
			points[2], nPts, cvSize(win_size, win_size), Level, 0, 0,
			cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03),
			CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY
					| CV_LKFLOW_PYR_B_READY );

	normCrossCorrelation(IMG[I], IMG[J], points[0], points[1], nPts, status,
			ncc, Winsize, CV_TM_CCOEFF_NORMED);
	euclideanDistance(points[0], points[2], fb, nPts);

	// Output
	int M = 4;
	Eigen::MatrixXd output(M, 150);
	for (int i = 0; i < nPts; i++) {
		if (status[i] == 1) {
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
