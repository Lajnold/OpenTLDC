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

#ifndef TLD_H_
#define TLD_H_

#include <iostream>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "structs.h"

using namespace cv;

#define PI 3.14159265358979L

/* Main Loop */
void tldExample(TldStruct* opt, Config& cfg);

/* Shows Results in additional window */
void tldDisplay(int i, unsigned long index, TldStruct& tld, double fps);

/* Detects learned patches */
Eigen::Matrix<double, 20, 1> tldDetection(TldStruct& tld, int i, Eigen::Matrix<
		double, 4, 20>& dBB, int& n);

/* measures initial structures */
void tldInit(TldStruct& tld/*, CamImage& source, Person& persondetect*/);

/* random features */
void tldGenerateFeatures(TldStruct& tld, unsigned int nTREES,
		unsigned int nFEAT);

/* main method, is called on each loop */
void tldProcessFrame(TldStruct& tld, unsigned long i);

/* tracks bounding box with lucas kanade */
Eigen::VectorXd tldTracking(TldStruct& tld, Eigen::VectorXd const & bb, int i,
		int j);

/* duplicates slightly altered previous found positive patches */
Eigen::Vector4d tldGeneratePositiveData(TldStruct& tld,
		Eigen::MatrixXd const & overlap, ImgType& img, p_par& p_par,
		Eigen::Matrix<double, NTREES, Eigen::Dynamic>& pX, Eigen::Matrix<
				double, (PATCHSIZE * PATCHSIZE), Eigen::Dynamic>& pEx);

/* pickups bbox and converts to Eigen matrix */
Eigen::Matrix<double, (PATCHSIZE * PATCHSIZE), Eigen::Dynamic> tldGetPattern(
		ImgType& img, Eigen::Matrix<double, 4, Eigen::Dynamic> const & bb,
		Patchsize& patchsize, unsigned int flip);

/* generates initial some random negative patches */
void tldGenerateNegativeData(TldStruct& tld,
		Eigen::RowVectorXd const & overlap, ImgType& img, Eigen::Matrix<double,
				NTREES, Eigen::Dynamic>& nX, Eigen::Matrix<double, (PATCHSIZE
				* PATCHSIZE), Eigen::Dynamic>& nEx);

/* random permutation of generated negatives and splits it to validation and training set */
void tldSplitNegativeData(
				Eigen::Matrix<double, NTREES, Eigen::Dynamic> const & nX,
				Eigen::Matrix<double, PATCHSIZE * PATCHSIZE, Eigen::Dynamic> const & nEx,
				Eigen::Matrix<double, NTREES, Eigen::Dynamic>& spnX,
				Eigen::Matrix<double, PATCHSIZE * PATCHSIZE, Eigen::Dynamic>& spnEx);

/* Converts an image to Eigen Matrix */
Eigen::Matrix<double, PATCHSIZE * PATCHSIZE, 1> tldPatch2Pattern(
		CvImage patch, Patchsize const& patchsize);

/* Trains nearest neighbor */
void tldTrainNN(
				Eigen::Matrix<double, PATCHSIZE * PATCHSIZE, Eigen::Dynamic> const & pEx,
				Eigen::Matrix<double, PATCHSIZE * PATCHSIZE, Eigen::Dynamic> const & nEx1,
				TldStruct& tld);

/* Classifies examples as positive or negative */
Eigen::Matrix<double, 3, Eigen::Dynamic> tldNN(Eigen::Matrix<double, PATCHSIZE
		* PATCHSIZE, Eigen::Dynamic> const & nEx2, TldStruct& tld);

/* Shows positive examples */
void embedPex(CvImage img, TldStruct& tld);

/* Shows negative examples */
void embedNex(CvImage img, TldStruct& tld);

/* Learns detected pattern */
void tldLearning(TldStruct& tld, unsigned long I);

#endif /* TLD_H_ */

