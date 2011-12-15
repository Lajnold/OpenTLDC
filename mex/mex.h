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

#ifndef MEX_H_
#define MEX_H_

#include <Eigen/Core>
#include <opencv/cv.h>
#include <opencv/cvaux.h>

#include "../tld/structs.h"

/* lucas kanade */
void lkInit();
Eigen::Matrix<double, 4, 150> lk2(IplImage* imgI, IplImage* imgJ,
		Eigen::Matrix<double, 2, 150> const & pointsI, Eigen::Matrix<double, 2,
				150> const & pointsJ, unsigned int sizeI, unsigned int sizeJ,
		unsigned int level);

/* fern */
void fern0();
void fern1(IplImage* source,
		Eigen::Matrix<double, 6, Eigen::Dynamic> const & grid, Eigen::Matrix<
				double, 4 * TLD_NFEATURES, TLD_NTREES> const & features, Eigen::Matrix<
				double, 2, 21> const & scales);

Eigen::RowVectorXd fern2(Eigen::Matrix<double, 10, Eigen::Dynamic> const & X,
		Eigen::VectorXd const & Y, double margin, unsigned char bootstrap,
		Eigen::VectorXd const & idx);

Eigen::RowVectorXd fern3(Eigen::Matrix<double, 10, Eigen::Dynamic> const & nX2, int n);

void fern4(ImgType& img, double maxBBox, double minVar, Eigen::VectorXd& conf,
		Eigen::Matrix<double, 10, Eigen::Dynamic>& patt);

Eigen::Matrix<double, TLD_NTREES, Eigen::Dynamic> fern5(ImgType& img, std::vector<
		int>& idx, double var);

Eigen::MatrixXd
distance(Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), 1> const &x1,
		Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), TLD_MAXPATCHES> const &x2,
		int nx2, unsigned int flag);

IplImage* warp(IplImage* img, Eigen::Matrix3d const & H,
		Eigen::Vector4d const & box);

Eigen::VectorXd
bb_overlap(Eigen::Matrix<double, 4, Eigen::Dynamic> const & bb1);
Eigen::MatrixXd bb_overlap(Eigen::Matrix<double, 4, Eigen::Dynamic> const & bb, int n1,
		Eigen::Matrix<double, 4, Eigen::Dynamic> const & bb1);
Eigen::MatrixXd bb_overlap(
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> const & bb,
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> const & bb1,
		int x);
double bb_overlaphelper(Eigen::Vector4d const & bb1,
		Eigen::Vector4d const & bb2);
#endif /* MEX_H_ */
