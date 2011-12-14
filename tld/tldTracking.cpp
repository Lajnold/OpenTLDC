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

#include "tld.h"
#include "../bbox/bbox.h"
#include "../mex/mex.h"
#include "../utils/median.h"
#include <limits>

Eigen::VectorXd tldTracking(TldStruct& tld, Eigen::VectorXd const & bb, int i, int j) {
	//BB2    = []; % estimated bounding
	//Conf   = []; % confidence of prediction
	//Valid  = 0;  % is the predicted bounding box valid? if yes, learning will take place ...

	Eigen::Vector4d bb2;
	Eigen::MatrixXd Conf;
	double valid = 0;

	bb2 = Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());

	if (!bb_isdef(bb))
		return bb;

	Eigen::MatrixXd xFI(2, 150);
	int xFISize = 0;
	//generate 10x10 points on bb
	xFI = bb_points(bb, 10, 10, 5, xFISize);
	Eigen::MatrixXd xFJ(4, 150);
	//get all reliable points
	xFJ = lk2(tld.prevImg.input, tld.currentImg.input, xFI, xFI, xFISize, xFISize, 0);

	double medFB = median(xFJ.leftCols(xFISize).row(2));

	double medNCC = median(xFJ.leftCols(xFISize).row(3));

	int counter = 0;
	//get indexes of reliable points
	Eigen::VectorXd idxF(1);
	Eigen::VectorXd idxFbak(1);
	for (int n = 0; n < xFISize; n++) {
		if (xFJ(2, n) <= medFB && xFJ(3, n) >= medNCC) {
			if (counter == 0) {
				idxF(0) = n;
			} else {
				idxFbak.resize(idxF.size());
				idxFbak = idxF;

				idxF.resize(idxF.size() + 1);
				idxF.topRows(counter) = idxFbak;
				idxF(counter) = n;
			}
			counter++;
		}
	}

	Eigen::MatrixXd xFInew(xFI.rows(), idxF.rows());

	for (int f = 0; f < idxF.rows(); f++)
		xFInew.col(f) = xFI.col(idxF(f));

	Eigen::MatrixXd xFJnew(2, idxF.rows());

	for (int k = 0; k < idxF.rows(); k++)
		xFJnew.col(k) = xFJ.block(0, idxF(k), 2, 1);

	//predict the bounding box
	bb2 = bb_predict(bb, xFInew, xFJnew);

	//bounding box out of image
	Eigen::Vector2i imgsize;
	imgsize(0) = tld.currentImg.input.width();
	imgsize(1) = tld.currentImg.input.height();
	if (!bb_isdef(bb2) || bb_isout(bb2, imgsize)) {
		bb2 = Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());
		return bb2;
	}
	//too unstable predictions
	if (tld.control.maxbbox > 0 && medFB > 10) {
		bb2 = Eigen::Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());
		return bb2;
	}
	//sample patch in current image
	Eigen::MatrixXd patchJ = tldGetPattern(tld.currentImg, bb2,
			tld.model->patchsize, 0);
	Conf = tldNN(patchJ, tld);
	valid = tld.prevValid;

	unsigned int confLen = Conf.cols() / 3;
	double consSim;
	consSim = Conf(0, confLen);
	//tracking takes place
	if (consSim > tld.model->thr_nn_valid)
		valid = 1;

	tld.currentBB = bb2;

	tld.conf = consSim;

	tld.currentValid = valid;

	return bb2;
}

