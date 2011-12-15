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
#include "../mex/mex.h"
#include <limits>
#include "../img/img.h"
#include "../utils/utility.h"

/**
 * Learns new found positive and negative patches from tldDetection.
 *
 * @param tld learned structures
 * @param I index of image
 */
void tldLearning(TldStruct& tld, unsigned long I) {

	Eigen::Vector4d bb;
	bb = tld.currentBB;
	ImgType img = tld.currentImg;

	// get pattern of current bbox
	Eigen::MatrixXd pPatt = tldGetPattern(img, bb, tld.model.patchsize, 0);

	// measure nearest neighbor
	Eigen::MatrixXd nn = tldNN(pPatt, tld);

	if (nn(0, 0) < 0.5) {
		std::cout << "Fast change" << std::endl;
		tld.currentValid = 0;
		return;
	}

	unsigned int patRows = pPatt.rows();
	Eigen::VectorXd pPattVec(patRows);
	pPattVec = pPatt.col(0);

	if (variance(pPattVec, patRows) < tld.var) {
		std::cout << "Low variance" << std::endl;
		tld.currentValid = 0;
		return;
	}

	if (nn(2, 2) == 1) {
		std::cout << "In negative data" << std::endl;
		tld.currentValid = 0;
		return;
	}

	// measure overlap of the current bounding box with the bounding boxes on the grid
	Eigen::MatrixXd overlap = bb_overlap(bb, tld.nGrid,
			tld.grid.topRows(4));

	// generate positive examples from all bounding boxes that are highly
	// overlapping with current bounding box
	Eigen::Matrix<double, NTREES, Eigen::Dynamic> pX; // pX: 10 rows
	Eigen::Matrix<double, (PATCHSIZE * PATCHSIZE), Eigen::Dynamic> pEx;
	Eigen::Vector4d ret = tldGeneratePositiveData(tld, overlap, img,
			tld.p_par_update, pX, pEx);

	// labels of the positive patches
	Eigen::VectorXd pY = Eigen::VectorXd::Ones(pX.cols());

	// get indexes of negative bounding boxes on the grid (bounding boxes on the grid
	// that are far from current bounding box and which confidence was larger than 0)
	std::vector<unsigned int> idx;
	unsigned int len = overlap.cols();
	for (unsigned int k = 0; k < len; k++)
		if (overlap(k) < tld.n_par.overlap && tld.tmp.conf(k) >= 1)
			idx.push_back(k);

	// measure overlap of the current bounding box with detections
	overlap = bb_overlap(bb, tld.dt.nbb, tld.dt.bb);

	// get negative patches that are far from current bounding box
	len = overlap.cols();
	std::vector<unsigned int> idxOverlap;
	for (unsigned int k = 0; k < len; k++)
		if (overlap(k) < tld.n_par.overlap)
			idxOverlap.push_back(k);

	unsigned int ovpSize = idxOverlap.size();

	Eigen::MatrixXd nEx(tld.dt.patch.rows(), ovpSize);
	for (unsigned int k = 0; k < ovpSize; k++)
		nEx.col(k) = tld.dt.patch.col(idxOverlap[k]);

	// update the Ensemble Classifier (reuses the computation made by detector)
	unsigned int pXcols = pX.cols();
	Eigen::MatrixXd X(pX.rows(), pXcols + idx.size());
	X.leftCols(pXcols) = pX;
	for (unsigned int k = 0; k < idx.size(); k++)
		X.col(pXcols + k) = tld.tmp.patt.col(idx[k]);

	Eigen::VectorXd pY2 = Eigen::VectorXd::Zero(idx.size());
	Eigen::VectorXd Y(pY.size() + idx.size());
	Y << pY, pY2;

	Eigen::VectorXd dummy(1);
	dummy(0) = -1;

	fern2(X, Y, tld.model.thr_fern, 2, dummy);

	// update nearest neighbour
	tldTrainNN(pEx, nEx, tld);

}
