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
#include <limits>
#include "../img/img.h"
#include "../utils/utility.h"

/**
 * Initialize all structures.
 *
 * @param tld learning structures
 */
void tldInit(TldStruct& tld) {

	// initialize lucas kanade
	lkInit();

	// Get initial image
	tld.cfg->imgsource->nextImage();
	tld.currentImg.input = tld.cfg->imgsource->getGrayImage();
	tld.currentImg.blur = img_blur(tld.currentImg.input);

	// get initial bounding box
	Eigen::Vector4d bb;
	bb = tld.cfg->init;

	Eigen::Vector2i imsize;
	imsize(0) = tld.currentImg.input.width();
	imsize(1) = tld.currentImg.input.height();
	bb_scan(tld, bb, imsize, tld.model->min_win);

	// Features
	tldGenerateFeatures(tld, tld.model->num_trees, tld.model->num_features);

	// Initialize Detector
	fern0();

	// allocate structures
	fern1(tld.currentImg.input, tld.grid, tld.features, tld.scales);

	// Temporal structures
	Tmp temporal;
	temporal.conf = Eigen::VectorXd::Zero(tld.nGrid);
	temporal.patt = Eigen::Matrix<double, 10, Eigen::Dynamic>::Zero(tld.model->num_trees, tld.nGrid);
	tld.tmp = temporal;

	// RESULTS =================================================================

	// Initialize Trajectory

	tld.prevBB = Eigen::Vector4d::Constant(
			std::numeric_limits<double>::quiet_NaN());
	tld.currentBB = bb;
	tld.conf = 1;
	tld.currentValid = 1;
	tld.size = 1;

	// TRAIN DETECTOR ==========================================================

	Eigen::RowVectorXd overlap = bb_overlap(tld.currentBB, tld.nGrid, tld.grid.topRows(4));

	tld.target = img_patch(tld.currentImg.input, tld.currentBB);

	// Generate Positive Examples
	Eigen::Matrix<double, NTREES, Eigen::Dynamic> pX; // pX: 10 rows
	Eigen::Matrix<double, (PATCHSIZE * PATCHSIZE), Eigen::Dynamic> pEx;
	tld.currentBB = tldGeneratePositiveData(tld, overlap, tld.currentImg,
			tld.p_par_init, pX, pEx);

	Eigen::MatrixXd pY = Eigen::MatrixXd::Ones(1, pX.cols());
	// Generate Negative Examples
	Eigen::Matrix<double, NTREES, Eigen::Dynamic> nX; // nX: 10 rows
	Eigen::Matrix<double, (PATCHSIZE * PATCHSIZE), Eigen::Dynamic> nEx;
	tldGenerateNegativeData(tld, overlap, tld.currentImg, nX, nEx);

	// Split Negative Data to Training set and Validation set
	Eigen::Matrix<double, NTREES, Eigen::Dynamic> spnX;
	Eigen::Matrix<double, (PATCHSIZE * PATCHSIZE), Eigen::Dynamic> spnEx;
	tldSplitNegativeData(nX, nEx, spnX, spnEx);

	Eigen::MatrixXd nY1 = Eigen::MatrixXd::Zero(1, spnX.cols() / 2);

	Eigen::MatrixXd xCombined(pX.rows(), pX.cols() + spnX.cols() / 2);
	xCombined << pX, spnX.leftCols(spnX.cols() / 2);
	Eigen::MatrixXd yCombined(pY.rows(), pY.cols() + nY1.cols());
	yCombined << pY, nY1;
	Eigen::RowVectorXd idx(xCombined.cols());
	for (int i = 0; i < xCombined.cols(); i++)
		idx(i) = i;

	idx = permutate_cols(idx);

	Eigen::MatrixXd permX(xCombined.rows(), xCombined.cols());
	Eigen::VectorXd permY(yCombined.cols());
	for (int i = 0; i < idx.cols(); i++) {
		permX.col(i) = xCombined.col(idx(i));
		permY(i) = yCombined(0, idx(i));
	}

	// Train using training set ------------------------------------------------

	// Fern
	unsigned char bootstrap = 2;
	Eigen::VectorXd dummy(1);
	dummy(0) = -1;
	fern2(permX, permY, tld.model->thr_fern, bootstrap, dummy);

	// Nearest Neighbour
	tld.npex = 0;
	tld.nnex = 0;

	tldTrainNN(pEx, spnEx.leftCols(spnEx.cols() / 2), tld);
	tld.model->num_init = tld.npex;

	// Estimate thresholds on validation set  ----------------------------------

	// Fern
	unsigned int ferninsize = spnX.cols() / 2;
	Eigen::RowVectorXd conf_fern(ferninsize);
	Eigen::Matrix<double, 10, Eigen::Dynamic> fernin(10, ferninsize);
	fernin.leftCols(ferninsize) = spnX.rightCols(ferninsize);
	conf_fern = fern3(fernin, ferninsize);
	tld.model->thr_fern = std::max(conf_fern.maxCoeff() / tld.model->num_trees,
			tld.model->thr_fern);

	// Nearest neighbor
	Eigen::MatrixXd conf_nn(3, 3);
	conf_nn = tldNN(spnEx.rightCols(spnEx.cols() / 2), tld);

	tld.model->thr_nn = std::max(tld.model->thr_nn, conf_nn.block(0, 0, 1,
			conf_nn.cols() / 3).maxCoeff());
	tld.model->thr_nn_valid = std::max(tld.model->thr_nn_valid,
			tld.model->thr_nn);
}

