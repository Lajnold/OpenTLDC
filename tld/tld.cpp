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

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include <Eigen/Core>
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include "tld.h"
#include "tldconst.h"
#include "../bbox/bbox.h"
#include "../img/img.h"
#include "../mex/mex.h"
#include "../utils/median.h"
#include "../utils/utility.h"

using namespace Eigen;

/**
 * Used to get sorted indices
 */
struct Indices {
	double conf;
	int index;

	Indices() :
		conf(0), index(0) {
	}
	Indices(double con, int idx) :
		conf(con), index(idx) {
	}

	static bool compare(const Indices& a, const Indices& b) {
		return a.conf > b.conf;
	}
};

Matrix<double, 20, 1> tldDetection(TldStruct& tld, int i, Matrix<double, 4, 20>& dBB, int& n);

Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), Dynamic> tldGetPattern(ImgType& img, Matrix<double, 4, Dynamic> const & bb, Patchsize& patchsize, unsigned int flip);

void tldGenerateFeatures(TldStruct& tld, unsigned int nTREES, unsigned int nFEAT);

void tldGenerateNegativeData(TldStruct& tld, RowVectorXd const & overlap, ImgType& img, Matrix<double, TLD_NTREES, Dynamic>& nX, Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), Dynamic>& nEx);

Vector4d tldGeneratePositiveData(TldStruct& tld, MatrixXd const & overlap, ImgType& img, P_par& p_par, Matrix<double, TLD_NTREES, Dynamic>& pX, Matrix< double, (TLD_PATCHSIZE * TLD_PATCHSIZE), Dynamic>& pEx);

void tldLearning(TldStruct& tld, unsigned long I);

Matrix<double, 3, Dynamic> tldNN(Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Dynamic> const & nEx2, TldStruct& tld);

Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, 1> tldPatch2Pattern(CvImage patch, Patchsize const& patchsize); 

void tldSplitNegativeData(Matrix<double, TLD_NTREES, Dynamic> const & nX, Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Dynamic> const & nEx, Matrix<double, TLD_NTREES, Dynamic>& spnX, Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Dynamic>& spnEx);

VectorXd tldTracking(TldStruct& tld, VectorXd const & bb, int i, int j);

void tldTrainNN(Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Dynamic> const & pEx, Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Dynamic> const & nEx1, TldStruct& tld);


/**
 * Puts all positive patches on image.
 *
 * @param img output image
 * @param tld learned structures
 */
static void embedPex(CvImage img, TldStruct& tld) {


	double rescale = tld.cfg.plot.patch_rescale;

	// measure number of possible rows of patches
	int nrow = floor(tld.currentImg.input.height() / (rescale * tld.model.patchsize.y));

	// measure number of possible columns of patches
	int ncol = floor(tld.currentImg.input.width() / (rescale * tld.model.patchsize.x));

	// get prepared Eigen Matrix
	MatrixXd pex;
	if (tld.npex > nrow * ncol) { // max nrow * ncol
		pex = mat2img(tld.pex.leftCols(nrow * ncol), nrow * ncol, nrow);
	} else {
		pex = mat2img(tld.pex, tld.npex, nrow);
	}

	int pH = pex.rows();
	int pW = pex.cols();

	// include in output image
	for (int y = 0; y < pH; y++)
		for (int x = img.width() - pW; x < img.width(); x++) {
			((uchar*) (img.data() + img.step() * (y)))[x] = 255 * pex(
					y, x - (img.width() - pW));

		}
}

/**
 * Puts all negative patches on image.
 *
 * @param img output image
 * @param tld learned structures
 */
static void embedNex(CvImage img, TldStruct& tld) {

	double rescale = tld.cfg.plot.patch_rescale;

	// measure number of possible rows of patches
	int nrow = floor(tld.currentImg.input.height() / (rescale * tld.model.patchsize.y));

	// measure number of possible columns of patches
	int ncol = floor(tld.currentImg.input.width() / (rescale * tld.model.patchsize.x));

	// get prepared Eigen Matrix
	MatrixXd nex;
	if (tld.nnex > nrow * ncol) { // max nrow * ncol
		nex = mat2img(tld.nex.leftCols(nrow * ncol), nrow * ncol, nrow);
	} else {
		nex = mat2img(tld.nex, tld.nnex, nrow);
	}

	int pH = nex.rows();
	int pW = nex.cols();

	// include in output image
	for (int y = 0; y < pH; y++)
		for (int x = 0; x < pW; x++) {
			((uchar*) (img.data() + img.step() * (y)))[x]
					= 255 * nex(y, x);
		}
}

/**
 * Detects new patches with nearest Neighbor and returns
 * a number of confidential bounding boxes.
 *
 * @param tld learned structure
 * @param i index
 * @param dBB reference to confidential bounding boxes
 * @return confidential values
 *
 */
Matrix<double, 20, 1> tldDetection(TldStruct& tld, int i, Matrix<
		double, 4, 20>& dBB, int& n) {

	dBB = Matrix<double, 4, 20>::Constant(4, 20, std::numeric_limits<
			double>::quiet_NaN());

	Matrix<double, 20, 1> confi = Matrix<double, 20, 1>::Zero(20);

	// evaluates Ensemble Classifier: saves sum of posteriors to 'tld.tmp.conf', saves
	// measured codes to 'tld.tmp.patt'
	fern4(tld.currentImg, tld.control.maxbbox, tld.var, tld.tmp.conf,
			tld.tmp.patt);

	// get indexes of bounding boxes that passed through the Ensemble Classifier
	std::vector<int> idx_dt;
	for (int j = 0; j < tld.tmp.conf.size(); j++) {
		if (tld.tmp.conf(j) > tld.model.num_trees * tld.model.thr_fern)
			idx_dt.push_back(j);
	}

	// Get max 100 bounding boxes with best confidence
	if (idx_dt.size() > 100) {
		std::vector<Indices> idx;
		for (unsigned int i = 0; i < idx_dt.size(); i++) {
			Indices id(tld.tmp.conf(idx_dt[i]), idx_dt[i]);
			idx.push_back(id);
		}
		std::sort(idx.begin(), idx.end(), Indices::compare);
		idx_dt.clear();
		for (int j = 0; j < 100; j++)
			idx_dt.push_back(idx[j].index);
	}
	Detection dt;

	// get the number detected bounding boxes so-far
	int num_dt = idx_dt.size();

	// if nothing detected, return
	if (num_dt == 0) {
		tld.dt = dt;
		n = 1;
		return confi;
	}

	// initialize detection structure
	std::vector<int> idxcopy;

	dt.nbb = num_dt;
	for (int j = 0; j < num_dt; j++) {
		dt.bb.col(j) = tld.grid.block(0, idx_dt[j], 4, 1);

		Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), 1> ex =
				tldGetPattern(tld.currentImg, dt.bb.col(j),
						tld.model.patchsize, 0); // measure patch

		MatrixXd result = tldNN(ex, tld); // evaluate nearest neighbour classifier

		// fill detection structure
		if (result(0, 0) > tld.model.thr_nn)
			idxcopy.push_back(j);
		dt.conf2(j) = result(0, 1);
		dt.patch.col(j) = ex;
	}

	// get all indexes that made it through the nearest neighbour
	int idxsize = idxcopy.size();

	// if no conf bb was detected return nan
	if (idxsize == 0) {
		dBB = Matrix<double, 4, 20>::Constant(4, 20,
				std::numeric_limits<double>::quiet_NaN());
		tld.dt = dt;
		n = 1;
		return confi;
	}
	// save detected bounding boxes
	//dBB.resize(4, idxsize);
	if (idxsize > 20)
		idxsize = 20;
	for (int o = 0; o < idxsize; o++)
		dBB.col(o) = dt.bb.col(idxcopy[o]);

	// conservative confidences
	//confi.resize(idxsize);
	for (int o = 0; o < idxsize; o++)
		confi(o) = dt.conf2(idxcopy[o]);

	// save the whole detection structure
	tld.dt = dt;
	n = idxsize;

	return confi;
}

/**
 * Is used to show results on window. Current bounding box,
 * positives and negatives, current target.
 *
 * @param i initial sign
 * @param index
 * @param tld learned structures
 * @param fps number of frames per second
 */
void tldDisplay(int i, unsigned long index, TldStruct& tld, double fps) {

	CvImage inputClone = tld.currentImg.input.clone();

	if (i == 0) {

		// draw bounding box
		bb_draw_add_color(inputClone, tld.currentBB);

		tld.handle = CvImage(inputClone.size(), tld.cfg.imgsource->getImage().depth(), 3);

		cvCvtColor(inputClone, tld.handle, CV_GRAY2BGR);

		cvNamedWindow("Result", CV_WINDOW_AUTOSIZE);

		cvShowImage("Result", tld.handle);
		if (waitKey(10) >= 0)
			std::cout << "key pressed" << std::endl;
	} else {

		// show positive patches
		if (tld.cfg.plot.pex == 1)
			embedPex(inputClone, tld);

		// show negative patches
		if (tld.cfg.plot.nex == 1)
			embedNex(inputClone, tld);

		CvFont font;
		fps = 1 / fps;
		std::ostringstream fpsc;
		fpsc << fps;
		cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1.0, 1.0, 0, 1, CV_AA);

		tld.handle = CvImage(inputClone.size(), tld.cfg.imgsource->getImage().depth(), 3);
		cvCvtColor(inputClone, tld.handle, CV_GRAY2BGR); // colored output image

		// put fps
		cvPutText(tld.handle, (fpsc.str()).c_str(), cvPoint(
				40, 40), &font, cvScalar(0, 0, 255, 0));

		unsigned char size = 100;

		// show current target (not tested)
		if (tld.cfg.plot.target == 1) {

			Vector2d vecin;
			vecin << 4, 4;
			Vector4d bb = bb_rescalerel(tld.currentBB,
					vecin);

			CvImage patch = img_patch(tld.handle, bb);
			CvImage dest = CvImage(cvSize((int) size, (int) size),
					patch.depth(), patch.channels());

			cvResize(patch, dest);

			for (unsigned int y = 0; y < size; y++)
				for (unsigned int x = 0; x < size; x++) {
					((uchar*) (tld.handle.data() + tld.handle.step()
							* (y)))[x] = ((uchar*) (dest.data()
							+ dest.step() * (y)))[x];
				}
		}

		// Replace
		if (tld.cfg.plot.replace == 1) {

			Vector4d bb;
			bb(0) = floor(tld.currentBB(0) + 0.5);
			bb(1) = floor(tld.currentBB(1) + 0.5);
			bb(2) = floor(tld.currentBB(2) + 0.5);
			bb(3) = floor(tld.currentBB(3) + 0.5);

			Vector2i imsize;
			imsize << tld.currentImg.input.width(), tld.currentImg.input.height();

			if (bb_isin(bb, imsize)) {
				unsigned int width = (int) (bb(2) - bb(0)) + 1;
				unsigned int height = (int) (bb(3) - bb(1)) + 1;

				CvImage patch = CvImage(cvSize(width, height),
						tld.target.depth(), tld.target.channels());

				cvResize(tld.target, patch);

				for (unsigned int y = bb(1); y <= bb(3); y++)
					for (unsigned int x = bb(0); x <= bb(2); x++) {
						((uchar*) (tld.handle.data()
								+ tld.handle.step() * (y)))[x]
								= ((uchar*) (patch.data()
										+ patch.step() * (y - int(bb(1)))))[x
										- int(bb(0))];
					}
			}
		}


		// Draw Track
		unsigned int linewidth = 2;
		if (tld.currentValid == 1)
			linewidth = 4;
		cv::Scalar color = cv::Scalar(0, 255, 255);

		// grab current bounding box
		Vector4d bb = tld.currentBB;
		Vector2d vecin;
		vecin << 1.2, 1.2;

		if (!isnan(bb(0))) {
			switch (tld.cfg.plot.drawoutput) {
			case 1:
				bb = bb_rescalerel(bb_square(bb), vecin); // scale 1.2
				bb_draw(tld.handle, bb, color, linewidth);
				break;
			case 3:
				bb_draw(tld.handle, bb, color, linewidth);
			}
		}

		cvShowImage("Result", tld.handle);
		if (waitKey(10) >= 0)
			std::cout << "key pressed" << std::endl;

	}
}

/**
 * Initialize all structures.
 *
 * @param tld learning structures
 */
bool tldInit(TldStruct& tld) {

	// initialize lucas kanade
	lkInit();

	// Get initial image
	if(!tld.cfg.imgsource->nextImage())
		return false;

	tld.currentImg.input = tld.cfg.imgsource->getGrayImage();
	tld.currentImg.blur = img_blur(tld.currentImg.input);

	// get initial bounding box
	Vector4d bb = tld.cfg.initBB;

	Vector2i imsize;
	imsize(0) = tld.currentImg.input.width();
	imsize(1) = tld.currentImg.input.height();
	bb_scan(tld, bb, imsize, tld.model.min_win);

	// Features
	tldGenerateFeatures(tld, tld.model.num_trees, tld.model.num_features);

	// Initialize Detector
	fern0();

	// allocate structures
	fern1(tld.currentImg.input, tld.grid, tld.features, tld.scales);

	// Temporal structures
	Tmp temporal;
	temporal.conf = VectorXd::Zero(tld.nGrid);
	temporal.patt = Matrix<double, 10, Dynamic>::Zero(tld.model.num_trees, tld.nGrid);
	tld.tmp = temporal;

	// RESULTS =================================================================

	// Initialize Trajectory

	tld.prevBB = Vector4d::Constant(
			std::numeric_limits<double>::quiet_NaN());
	tld.currentBB = bb;
	tld.conf = 1;
	tld.currentValid = 1;
	//tld.size = 1;

	// TRAIN DETECTOR ==========================================================

	RowVectorXd overlap = bb_overlap(tld.currentBB, tld.nGrid, tld.grid.topRows(4));

	tld.target = img_patch(tld.currentImg.input, tld.currentBB);

	// Generate Positive Examples
	Matrix<double, TLD_NTREES, Dynamic> pX; // pX: 10 rows
	Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), Dynamic> pEx;
	tld.currentBB = tldGeneratePositiveData(tld, overlap, tld.currentImg,
			tld.p_par_init, pX, pEx);

	MatrixXd pY = MatrixXd::Ones(1, pX.cols());
	// Generate Negative Examples
	Matrix<double, TLD_NTREES, Dynamic> nX; // nX: 10 rows
	Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), Dynamic> nEx;
	tldGenerateNegativeData(tld, overlap, tld.currentImg, nX, nEx);

	// Split Negative Data to Training set and Validation set
	Matrix<double, TLD_NTREES, Dynamic> spnX;
	Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), Dynamic> spnEx;
	tldSplitNegativeData(nX, nEx, spnX, spnEx);

	MatrixXd nY1 = MatrixXd::Zero(1, spnX.cols() / 2);

	MatrixXd xCombined(pX.rows(), pX.cols() + spnX.cols() / 2);
	xCombined << pX, spnX.leftCols(spnX.cols() / 2);
	MatrixXd yCombined(pY.rows(), pY.cols() + nY1.cols());
	yCombined << pY, nY1;
	RowVectorXd idx(xCombined.cols());
	for (int i = 0; i < xCombined.cols(); i++)
		idx(i) = i;

	idx = permutate_cols(idx);

	MatrixXd permX(xCombined.rows(), xCombined.cols());
	VectorXd permY(yCombined.cols());
	for (int i = 0; i < idx.cols(); i++) {
		permX.col(i) = xCombined.col(idx(i));
		permY(i) = yCombined(0, idx(i));
	}

	// Train using training set ------------------------------------------------

	// Fern
	unsigned char bootstrap = 2;
	VectorXd dummy(1);
	dummy(0) = -1;
	fern2(permX, permY, tld.model.thr_fern, bootstrap, dummy);

	// Nearest Neighbour
	tld.npex = 0;
	tld.nnex = 0;

	tldTrainNN(pEx, spnEx.leftCols(spnEx.cols() / 2), tld);
	//tld.model.num_init = tld.npex;

	// Estimate thresholds on validation set  ----------------------------------

	// Fern
	unsigned int ferninsize = spnX.cols() / 2;
	RowVectorXd conf_fern(ferninsize);
	Matrix<double, 10, Dynamic> fernin(10, ferninsize);
	fernin.leftCols(ferninsize) = spnX.rightCols(ferninsize);
	conf_fern = fern3(fernin, ferninsize);
	tld.model.thr_fern = std::max(conf_fern.maxCoeff() / tld.model.num_trees,
			tld.model.thr_fern);

	// Nearest neighbor
	MatrixXd conf_nn(3, 3);
	conf_nn = tldNN(spnEx.rightCols(spnEx.cols() / 2), tld);

	tld.model.thr_nn = std::max(tld.model.thr_nn, conf_nn.block(0, 0, 1,
			conf_nn.cols() / 3).maxCoeff());
	tld.model.thr_nn_valid = std::max(tld.model.thr_nn_valid,
			tld.model.thr_nn);

	return true;
}

/**
 * Generates random features for training set.
 *
 * @param tld learning structure
 * @param nTREES number of columns in features
 * @param nFEAT number of values per tree
 */
void tldGenerateFeatures(TldStruct& tld, unsigned int nTREES,
		unsigned int nFEAT) {

	const double SHIFT = 0.2;
	const double SCA = 1;

	const unsigned int N = 6;
	std::vector<double> a(N), b(N);

	// all values between zero and one
	for (unsigned int i = 0; i < N; i++)
		a[i] = (i * SHIFT);

	for (unsigned int i = 0; i < N; i++)
		b[i] = (i * SHIFT);

	Matrix4Xd x(4, 8 * N * N);
	Matrix4Xd x1(4, N * N);

	unsigned int column = 0;

	// all possible tuples
	for (unsigned int i = 0; i < N; i++)
		for (unsigned int j = 0; j < N; j++) {
			x1.col(column) << a[i], b[j], a[i], b[j];
			column++;
		}

	Matrix4Xd x2(4, 2 * N * N);
	x2 << x1, x1.array() + (SHIFT / 2);

	unsigned int len = x2.cols();

	// add / sub random values between zero and 1
	Matrix4Xd r(4, len);
	r = x2;
	for (unsigned int i = 0; i < len; i++)
		r(2, i) += ((SCA * uniform()) + SHIFT);

	Matrix4Xd l(4, len);
	l = x2;
	for (unsigned int i = 0; i < len; i++)
		l(2, i) -= ((SCA * uniform()) + SHIFT);

	Matrix4Xd t(4, len);
	t = x2;
	for (unsigned int i = 0; i < len; i++)
		t(3, i) -= ((SCA * uniform()) + SHIFT);

	Matrix4Xd bo(4, len);
	bo = x2;
	for (unsigned int i = 0; i < len; i++)
		bo(3, i) += ((SCA * uniform()) + SHIFT);

	// right, left, top, bottom border
	x << r, l, t, bo;

	std::vector<int> lo;
	int cnt = 0;

	// find coefficients that are less than 1 and greater than 0 in rows 0 and 1
	for (unsigned int i = 0; i < 8 * N * N; i++) {
		if (x(0, i) < 1 && x(1, i) < 1 && x(0, i) > 0 && x(1, i) > 0) {
			lo.push_back(1);
			cnt++;
		} else
			lo.push_back(0);

	}

	Matrix4Xd y(4, cnt);
	unsigned int p = 0;

	// values greater than 1 will be set to 1, less than 0 to 0
	for (unsigned int i = 0; i < 8 * N * N; i++) {
		if (lo[i] == 1) {
			y.col(p) = x.col(i);
			if (y(2, p) > 1)
				y(2, p) = 1;
			else if (y(2, p) < 0)
				y(2, p) = 0;
			if (y(3, p) > 1)
				y(3, p) = 1;
			else if (y(3, p) < 0)
				y(3, p) = 0;

			p++;
		}
	}

	// random permutation
	y = permutate_cols(y);

	// cut first elements with length of nFEAT times nTREES
	y = y.block(0, 0, 4, (nFEAT * nTREES));

	MatrixXd z(4 * nFEAT, nTREES); // 52 x 10

	// reshape
	for (unsigned int i = 0; i < nTREES; i++)
		for (p = 0; p < nFEAT; p++)
			z.block(p * 4, i, 4, 1) = y.block(0, i * nFEAT + p, 4, 1);

	tld.features = z;

}

/**
 * Generates initial some random negative patches.
 *
 * @param tld learning structure
 * @param overlap vector of overlapping values of bounding boxes in grid
 * @param img image structure
 * @param nX indices of negative patches
 * @param nEx negative patches
 */
void tldGenerateNegativeData(TldStruct& tld,
		RowVectorXd const & overlap, ImgType& img, Matrix<double,
				TLD_NTREES, Dynamic>& nX, Matrix<double, (TLD_PATCHSIZE
				* TLD_PATCHSIZE), Dynamic>& nEx) {

	// Measure patterns on all bboxes that are far from initial bbox
	std::vector<int> idxN;
	for (unsigned int i = 0; i < overlap.size(); i++) {
		if (overlap(i) < tld.n_par.overlap) {
			idxN.push_back(i);
		}
	}

	MatrixXd fernpat(TLD_NTREES, idxN.size() + 1);
	fernpat = fern5(img, idxN, tld.var / 2);

	// bboxes far and with big variance
	unsigned int len = fernpat.cols();
	unsigned int numOnes = (fernpat.block(0, (len / 2), 1, len / 2).array()
			== 1).count();
	std::vector<int> idxN2(numOnes);

	nX.resize(fernpat.rows(), numOnes);
	unsigned int cnt = 0;
	for (unsigned int i = 0; i < len / 2; i++)
		if (fernpat(0, (len / 2) + i) == 1) {
			idxN2[cnt] = idxN[i];
			nX.col(cnt) = fernpat.col(i);
			cnt++;
		}

	// Randomly select 'num_patches' bboxes and measure patches
	RowVectorXd in(idxN2.size());
	for (unsigned int i = 0; i < idxN2.size(); i++)
		in(i) = i;

	RowVectorXd idx(tld.n_par.num_patches);
	idx = randvalues(in, tld.n_par.num_patches);

	// get bboxes
	MatrixXd bb(4, idx.cols());
	for (unsigned int i = 0; i < idx.cols(); i++)
		bb.col(i) = tld.grid.block(0, idxN2[int(idx(0, i))], 4, 1);

	// get patches from image
	nEx.resize(tld.model.patchsize.x * tld.model.patchsize.y, bb.cols());
	nEx = tldGetPattern(img, bb, tld.model.patchsize, 0);

}

/**
 * Duplicates slightly altered previous found positive patches.
 *
 * @param tld learned structure
 * @param overlap overlapping values of current bb in grid
 * @param img current image
 * @param p_par structure of constants
 * @param pX indices of positive patches in grid
 * @param pEx positive patches
 * @return bbP0 closest bbox
 */
Vector4d tldGeneratePositiveData(TldStruct& tld,
		MatrixXd const & overlap, ImgType& img, P_par& p_par,
		Matrix<double, TLD_NTREES, Dynamic>& pX, Matrix<
				double, (TLD_PATCHSIZE * TLD_PATCHSIZE), Dynamic>& pEx) {

	// Get closest bbox
	MatrixXf::Index maxRow, maxCol;
	unsigned int idxP = overlap.maxCoeff(&maxRow, &maxCol);
	idxP = maxCol;

	MatrixXd bbP0(4, 1);
	bbP0 = tld.grid.block(0, idxP, 4, 1);

	// overlapping bboxes
	// find indices greater 0.6
	std::vector<int> idxPi;
	for (int p = 0; p < overlap.cols(); p++)
		if (overlap(0, p) > 0.6)
			idxPi.push_back(p);

	// use up to 'num_closest' bboxes
	if (idxPi.size() > p_par.num_closest) {

		std::vector<double> overlapB;
		for (int i = 0; i < overlap.cols(); i++)
			overlapB.push_back(overlap(0, i));
		std::vector<size_t> sortedIndices;
		for (int i = 0; i < overlap.cols(); i++)
			sortedIndices.push_back(i);
		std::sort(sortedIndices.begin(), sortedIndices.end(), index_cmp<
				std::vector<double>&> (overlapB));

		// reverse (descending)
		std::reverse(sortedIndices.begin(), sortedIndices.end());

		idxPi.clear();

		// sorted indices
		for (unsigned int i = 0; i < p_par.num_closest; i++)
			idxPi.push_back(sortedIndices[i]);

	}

	MatrixXd bbP(tld.grid.rows(), idxPi.size());
	for (unsigned int i = 0; i < idxPi.size(); i++)
		bbP.col(i) = tld.grid.col(idxPi[i]);

	if (idxPi.size() == 0)
		return Vector4d::Zero();

	// Get hull
	Vector4d bbH;
	bbH = bb_hull(bbP);

	// get a copy of current images
	ImgType im1;
	im1.input = cvCloneImage(img.input);
	im1.blur = cvCloneImage(img.blur);

	// get positive patches from image
	pEx = tldGetPattern(im1, bbP0, tld.model.patchsize, 0);


	if (tld.model.fliplr == 1) {
		MatrixXd pExbuf(pEx.rows(), pEx.cols());
		pExbuf = pEx;
		pEx.resize(pEx.rows(), pEx.cols() * 2);
		pEx << pExbuf, tldGetPattern(im1, bbP0, tld.model.patchsize, 1);
	}

	pX.resize(TLD_NTREES, idxPi.size() * p_par.num_warps);

	// warp blur image to duplicate
	for (unsigned int i = 0; i < p_par.num_warps; i++) {


		if (i > 0) {
			double randomize = uniform();

			// warp image randomly
			CvImage patch_blur = img_patch(img.blur, bbH, randomize, p_par);

			// include in in image
			for (unsigned int y = bbH(1); y <= bbH(3); y++)
				for (unsigned int x = bbH(0); x <= bbH(2); x++) {
					((uchar*) (im1.blur.data() + im1.blur.step() * (y)))[x]
							= ((uchar*) (patch_blur.data()
									+ patch_blur.step() * (y - int(bbH(1)))))[x
									- int(bbH(0))];
				}
		}

		// Measures on blured image
		MatrixXd fernout(TLD_NTREES, idxPi.size() * 2);
		fernout = fern5(im1, idxPi, 0);
		unsigned int frncols = fernout.cols() / 2;

		 // save indices
		pX.block(0, frncols * i, TLD_NTREES, frncols) = fernout.leftCols(frncols);

	}

	tld.var = variance(pEx, pEx.rows()) / 2;
	return bbP0;

}

/**
 * Pickups bbox and converts to Eigen matrix.
 *
 * @param img current img
 * @param bb matrix of bounding boxes
 * @param patchsize size of one patch
 * @param flip use mirrored patches yes / no
 *
 * @return patterns
 */
Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), Dynamic> tldGetPattern(
		ImgType& img, Matrix<double, 4, Dynamic> const & bb,
		Patchsize& patchsize, unsigned int flip) {

	// get patch under bounding box (bb), normalize it size,
	// reshape to a column
	// vector and normalize to zero mean and unit variance (ZMUV)

	// initialize output variable
	unsigned int nBB = bb.cols();
	MatrixXd pattern = MatrixXd::Zero(patchsize.x * patchsize.y,
			nBB);

	// for every bounding box
	for (unsigned int i = 0; i < nBB; i++) {

		// sample patch
		Vector4d tmpbb = bb.col(i);
		CvImage patch = img_patch(img.input, tmpbb);

		// flip if needed
		if (flip) {
			cvFlip(patch, NULL, 1);
		}

		// normalize size to 'patchsize' and nomalize intensities to ZMUV
		pattern.col(i) = tldPatch2Pattern(patch, patchsize);
	}

	return pattern;

}

/**
 * Learns new found positive and negative patches from tldDetection.
 *
 * @param tld learned structures
 * @param I index of image
 */
void tldLearning(TldStruct& tld, unsigned long I) {

	Vector4d bb;
	bb = tld.currentBB;
	ImgType img = tld.currentImg;

	// get pattern of current bbox
	MatrixXd pPatt = tldGetPattern(img, bb, tld.model.patchsize, 0);

	// measure nearest neighbor
	MatrixXd nn = tldNN(pPatt, tld);

	if (nn(0, 0) < 0.5) {
		std::cout << "Fast change" << std::endl;
		tld.currentValid = 0;
		return;
	}

	unsigned int patRows = pPatt.rows();
	VectorXd pPattVec(patRows);
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
	MatrixXd overlap = bb_overlap(bb, tld.nGrid,
			tld.grid.topRows(4));

	// generate positive examples from all bounding boxes that are highly
	// overlapping with current bounding box
	Matrix<double, TLD_NTREES, Dynamic> pX; // pX: 10 rows
	Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), Dynamic> pEx;
	Vector4d ret = tldGeneratePositiveData(tld, overlap, img,
			tld.p_par_update, pX, pEx);

	// labels of the positive patches
	VectorXd pY = VectorXd::Ones(pX.cols());

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

	MatrixXd nEx(tld.dt.patch.rows(), ovpSize);
	for (unsigned int k = 0; k < ovpSize; k++)
		nEx.col(k) = tld.dt.patch.col(idxOverlap[k]);

	// update the Ensemble Classifier (reuses the computation made by detector)
	unsigned int pXcols = pX.cols();
	MatrixXd X(pX.rows(), pXcols + idx.size());
	X.leftCols(pXcols) = pX;
	for (unsigned int k = 0; k < idx.size(); k++)
		X.col(pXcols + k) = tld.tmp.patt.col(idx[k]);

	VectorXd pY2 = VectorXd::Zero(idx.size());
	VectorXd Y(pY.size() + idx.size());
	Y << pY, pY2;

	VectorXd dummy(1);
	dummy(0) = -1;

	fern2(X, Y, tld.model.thr_fern, 2, dummy);

	// update nearest neighbour
	tldTrainNN(pEx, nEx, tld);
}

/* Classifies examples as positive or negative */
Matrix<double, 3, Dynamic> tldNN(Matrix<double, TLD_PATCHSIZE
		* TLD_PATCHSIZE, Dynamic> const & nEx2, TldStruct& tld) {
	//	function [conf1,conf2,isin] = tldNN(x,tld)
	//'conf1' ... full model (Relative Similarity)
	//'conf2' ... validated part of model (Conservative Similarity)
	//'isnin' ... inside positive ball, id positive ball, inside negative ball
	unsigned int N = nEx2.cols();
	MatrixXd isin = MatrixXd::Constant(3, N, std::numeric_limits<
			double>::quiet_NaN());
	//IF positive examples in the model are not defined THEN everything is negative
	if (tld.npex == 0) {
		MatrixXd conf1 = MatrixXd::Zero(3, N);
		MatrixXd conf2 = MatrixXd::Zero(3, N);
		MatrixXd out(3, 3 * N);
		out << conf1, conf2, isin;
		return out;
	}
	//IF negative examples in the model are not defined THEN everything is positive
	if (tld.nnex == 0) {
		MatrixXd conf1 = MatrixXd::Ones(3, N);
		MatrixXd conf2 = MatrixXd::Ones(3, N);
		MatrixXd out(3, 3 * N);
		out << conf1, conf2, isin;
		return out;
	}
	MatrixXd conf1 = MatrixXd::Constant(3, N,
			std::numeric_limits<double>::quiet_NaN());
	MatrixXd conf2 = MatrixXd::Constant(3, N,
			std::numeric_limits<double>::quiet_NaN());
	//for every patch that is tested
	for (unsigned int i = 0; i < N; i++) {
		MatrixXd nccP(1, tld.npex);
		MatrixXd nccN(1, tld.nnex);
		//measure NCC to positive examples
		nccP = distance(nEx2.col(i), tld.pex, tld.npex, 1);
		//measure NCC to negative examples
		nccN = distance(nEx2.col(i), tld.nex, tld.nnex, 1);
		//set isin
		//IF the query patch is highly correlated with any positive patch in the model THEN it is considered to be one of them
		if ((nccP.array() > tld.model.ncc_thesame).any())
			isin(0, i) = 1;
		MatrixXd::Index maxRow, maxCol;
		double dN, dP;
		//get the index of the most correlated positive patch
		dN = nccP.maxCoeff(&maxRow, &maxCol);
		isin(1, i) = double(maxCol);
		//IF the query patch is highly correlated with any negative patch in the model THEN it is considered to be one of them
		if ((nccN.array() > tld.model.ncc_thesame).any())
			isin(2, i) = 1;
		//measure Relative Similarity
		dN = 1 - nccN.maxCoeff();
		dP = 1 - nccP.maxCoeff();
		conf1(0, i) = dN / (dN + dP);
		//measure Conservative Similarity
		double
				maxP =
						nccP.block(0, 0, 1, ceil(tld.model.valid * tld.npex)).maxCoeff();
		dP = 1 - maxP;
		conf2(0, i) = dN / (dN + dP);
	}

	MatrixXd out(3, 3 * N);
	out << conf1, conf2, isin;
	return out;
}

/* Converts an image to Eigen Matrix */
Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, 1> tldPatch2Pattern(CvImage patch,
		Patchsize const& patchsize) {

	CvImage dest = CvImage(
			cvSize((int) patchsize.x, (int) patchsize.y), patch.depth(),
			patch.channels());

	//bilinear' is faster
	cvResize(patch, dest);
	MatrixXd pattern(patchsize.x * patchsize.y, 1);
	for (int x = 0; x < dest.width(); x++)
		for (int y = 0; y < dest.height(); y++)
			pattern(x*patchsize.x + y, 0) = double(((uchar*) (dest.data() + dest.step()
					* (y)))[x]);

	// calculate column-wise mean
	RowVectorXd mean(patchsize.x);
	mean = pattern.colwise().mean();
	pattern = pattern.rowwise() - mean;
	return pattern;
}

bool tldProcessFrame(TldStruct& tld, unsigned long i) {

	tld.prevImg = tld.currentImg;

	if(!tld.cfg.imgsource->nextImage())
		return false;

	//Input image
	ImgType im0;
	im0.input = tld.cfg.imgsource->getGrayImage();

	//Blurred image
	im0.blur = img_blur(im0.input);

	//color image with 3 channels
	tld.currentImg = im0;

	//switch from current to previous
	/* bbox */
	tld.prevBB = tld.currentBB;
	tld.currentBB = Vector4d::Constant(
			std::numeric_limits<double>::quiet_NaN());

	/* valid */
	tld.prevValid = tld.currentValid;
	tld.currentValid = 0;

	//TRACKER
	//rame-to-frame tracking (MedianFlow)
	VectorXd tldTrack = tldTracking(tld, tld.prevBB, i - 1, i);

	//DETECTOR
	//detect appearances by cascaded detector (variance filter -> ensemble classifier -> nearest neighbor)
	Matrix<double, 4, 20> dBB;
	int nD = 0;
	Matrix<double, 20, 1> detConf = tldDetection(tld, i, dBB, nD);

	//INTEGRATOR
	//Tracker defined?
	unsigned int TR = 1;
	if (isnan(tldTrack(0))) {
		TR = 0;
	}
	//Detector defined?
	unsigned int DT = 1;
	if (isnan(dBB(0, 0))) {
		DT = 0;
	}
	if (TR) {
		//copy tracker's result
		//tld.size = 1;
		if (DT) {
			//cluster detections
			MatrixXd cluster = bb_cluster_confidence(dBB, detConf, nD);

			//get indexes of all clusters that are far from tracker and are more confident than the tracker
			unsigned int len = cluster.cols() / 3;
			MatrixXd cBB(4, len);
			cBB = cluster.block(0, 0, 4, len);
			MatrixXd overlap = bb_overlap(tld.currentBB, len, cBB);

			MatrixXd cConf(1, len);
			cConf = cluster.block(0, len, 1, len);
			MatrixXd cSize(1, len);
			cSize = cluster.block(0, 2 * len, 1, len);
			std::vector<unsigned int> id;
			for (unsigned int j = 0; j < len; j++)
				if (overlap(0, j) < 0.5 && cConf(0, j) > tld.conf)
					id.push_back(j);
			//if there is ONE such cluster, re-initialize the tracker
			if (id.size() == 1) {
				tld.currentBB = cBB.col(id[0]);
				tld.conf = cConf(0, id[0]);
				//tld.size = cSize(0, id[0]);
				tld.currentValid = 0;
			} else {
				//adjust the tracker's trajectory
				//get indexes of close detections
				overlap = bb_overlap(tldTrack.topRows(4), tld.dt.nbb, tld.dt.bb);
				std::vector<unsigned int> idTr;
				for (int p = 0; p < overlap.cols(); p++)
					if (overlap(0, p) > 0.7)
						idTr.push_back(p);
				//weighted average trackers trajectory with the close detections
				MatrixXd meanmat(4, 10 + idTr.size());
				for (int p = 0; p < 10; p++) {
					meanmat.col(p) = tldTrack.topRows(4);
				}
				for (unsigned int p = 10; p < 10 + idTr.size(); p++)
					meanmat.col(p) = tld.dt.bb.col(idTr[p - 10]);
				tld.currentBB = meanmat.rowwise().mean();
			}
		}
	} else if (DT) {
			//cluster detections
			MatrixXd cluster = bb_cluster_confidence(dBB, detConf, nD);
			//if there is just a single cluster, re-initialize the tracker
			if (cluster.cols() / 3 == 1) {
				tld.currentBB = cluster.col(0);
				tld.conf = cluster(0, 1);
				//tld.size = cluster(0, 2);
				tld.currentValid = 0;
			}
	} else if (tld.hasNewBB) {
		tld.currentBB = tld.newBB;
		tld.conf = 1;
		//tld.size = ???
		tld.currentValid = 1;
		tld.hasNewBB = false;
	}

	//LEARNING
	if (tld.control.update_detector && tld.currentValid == 1)
		tldLearning(tld, i);

	return true;
}

void tldSetBB(TldStruct& tld, Vector4d& bb) {
    tld.newBB = bb;
    tld.hasNewBB = true;
}

/*Splits negative data to training and validation set*/
void tldSplitNegativeData(
		Matrix<double, TLD_NTREES, Dynamic> const & nX,
		Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Dynamic> const & nEx,
		Matrix<double, TLD_NTREES, Dynamic>& spnX,
		Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Dynamic>& spnEx) {

	unsigned int N = nX.cols();
	RowVectorXd Nvec(N);

	for (unsigned int i = 0; i < N; i++)
		Nvec(i) = i;

	Nvec = permutate_cols(Nvec);

	MatrixXd permnX(nX.rows(), nX.cols());

	for (unsigned int i = 0; i < N; i++)
		permnX.col(i) = nX.col(Nvec(i));

	spnX = permnX;

	N = nEx.cols();
	RowVectorXd Nvec2(N);

	for (unsigned int i = 0; i < N; i++)
		Nvec2(i) = i;

	Nvec2 = permutate_cols(Nvec2);

	MatrixXd permnEx(nEx.rows(), nEx.cols());

	for (unsigned int i = 0; i < N; i++)
		permnEx.col(i) = nEx.col(Nvec2(i));

	spnEx = permnEx;
}

VectorXd tldTracking(TldStruct& tld, VectorXd const & bb, int i, int j) {
	//BB2    = []; % estimated bounding
	//Conf   = []; % confidence of prediction
	//Valid  = 0;  % is the predicted bounding box valid? if yes, learning will take place ...

	Vector4d bb2;
	MatrixXd Conf;
	double valid = 0;

	bb2 = Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());

	if (!bb_isdef(bb))
		return bb;

	MatrixXd xFI(2, 150);
	int xFISize = 0;
	//generate 10x10 points on bb
	xFI = bb_points(bb, 10, 10, 5, xFISize);
	MatrixXd xFJ(4, 150);
	//get all reliable points
	xFJ = lk2(tld.prevImg.input, tld.currentImg.input, xFI, xFI, xFISize, xFISize, 0);

	double medFB = median(xFJ.leftCols(xFISize).row(2));

	double medNCC = median(xFJ.leftCols(xFISize).row(3));

	int counter = 0;
	//get indexes of reliable points
	VectorXd idxF(1);
	VectorXd idxFbak(1);
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

	MatrixXd xFInew(xFI.rows(), idxF.rows());

	for (int f = 0; f < idxF.rows(); f++)
		xFInew.col(f) = xFI.col(idxF(f));

	MatrixXd xFJnew(2, idxF.rows());

	for (int k = 0; k < idxF.rows(); k++)
		xFJnew.col(k) = xFJ.block(0, idxF(k), 2, 1);

	//predict the bounding box
	bb2 = bb_predict(bb, xFInew, xFJnew);

	//bounding box out of image
	Vector2i imgsize;
	imgsize(0) = tld.currentImg.input.width();
	imgsize(1) = tld.currentImg.input.height();
	if (!bb_isdef(bb2) || bb_isout(bb2, imgsize)) {
		bb2 = Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());
		return bb2;
	}
	//too unstable predictions
	if (tld.control.maxbbox > 0 && medFB > 10) {
		bb2 = Vector4d::Constant(std::numeric_limits<double>::quiet_NaN());
		return bb2;
	}
	//sample patch in current image
	MatrixXd patchJ = tldGetPattern(tld.currentImg, bb2,
			tld.model.patchsize, 0);
	Conf = tldNN(patchJ, tld);
	valid = tld.prevValid;

	unsigned int confLen = Conf.cols() / 3;
	double consSim;
	consSim = Conf(0, confLen);
	//tracking takes place
	if (consSim > tld.model.thr_nn_valid)
		valid = 1;

	tld.currentBB = bb2;

	tld.conf = consSim;

	tld.currentValid = valid;

	return bb2;
}

/*Trains nearest neighbor*/
void tldTrainNN(
		Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Dynamic> const & pEx,
		Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Dynamic> const & nEx1,
		TldStruct& tld) {

	unsigned int nP = pEx.cols(); //number of positive examples
	unsigned int nN = nEx1.cols(); //number of negative examples

	//x = [pEx,nEx];
	MatrixXd x(pEx.rows(), nN + nP);

	if (nN > 0)
		x << pEx, nEx1;
	else
		x = pEx;

	//y = [ones(1,nP), zeros(1,nN)];
	MatrixXd y(1, nP + nN);
	MatrixXd yi = MatrixXd::Ones(1, nP);
	MatrixXd yii = MatrixXd::Zero(1, nN);

	if (nN > 0)
		y << yi, yii;
	else
		y = yi;

	//Permutate the order of examples
	RowVectorXd idx(nP + nN);
	for (unsigned int i = 0; i < nP + nN; i++)
		idx(i) = i;

	idx = permutate_cols(idx);

	MatrixXd x2(x.rows(), nP + nN + 1);
	MatrixXd y2(1, nP + nN + 1);

	//No positive example yet
	if (nP > 0) {
		//Always add the first positive patch as the first (important in initialization).
		for (unsigned int i = 0; i < nP + nN; i++)
			x2.col(i + 1) = x.col(idx(i));
		x2.col(0) = pEx.col(0);
		for (unsigned int i = 0; i < nP + nN; i++)
			y2.col(i + 1) = y.col(idx(i));
		y2(0, 0) = 1;
	}
	//Bootstrap
	for (unsigned int i = 0; i < nP + nN + 1; i++) {
		//Measure Relative similarity
		MatrixXd conf(3, 3);
		conf = tldNN(x2.col(i), tld);
		//Positive
		if (y2(i) == 1 && conf(0, 0) <= tld.model.thr_nn && tld.npex < TLD_MAXPATCHES) {
			if (isnan(conf(1, 2))) {
				tld.npex = 1;
				tld.pex.col(0) = x2.col(i);
				continue;
			}
			//Add to model
			MatrixXd pex1(tld.pex.rows(), conf(1, 2) + 1);
			pex1 = tld.pex.block(0, 0, tld.pex.rows(), conf(1, 2) + 1);

			MatrixXd pex2(tld.pex.rows(), tld.npex - pex1.cols());
			pex2 = tld.pex.block(0, conf(1, 2) + 1, tld.pex.rows(), tld.npex
					- pex1.cols());

			if (pex2.cols() > 0) {
				tld.pex.leftCols(conf(1, 2) + 1) = pex1;
				tld.pex.col(conf(1, 2) + 1) = x2.col(i);
				tld.pex.block(0, conf(1, 2) + 2, (TLD_PATCHSIZE * TLD_PATCHSIZE),
						pex2.cols()) = pex2;
			} else {
				tld.pex.leftCols(conf(1, 2) + 1) = pex1;
				tld.pex.col(conf(1, 2) + 1) = x2.col(i);
			}
			tld.npex++;
		}

		//
		//Negative
		if (y2(i) == 0 && conf(0, 0) > 0.5 && tld.nnex < TLD_MAXPATCHES) {
			tld.nex.col(tld.nnex) = x2.col(i);
			tld.nnex++;
		}

	}
}
