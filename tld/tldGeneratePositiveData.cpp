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
#include "../utils/utility.h"
#include "../img/img.h"
#include "../mex/mex.h"

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
Eigen::Vector4d tldGeneratePositiveData(TldStruct& tld,
		Eigen::MatrixXd const & overlap, ImgType& img, p_par& p_par,
		Eigen::Matrix<double, NTREES, Eigen::Dynamic>& pX, Eigen::Matrix<
				double, (PATCHSIZE * PATCHSIZE), Eigen::Dynamic>& pEx) {

	// Get closest bbox
	Eigen::MatrixXf::Index maxRow, maxCol;
	unsigned int idxP = overlap.maxCoeff(&maxRow, &maxCol);
	idxP = maxCol;

	Eigen::MatrixXd bbP0(4, 1);
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

	Eigen::MatrixXd bbP(tld.grid.rows(), idxPi.size());
	for (unsigned int i = 0; i < idxPi.size(); i++)
		bbP.col(i) = tld.grid.col(idxPi[i]);

	if (idxPi.size() == 0)
		return Eigen::Vector4d::Zero();

	// Get hull
	Eigen::Vector4d bbH;
	bbH = bb_hull(bbP);

	// get a copy of current images
	ImgType im1;
	im1.input = cvCloneImage(img.input);
	im1.blur = cvCloneImage(img.blur);

	// get positive patches from image
	pEx = tldGetPattern(im1, bbP0, tld.model.patchsize, 0);


	if (tld.model.fliplr == 1) {
		Eigen::MatrixXd pExbuf(pEx.rows(), pEx.cols());
		pExbuf = pEx;
		pEx.resize(pEx.rows(), pEx.cols() * 2);
		pEx << pExbuf, tldGetPattern(im1, bbP0, tld.model.patchsize, 1);
	}

	pX.resize(NTREES, idxPi.size() * p_par.num_warps);

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
		Eigen::MatrixXd fernout(NTREES, idxPi.size() * 2);
		fernout = fern5(im1, idxPi, 0);
		unsigned int frncols = fernout.cols() / 2;

		 // save indices
		pX.block(0, frncols * i, NTREES, frncols) = fernout.leftCols(frncols);

	}

	tld.var = variance(pEx, pEx.rows()) / 2;
	return bbP0;

}

