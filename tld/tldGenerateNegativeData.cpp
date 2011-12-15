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
#include "../utils/utility.h"

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
		Eigen::RowVectorXd const & overlap, ImgType& img, Eigen::Matrix<double,
				TLD_NTREES, Eigen::Dynamic>& nX, Eigen::Matrix<double, (TLD_PATCHSIZE
				* TLD_PATCHSIZE), Eigen::Dynamic>& nEx) {

	// Measure patterns on all bboxes that are far from initial bbox
	std::vector<int> idxN;
	for (unsigned int i = 0; i < overlap.size(); i++) {
		if (overlap(i) < tld.n_par.overlap) {
			idxN.push_back(i);
		}
	}

	Eigen::MatrixXd fernpat(TLD_NTREES, idxN.size() + 1);
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
	Eigen::RowVectorXd in(idxN2.size());
	for (unsigned int i = 0; i < idxN2.size(); i++)
		in(i) = i;

	Eigen::RowVectorXd idx(tld.n_par.num_patches);
	idx = randvalues(in, tld.n_par.num_patches);

	// get bboxes
	Eigen::MatrixXd bb(4, idx.cols());
	for (unsigned int i = 0; i < idx.cols(); i++)
		bb.col(i) = tld.grid.block(0, idxN2[int(idx(0, i))], 4, 1);

	// get patches from image
	nEx.resize(tld.model.patchsize.x * tld.model.patchsize.y, bb.cols());
	nEx = tldGetPattern(img, bb, tld.model.patchsize, 0);

}
