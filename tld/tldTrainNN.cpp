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
#include "../utils/utility.h"
#include <limits>
/*Trains nearest neighbor*/
void tldTrainNN(
		Eigen::Matrix<double, PATCHSIZE * PATCHSIZE, Eigen::Dynamic> const & pEx,
		Eigen::Matrix<double, PATCHSIZE * PATCHSIZE, Eigen::Dynamic> const & nEx1,
		TldStruct& tld) {

	unsigned int nP = pEx.cols(); //number of positive examples
	unsigned int nN = nEx1.cols(); //number of negative examples

	//x = [pEx,nEx];
	Eigen::MatrixXd x(pEx.rows(), nN + nP);

	if (nN > 0)
		x << pEx, nEx1;
	else
		x = pEx;

	//y = [ones(1,nP), zeros(1,nN)];
	Eigen::MatrixXd y(1, nP + nN);
	Eigen::MatrixXd yi = Eigen::MatrixXd::Ones(1, nP);
	Eigen::MatrixXd yii = Eigen::MatrixXd::Zero(1, nN);

	if (nN > 0)
		y << yi, yii;
	else
		y = yi;

	//Permutate the order of examples
	Eigen::RowVectorXd idx(nP + nN);
	for (unsigned int i = 0; i < nP + nN; i++)
		idx(i) = i;

	idx = permutate_cols(idx);

	Eigen::MatrixXd x2(x.rows(), nP + nN + 1);
	Eigen::MatrixXd y2(1, nP + nN + 1);

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
		Eigen::MatrixXd conf(3, 3);
		conf = tldNN(x2.col(i), tld);
		//Positive
		if (y2(i) == 1 && conf(0, 0) <= tld.model.thr_nn && tld.npex < MAXPATCHES) {
			if (isnan(conf(1, 2))) {
				tld.npex = 1;
				tld.pex.col(0) = x2.col(i);
				continue;
			}
			//Add to model
			Eigen::MatrixXd pex1(tld.pex.rows(), conf(1, 2) + 1);
			pex1 = tld.pex.block(0, 0, tld.pex.rows(), conf(1, 2) + 1);

			Eigen::MatrixXd pex2(tld.pex.rows(), tld.npex - pex1.cols());
			pex2 = tld.pex.block(0, conf(1, 2) + 1, tld.pex.rows(), tld.npex
					- pex1.cols());

			if (pex2.cols() > 0) {
				tld.pex.leftCols(conf(1, 2) + 1) = pex1;
				tld.pex.col(conf(1, 2) + 1) = x2.col(i);
				tld.pex.block(0, conf(1, 2) + 2, (PATCHSIZE * PATCHSIZE),
						pex2.cols()) = pex2;
			} else {
				tld.pex.leftCols(conf(1, 2) + 1) = pex1;
				tld.pex.col(conf(1, 2) + 1) = x2.col(i);
			}
			tld.npex++;
		}

		//
		//Negative
		if (y2(i) == 0 && conf(0, 0) > 0.5 && tld.nnex < MAXPATCHES) {
			tld.nex.col(tld.nnex) = x2.col(i);
			tld.nnex++;
		}

	}

}
