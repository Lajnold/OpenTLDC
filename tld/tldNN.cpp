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
#include <limits>
#include "../mex/mex.h"
/* Classifies examples as positive or negative */
Eigen::Matrix<double, 3, Eigen::Dynamic> tldNN(Eigen::Matrix<double, PATCHSIZE
		* PATCHSIZE, Eigen::Dynamic> const & nEx2, TldStruct& tld) {
	//	function [conf1,conf2,isin] = tldNN(x,tld)
	//'conf1' ... full model (Relative Similarity)
	//'conf2' ... validated part of model (Conservative Similarity)
	//'isnin' ... inside positive ball, id positive ball, inside negative ball
	unsigned int N = nEx2.cols();
	Eigen::MatrixXd isin = Eigen::MatrixXd::Constant(3, N, std::numeric_limits<
			double>::quiet_NaN());
	//IF positive examples in the model are not defined THEN everything is negative
	if (tld.npex == 0) {
		Eigen::MatrixXd conf1 = Eigen::MatrixXd::Zero(3, N);
		Eigen::MatrixXd conf2 = Eigen::MatrixXd::Zero(3, N);
		Eigen::MatrixXd out(3, 3 * N);
		out << conf1, conf2, isin;
		return out;
	}
	//IF negative examples in the model are not defined THEN everything is positive
	if (tld.nnex == 0) {
		Eigen::MatrixXd conf1 = Eigen::MatrixXd::Ones(3, N);
		Eigen::MatrixXd conf2 = Eigen::MatrixXd::Ones(3, N);
		Eigen::MatrixXd out(3, 3 * N);
		out << conf1, conf2, isin;
		return out;
	}
	Eigen::MatrixXd conf1 = Eigen::MatrixXd::Constant(3, N,
			std::numeric_limits<double>::quiet_NaN());
	Eigen::MatrixXd conf2 = Eigen::MatrixXd::Constant(3, N,
			std::numeric_limits<double>::quiet_NaN());
	//for every patch that is tested
	for (unsigned int i = 0; i < N; i++) {
		Eigen::MatrixXd nccP(1, tld.npex);
		Eigen::MatrixXd nccN(1, tld.nnex);
		//measure NCC to positive examples
		nccP = distance(nEx2.col(i), tld.pex, tld.npex, 1);
		//measure NCC to negative examples
		nccN = distance(nEx2.col(i), tld.nex, tld.nnex, 1);
		//set isin
		//IF the query patch is highly correlated with any positive patch in the model THEN it is considered to be one of them
		if ((nccP.array() > tld.model.ncc_thesame).any())
			isin(0, i) = 1;
		Eigen::MatrixXd::Index maxRow, maxCol;
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

	Eigen::MatrixXd out(3, 3 * N);
	out << conf1, conf2, isin;
	return out;
}
