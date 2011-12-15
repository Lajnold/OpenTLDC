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

#include <stdio.h>
#include "mex.h"
#include <iostream>
#include <math.h>

// correlation
double ccorr(Eigen::Matrix<double, Eigen::Dynamic, 1> const & f1,
		Eigen::Matrix<double, Eigen::Dynamic, 1> const & f2, int numDim) {
	double f = 0;
	for (int i = 0; i < numDim; i++) {
		f += f1(i) * f2(i);
	}
	return f;
}

// correlation normalized
double ccorr_normed(Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), 1> const & f1,
		Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), 1> const & f2, int numDim) {
	double corr = 0;
	double norm1 = 0;
	double norm2 = 0;

	for (int i = 0; i < numDim; i++) {
		corr += f1(i) * f2(i);
		norm1 += f1(i) * f1(i);
		norm2 += f2(i) * f2(i);
	}
	// normalization to <0,1>
	return (corr / sqrt(norm1 * norm2) + 1) / 2.0;
}

// euclidean distance
double euclidean(Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), 1> const & f1,
		Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), 1> const & f2, int numDim) {

	double sum = 0;
	for (int i = 0; i < numDim; i++) {
		sum += (f1(i) - f2(i)) * (f1(i) - f2(i));
	}
	return sqrt(sum);
}

Eigen::MatrixXd distance(
		Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), 1> const &x1,
		Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE), TLD_MAXPATCHES> const &x2,
		int nx2, unsigned int flag) {

	unsigned int N1 = x1.cols();
	unsigned int N2 = nx2;

	unsigned int M1 = x1.rows();
	unsigned int M2 = x2.rows();

	if (M1 != M2) {
		std::cout << "Wrong Input" << std::endl;
		Eigen::MatrixXd out = Eigen::MatrixXd::Constant(3, 3,
				std::numeric_limits<double>::quiet_NaN());
		return out;
	}

	Eigen::MatrixXd out(N1, N2);

	switch (flag) {
	case 1:
		for (unsigned int i = 0; i < N2; i++) {
			for (unsigned int ii = 0; ii < N1; ii++) {
				out(ii, i) = ccorr_normed(x1.col(ii), x2.col(i), M1);
			}
		}
		return out;
	case 2:

		for (unsigned int i = 0; i < N2; i++) {
			for (unsigned int ii = 0; ii < N1; ii++) {
				out(ii, i) = euclidean(x1.col(ii), x2.col(i), M1);
			}
		}

		return out;
	}

	return out;
}
