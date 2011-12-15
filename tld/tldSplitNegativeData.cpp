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
/*Splits negative data to training and validation set*/
void tldSplitNegativeData(
		Eigen::Matrix<double, TLD_NTREES, Eigen::Dynamic> const & nX,
		Eigen::Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Eigen::Dynamic> const & nEx,
		Eigen::Matrix<double, TLD_NTREES, Eigen::Dynamic>& spnX,
		Eigen::Matrix<double, TLD_PATCHSIZE * TLD_PATCHSIZE, Eigen::Dynamic>& spnEx) {

	unsigned int N = nX.cols();
	Eigen::RowVectorXd Nvec(N);

	for (unsigned int i = 0; i < N; i++)
		Nvec(i) = i;

	Nvec = permutate_cols(Nvec);

	Eigen::MatrixXd permnX(nX.rows(), nX.cols());

	for (unsigned int i = 0; i < N; i++)
		permnX.col(i) = nX.col(Nvec(i));

	spnX = permnX;

	N = nEx.cols();
	Eigen::RowVectorXd Nvec2(N);

	for (unsigned int i = 0; i < N; i++)
		Nvec2(i) = i;

	Nvec2 = permutate_cols(Nvec2);

	Eigen::MatrixXd permnEx(nEx.rows(), nEx.cols());

	for (unsigned int i = 0; i < N; i++)
		permnEx.col(i) = nEx.col(Nvec2(i));

	spnEx = permnEx;

}
