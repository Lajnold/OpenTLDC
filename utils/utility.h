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

#ifndef UTILITY_H_
#define UTILITY_H_

#include <algorithm>
#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Core>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "../tld/tldconst.h"

Eigen::MatrixXd mat2img(Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE),
		Eigen::Dynamic> const & data, int n, unsigned int nrow);

inline double uniform() {
	return double(rand()) / (double(RAND_MAX) + 1);
}

inline double uniform(double a, double b) {
	return a + (b - a) * uniform();
}

inline std::vector<double> uniqueval(std::vector<double> const& in) {

	std::vector<double> out(in.size());
	bool isin = false;
	for (unsigned int i = 0; i < in.size(); i++) {
		isin = false;
		for (unsigned int p = 0; p < out.size(); p++)
			if (out[p] == in[i]) {
				isin = true;
				break;
			}
		if (!isin)
			out.push_back(in[i]);
	}

	return out;
}

inline double variance(Eigen::VectorXd m, unsigned int size) {
	// Bisher nur für einen Spaltenvektor geeignet.
	double mean = m.mean();
	return (m.array() - mean).matrix().dot((m.array() - mean).matrix()) / size;

}

template<class T> struct index_cmp {
	index_cmp(const T arr) :
		arr(arr) {
	}
	bool operator()(const size_t a, const size_t b) const {
		return arr[a] < arr[b];
	}
	const T arr;
};

/**
 * Copyright: Eigen3 ( test / permmatrix_test )
 */
/**
 * written by Arne Kreutzmann
 **/
template<typename PermutationVectorType>
void randomPermutationVector(PermutationVectorType& v,
		typename PermutationVectorType::Index size) {
	typedef typename PermutationVectorType::Index Index;
	typedef typename PermutationVectorType::Scalar Scalar;
	v.resize(size);
	for (Index i = 0; i < size; ++i)
		v(i) = Scalar(i);
	if (size == 1)
		return;
	for (Index n = 0; n < 3 * size; ++n) {
		Index i = Eigen::internal::random<Index>(0, size - 1);
		Index j;
		do
			j = Eigen::internal::random<Index>(0, size - 1);
		while (j == i);
		std::swap(v(i), v(j));
	}
}

template<typename MatrixType>
MatrixType permutate_rows(MatrixType& matrix) {
	typedef Eigen::PermutationMatrix<MatrixType::RowsAtCompileTime>
			PermutationType;
	typedef Eigen::Matrix<int, MatrixType::RowsAtCompileTime, 1>
			PermutationVectorType;

	PermutationVectorType permutation_vec;
	randomPermutationVector(permutation_vec, matrix.rows());
	PermutationType permutation(permutation_vec);

	return permutation * matrix;
}

template<typename MatrixType>
MatrixType permutate_cols(MatrixType& matrix) {
	typedef Eigen::PermutationMatrix<MatrixType::ColsAtCompileTime>
			PermutationType;
	typedef Eigen::Matrix<int, 1, MatrixType::ColsAtCompileTime>
			PermutationVectorType;

	PermutationVectorType permutation_vec;
	randomPermutationVector(permutation_vec, matrix.cols());
	PermutationType permutation(permutation_vec);

	return matrix * permutation;
}

inline Eigen::RowVectorXd randvalues(Eigen::RowVectorXd const in,
		unsigned int k) {

	//	function out = randvalues(in,k)
	//	% Randomly selects 'k' values from vector 'in'.
	//
	//	out = [];
	//
	//	N = size(in,2);
	unsigned int N = in.cols();
	//
	//	if k == 0
	//	  return;
	//	end
	if (k == 0)
		return Eigen::RowVectorXd::Zero(1);
	//
	//	if k > N
	//	  k = N;
	//	end
	if (k > N)
		k = N;
	//
	//	if k/N < 0.0001
	if ((k / N) < 0.0001) {
		//	 i1 = unique(ceil(N*rand(1,k)));

		// random values rounded to nearest integer
		std::vector<double> rand(k);
		for (unsigned int i = 0; i < k; i++)
			rand[i] = (floor((uniform() * N) + 0.5));

		// unique
		std::vector<double> uniquerand = uniqueval(rand);

		Eigen::RowVectorXd out(uniquerand.size());

		//	 out = in(:,i1);
		for (unsigned int i = 0; i < uniquerand.size(); i++)
			out(i) = in(uniquerand[i]);

		return out;

	} else {
		//	 i2 = randperm(N);
		Eigen::MatrixXd i2(1, N);
		for (unsigned int i = 0; i < N; i++)
			i2(0, i) = i;

		i2 = permutate_cols(i2);

		//	 out = in(:,sort(i2(1:k)));
		std::vector<double> sortedI2(k);
		for (unsigned int i = 0; i < k; i++)
			sortedI2[i] = i2(0, i);

		std::sort(sortedI2.begin(), sortedI2.end());

		Eigen::RowVectorXd out(k);

		for (unsigned int i = 0; i < k; i++)
			out(i) = in(sortedI2[i]);

		return out;

	}
}

#endif
