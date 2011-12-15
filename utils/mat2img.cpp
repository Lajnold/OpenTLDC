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

#include "utility.h"


Eigen::MatrixXd mat2img(Eigen::Matrix<double, (TLD_PATCHSIZE * TLD_PATCHSIZE),
		Eigen::Dynamic> const & data, int n, unsigned int nrow) {

	double result_1 = sqrt(data.rows());


	// Check if matrix data has values
	if (data.isZero()) {
		return data;
	}

	double width = result_1;
	double height = result_1;
	unsigned int ncol_data = n;

	if (nrow > ncol_data) {
		nrow = ncol_data;
	}

	unsigned int ncol = ceil(double(ncol_data)/double(nrow));


	/*
	 * Declaring matrix img and setting number of columns and rows. All values
	 * are initialized with zeros
	 */

	Eigen::MatrixXd img = Eigen::MatrixXd::Zero(nrow*height, ncol*width);


	// Important for last for-loop
	int counter_row = 0;
	int counter_col = 0;

	for (int i = 0; i < n; i++) {

		Eigen::MatrixXd img0(result_1, result_1);

		int counter = 0;

		for (int j = 0; j < img0.cols(); j++) {

			int temp = counter;

			for (int k = 0; k < img0.rows(); k++) {

				// Copying values of column i from matrix data to matrix img0
				img0(k,j) = data(k + temp,i);

				counter++;
			}
		}

		// Finding maximum and minimum values in img0
		double min_img0 = img0.minCoeff();
		double max_img0 = img0.maxCoeff();

		/*
		 * Subtraction of min_img0 from each img0 value and
		 * multiplication by 1/(max_img0 - min_img0)
		 */

		img0 = (img0.array() - min_img0).matrix() / (max_img0 - min_img0);

		// Check whether row end of img is reached
		if (counter_row == img.rows()) {
			counter_row = 0;   // set counter_row to zero
			counter_col += img0.cols(); // set counter_col to the next free column value in img

		}

		// Writing img0 matrices to the correct locations in img
		int temp = counter_row;

		for (int j = 0; j < img0.cols(); j++) {
			for (int k = 0; k < img0.rows(); k++) {
				img(k + temp,j + counter_col) = img0(k,j);
				 // Important to determine the last row which was used by img0
			}
			counter_row++;
		}
	}
	return img;
}
