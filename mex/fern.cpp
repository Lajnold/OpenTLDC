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

#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

#include "mex.h"
#include "../utils/utility.h"

using namespace tld;

static const int BBOX_STEP = 7;
static const int nBIT = 1; // number of bits per feature

#define sub2idx(row,col,height) ((int) (floor((row)+0.5) + floor((col)+0.5)*(height)))

void iimg(IplImage *in, double *ii, int imH, int imW) {

	double *prev_line = ii;
	double s;

	unsigned char* p = (unsigned char*) in->imageData;
	unsigned char* tRow = p;

	*ii++ = (double) *p;
	p += in->widthStep;

	for (int x = 1; x < imH; x++) {
		*ii = *p + *(ii - 1);
		ii++;
		p += in->widthStep;
	}

	for (int y = 1; y < imW; y++) {
		p = tRow + y;
		s = 0;
		for (int x = 0; x < imH; x++) {
			s += (double) *p;
			*ii = s + *prev_line;
			ii++;
			p += in->widthStep;
			prev_line++;
		}
	}
}

void iimg2(IplImage *in, double *ii2, int imH, int imW) {

	double *prev_line = ii2;
	double s;

	unsigned char* p = (unsigned char*) in->imageData;
	unsigned char* tRow = p;

	*ii2++ = (double) ((*p) * (*p));
	p += in->widthStep;

	for (int x = 1; x < imH; x++) {
		*ii2 = (*p) * (*p) + *(ii2 - 1);
		ii2++;
		p += in->widthStep;
	}

	for (int y = 1; y < imW; y++) {
		p = tRow + y;
		s = 0;
		for (int x = 0; x < imH; x++) {
			s += (double) ((*p) * (*p));
			*ii2 = s + *prev_line;
			ii2++;
			p += in->widthStep;
			prev_line++;
		}
	}

}

double bbox_var_offset(double *ii, double *ii2, Point *off, int imgHeight) {
	// off[0-3] corners of bbox, off[4] area

	double mX = (ii[off[3].row + off[3].col * imgHeight] - ii[off[2].row
			+ off[2].col * imgHeight] - ii[off[1].row + off[1].col * imgHeight]
			+ ii[off[0].row + off[0].col * imgHeight]) / (double) off[4].row;

	double mX2 = (ii2[off[3].row + off[3].col * imgHeight] - ii2[off[2].row
			+ off[2].col * imgHeight] - ii2[off[1].row + off[1].col * imgHeight]
			+ ii2[off[0].row + off[0].col * imgHeight]) / (double) off[4].row;

	return mX2 - mX * mX;
}

void update(FernData &fernData, Eigen::Matrix<double, 10, 1> x, int C, int N) {
	for (int i = 0; i < fernData.nTrees; i++) {

		int idx = (int) x(i);

		(C == 1) ? fernData.nP[i][idx] += N : fernData.nN[i][idx] += N;

		if (fernData.nP[i][idx] == 0) {
			fernData.weights[i][idx] = 0;
		} else {
			fernData.weights[i][idx] = ((double) (fernData.nP[i][idx])) / (fernData.nP[i][idx] + fernData.nN[i][idx]);
		}
	}
}

double measure_forest(FernData &fernData, Eigen::Matrix<double, 10, 1> idx) {
	double votes = 0;
	for (int i = 0; i < fernData.nTrees; i++) {
		votes += fernData.weights[i][idx(i)];
	}
	return votes;
}

Point* create_offsets_bbox(FernData &fernData, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bb) {

	Point *offsets = new Point[BBOX_STEP * fernData.nBbox];
	Point *off = offsets;

	for (int i = 0; i < fernData.nBbox; i++) {
		(*off).row = (int) floor((bb(1, i) - 1) + 0.5);
		(*off).col = (int) floor((bb(0, i) - 1) + 0.5); // 0
		off++;
		(*off).row = (int) floor((bb(3, i) - 1) + 0.5);
		(*off).col = (int) floor((bb(0, i) - 1) + 0.5); // 1
		off++;
		(*off).row = (int) floor((bb(1, i) - 1) + 0.5);
		(*off).col = (int) floor((bb(2, i) - 1) + 0.5); // 2
		off++;
		(*off).row = (int) floor((bb(3, i) - 1) + 0.5);
		(*off).col = (int) floor((bb(2, i) - 1) + 0.5); // 3
		off++;
		(*off).row = (int) ((bb(2, i) - bb(0, i)) * (bb(3, i) - bb(1, i))); // 4
		off++;
		(*off).row = (int) (bb(4, i) - 1) * 2 * fernData.nFeat * fernData.nTrees; // 5
		off++;
		(*off).row = bb(5, i); // 6
		off++;
	}
	return offsets;
}

Point* create_offsets(FernData &fernData, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> scale,
		Eigen::Matrix<double, 4 * TLD_NFEATURES, TLD_NTREES> x) {


	Point *offsets = new Point[fernData.nScale * fernData.nTrees * fernData.nFeat * 2];
	Point *off = offsets;

	for (int k = 0; k < fernData.nScale; k++) {
		//double *scale = scale0 + 2 * k;
		for (int i = 0; i < fernData.nTrees; i++) {
			for (int j = 0; j < fernData.nFeat; j++) {
				(*off).row = (int) floor(((scale(0, k) - 1) * x(4 * j + 1, i))
						+ 0.5);
				(*off).col = (int) floor(((scale(1, k) - 1) * x(4 * j, i))
						+ 0.5);
				off++;
				(*off).row = (int) floor(((scale(0, k) - 1) * x(4 * j + 3, i))
						+ 0.5);
				(*off).col = (int) floor(((scale(1, k) - 1) * x(4 * j + 2, i))
						+ 0.5);
				off++;

			}
		}
	}
	return offsets;
}

int measure_tree_offset(FernData &fernData, IplImage* img, int idx_bbox, int idx_tree) {

	int index = 0;

	Point *bbox = fernData.bboxes + idx_bbox * BBOX_STEP;
	Point *off = fernData.offsets + bbox[5].row + idx_tree * 2 * fernData.nFeat;

	for (int i = 0; i < fernData.nFeat; i++) {
		index <<= 1;
		int fp0 = 0;
		if ((off[0].row + bbox[0].row) < fernData.imgHeight)
			fp0 = ((uchar*) (img->imageData + img->widthStep * (off[0].row
					+ bbox[0].row)))[off[0].col + bbox[0].col];

		int fp1 = 0;
		if ((off[1].row + bbox[0].row) < fernData.imgHeight)
			fp1 = ((uchar*) (img->imageData + img->widthStep * (off[1].row
					+ bbox[0].row)))[off[1].col + bbox[0].col];

		if (fp0 > fp1) {
			index |= 1;
		}
		off += 2;
	}
	return index;
}

double measure_bbox_offset(FernData &fernData, IplImage *blur, int idx_bbox, double minVar,
		Eigen::Matrix<double, 10, Eigen::Dynamic>& patt) {

	double conf = 0.0;

	double bboxvar = bbox_var_offset(fernData.integralImg, fernData.integralImg2, fernData.bboxes + idx_bbox * BBOX_STEP,
			fernData.imgHeight);

	if (bboxvar < minVar) {
		return conf;
	}

	for (int i = 0; i < fernData.nTrees; i++) {
		int idx = measure_tree_offset(fernData, blur, idx_bbox, i);
		patt(i, idx_bbox) = idx;
		conf += fernData.weights[i][idx];
	}
	return conf;
}

/*
 *  Initialization (source, grid, features, scales)
 */
void fern1(FernData &fernData, IplImage* source,
		Eigen::Matrix<double, 6, Eigen::Dynamic> const & grid, Eigen::Matrix<
				double, 4 * TLD_NFEATURES, TLD_NTREES> const & features, Eigen::Matrix<
				double, 2, 21> const & scales) {

	fernData.imgHeight = source->height;
	fernData.imgWidth = source->width;
	fernData.nTrees = features.cols();
	fernData.nFeat = features.rows() / 4; // feature has 4 values: x1,y1,x2,y2
	fernData.thrN = 0.5 * fernData.nTrees;
	fernData.nScale = scales.cols();

	fernData.integralImg = new double[fernData.imgHeight * fernData.imgWidth];
	fernData.integralImg2 = new double[fernData.imgHeight * fernData.imgWidth];

	// bboxes
	fernData.mBbox = grid.rows();
	fernData.nBbox = grid.cols();
	fernData.bboxes = create_offsets_bbox(fernData, grid);
	fernData.offsets = create_offsets(fernData, scales, features);

	for (int i = 0; i < fernData.nTrees; i++) {
		fernData.weights.push_back(std::vector<double> (pow(2.0, nBIT * fernData.nFeat), 0));
		fernData.nP.push_back(std::vector<int> (pow(2.0, nBIT * fernData.nFeat), 0));
		fernData.nN.push_back(std::vector<int> (pow(2.0, nBIT * fernData.nFeat), 0));
	}

	for (int i = 0; i < fernData.nTrees; i++) {
		for (unsigned int j = 0; j < fernData.weights[i].size(); j++) {
			fernData.weights[i].at(j) = 0;
			fernData.nP[i].at(j) = 0;
			fernData.nN[i].at(j) = 0;
		}
	}

	return;
}

Eigen::RowVectorXd fern2(FernData &fernData, Eigen::Matrix<double, 10, Eigen::Dynamic> const & X,
		Eigen::VectorXd const & Y, double margin,
		unsigned char bootstrap, Eigen::VectorXd const & idx) {

	int numX = X.cols();
	double thrP = margin * fernData.nTrees;

	int step = numX / 10;

	if (idx(0) == -1) {
		for (int j = 0; j < bootstrap; j++) {

			for (int i = 0; i < step; i++) {
				for (int k = 0; k < 10; k++) {

					int I = k * step + i;
					//double *x = X+nTrees*I;
					if (Y(I) == 1) {
						if (measure_forest(fernData, X.col(I)) <= thrP)
							update(fernData, X.col(I), 1, 1);
					} else {
						if (measure_forest(fernData, X.col(I)) >= fernData.thrN)
							update(fernData, X.col(I), 0, 1);
					}
				}
			}

		}
	} else {

		int nIdx = idx.size(); // ROWVECTOR!


		for (int j = 0; j < bootstrap; j++) {

			for (int i = 0; i < nIdx; i++) {
				int I = idx(i);
				//double *x = X+nTrees*I;
				if (Y(I) == 1) {
					if (measure_forest(fernData, X.col(I)) <= thrP)
						update(fernData, X.col(I), 1, 1);
				} else {
					if (measure_forest(fernData, X.col(I)) >= fernData.thrN)
						update(fernData, X.col(I), 0, 1);
				}
			}

		}
	}

	Eigen::MatrixXd out(1, numX);

	for (int i = 0; i < numX; i++) {
		out(0, i) = measure_forest(fernData, X.col(i));
	}

	return out;
}

Eigen::RowVectorXd fern3(FernData &fernData, Eigen::Matrix<double, 10, Eigen::Dynamic> const & nX2, int n) {

	int numX = n;
	Eigen::RowVectorXd out(numX);

	for (int i = 0; i < numX; i++)
		out(i) = measure_forest(fernData, nX2.col(i));

	return out;
}

void fern4(FernData &fernData, tld::ImgType& img, double maxBBox, double minVar, Eigen::VectorXd& conf,
		Eigen::Matrix<double, 10, Eigen::Dynamic>& patt) {

	for (int i = 0; i < fernData.nBbox; i++)
		conf(i) = -1;

	double probability = maxBBox;
	double nTest = fernData.nBbox * probability;
	if (nTest > fernData.nBbox)
		nTest = fernData.nBbox;

	double pStep = (double) fernData.nBbox / nTest;
	double pState = uniform() * pStep;

	iimg(img.input, fernData.integralImg, fernData.imgHeight, fernData.imgWidth);
	iimg2(img.input, fernData.integralImg2, fernData.imgHeight, fernData.imgWidth);

	unsigned int I = 0;

	while (1) {

		I = (unsigned int) floor(pState);
		pState += pStep;
		if (pState >= fernData.nBbox)
			break;
		conf(I) = measure_bbox_offset(fernData, img.blur, I, minVar, patt);
	}

}

Eigen::Matrix<double, TLD_NTREES, Eigen::Dynamic> fern5(FernData &fernData, tld::ImgType& img, std::vector<int>& idx, double var) {

	// bbox indexes
	int numIdx = idx.size();

	// minimal variance
	if (var > 0) {
		iimg(img.input, fernData.integralImg, fernData.imgHeight, fernData.imgWidth);
		iimg2(img.input, fernData.integralImg2, fernData.imgHeight, fernData.imgWidth);
	}

	// output patterns
	Eigen::MatrixXd patt(fernData.nTrees, numIdx);
	Eigen::MatrixXd status(fernData.nTrees, numIdx);

	for (int j = 0; j < numIdx; j++) {

		if (var > 0) {
			double bboxvar = bbox_var_offset(fernData.integralImg, fernData.integralImg2, fernData.bboxes + j * BBOX_STEP, fernData.imgHeight);
			if (bboxvar < var) {
				status(0, j) = 0;
				continue;
			}
		}

		status(0, j) = 1;
		for (int i = 0; i < fernData.nTrees; i++) {
			patt(i, j) = (double) measure_tree_offset(fernData, img.blur, idx[j], i);
		}
	}
	Eigen::MatrixXd outpattern(fernData.nTrees, numIdx * 2);
	outpattern << patt, status;


	return outpattern;

}

