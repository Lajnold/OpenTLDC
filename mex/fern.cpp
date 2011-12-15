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

#include "stdio.h"
#include "math.h"
#include <vector>
#include <map>
#include <set>
//#include "tld.h"
#ifdef _CHAR16T
#define CHAR16_T
#endif

#include "mex.h"
#include "../utils/utility.h"

typedef struct {
	int row;
	int col;
} PT;

static double thrN;
static int nBBOX;
static int mBBOX;
static int nTREES;
static int nFEAT;
static int nSCALE;
static int iHEIGHT;
static int iWIDTH;
static PT *BBOX = NULL;
static PT *OFF = NULL;
static double *IIMG = 0;
static double *IIMG2 = 0;
static vector<vector<double> > WEIGHT;
static vector<vector<int> > nP;
static vector<vector<int> > nN;
static int BBOX_STEP = 7;
static int nBIT = 1; // number of bits per feature

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

double bbox_var_offset(double *ii, double *ii2, PT *off, int iHEIGHT) {
	// off[0-3] corners of bbox, off[4] area

	double mX = (ii[off[3].row + off[3].col * iHEIGHT] - ii[off[2].row
			+ off[2].col * iHEIGHT] - ii[off[1].row + off[1].col * iHEIGHT]
			+ ii[off[0].row + off[0].col * iHEIGHT]) / (double) off[4].row;

	double mX2 = (ii2[off[3].row + off[3].col * iHEIGHT] - ii2[off[2].row
			+ off[2].col * iHEIGHT] - ii2[off[1].row + off[1].col * iHEIGHT]
			+ ii2[off[0].row + off[0].col * iHEIGHT]) / (double) off[4].row;

	return mX2 - mX * mX;
}

void update(Eigen::Matrix<double, 10, 1> x, int C, int N) {
	for (int i = 0; i < nTREES; i++) {

		int idx = (int) x(i);

		(C == 1) ? nP[i][idx] += N : nN[i][idx] += N;

		if (nP[i][idx] == 0) {
			WEIGHT[i][idx] = 0;
		} else {
			WEIGHT[i][idx] = ((double) (nP[i][idx]))
					/ (nP[i][idx] + nN[i][idx]);
		}
	}
}

double measure_forest(Eigen::Matrix<double, 10, 1> idx) {
	double votes = 0;
	for (int i = 0; i < nTREES; i++) {
		votes += WEIGHT[i][idx(i)];
	}
	return votes;
}

PT* create_offsets_bbox(
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> bb) {

	PT *offsets = (PT*) malloc(BBOX_STEP * nBBOX * sizeof(PT));
	PT *off = offsets;

	for (int i = 0; i < nBBOX; i++) {
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
		(*off).row = (int) (bb(4, i) - 1) * 2 * nFEAT * nTREES; // 5
		off++;
		(*off).row = bb(5, i); // 6
		off++;
	}
	return offsets;
}

PT* create_offsets(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> scale,
		Eigen::Matrix<double, 4 * TLD_NFEATURES, TLD_NTREES> x) {


	PT *offsets = (PT*) malloc(nSCALE * nTREES * nFEAT * 2 * sizeof(PT));
	PT *off = offsets;

	for (int k = 0; k < nSCALE; k++) {
		//double *scale = scale0 + 2 * k;
		for (int i = 0; i < nTREES; i++) {
			for (int j = 0; j < nFEAT; j++) {
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

int measure_tree_offset(IplImage* img, int idx_bbox, int idx_tree) {

	int index = 0;

	PT *bbox = BBOX + idx_bbox * BBOX_STEP;
	PT *off = OFF + bbox[5].row + idx_tree * 2 * nFEAT;

	for (int i = 0; i < nFEAT; i++) {
		index <<= 1;
		int fp0 = 0;
		if ((off[0].row + bbox[0].row) < iHEIGHT)
			fp0 = ((uchar*) (img->imageData + img->widthStep * (off[0].row
					+ bbox[0].row)))[off[0].col + bbox[0].col];

		int fp1 = 0;
		if ((off[1].row + bbox[0].row) < iHEIGHT)
			fp1 = ((uchar*) (img->imageData + img->widthStep * (off[1].row
					+ bbox[0].row)))[off[1].col + bbox[0].col];

		if (fp0 > fp1) {
			index |= 1;
		}
		off += 2;
	}
	return index;
}

double measure_bbox_offset(IplImage *blur, int idx_bbox, double minVar,
		Eigen::Matrix<double, 10, Eigen::Dynamic>& patt) {

	double conf = 0.0;

	double bboxvar = bbox_var_offset(IIMG, IIMG2, BBOX + idx_bbox * BBOX_STEP,
			iHEIGHT);

	if (bboxvar < minVar) {
		return conf;
	}

	for (int i = 0; i < nTREES; i++) {
		int idx = measure_tree_offset(blur, idx_bbox, i);
		patt(i, idx_bbox) = idx;
		conf += WEIGHT[i][idx];
	}
	return conf;
}

/*
 *  Cleanup
 */
void fern0() {

	thrN = 0;
	nBBOX = 0;
	mBBOX = 0;
	nTREES = 0;
	nFEAT = 0;
	nSCALE = 0;
	iHEIGHT = 0;
	iWIDTH = 0;

	free(BBOX);
	BBOX = 0;
	free(OFF);
	OFF = 0;
	free(IIMG);
	IIMG = 0;
	free(IIMG2);
	IIMG2 = 0;
	WEIGHT.clear();
	nP.clear();
	nN.clear();
	return;
}

/*
 *  Initialization (source, grid, features, scales)
 */
void fern1(IplImage* source,
		Eigen::Matrix<double, 6, Eigen::Dynamic> const & grid, Eigen::Matrix<
				double, 4 * TLD_NFEATURES, TLD_NTREES> const & features, Eigen::Matrix<
				double, 2, 21> const & scales) {

	iHEIGHT = source->height;
	iWIDTH = source->width;
	nTREES = features.cols();
	nFEAT = features.rows() / 4; // feature has 4 values: x1,y1,x2,y2
	thrN = 0.5 * nTREES;
	nSCALE = scales.cols();

	IIMG = (double*) malloc(iHEIGHT * iWIDTH * sizeof(double));
	IIMG2 = (double*) malloc(iHEIGHT * iWIDTH * sizeof(double));

	// BBOX
	mBBOX = grid.rows();
	nBBOX = grid.cols();
	BBOX = create_offsets_bbox(grid);
	OFF = create_offsets(scales, features);

	for (int i = 0; i < nTREES; i++) {
		WEIGHT.push_back(vector<double> (pow(2.0, nBIT * nFEAT), 0));
		nP.push_back(vector<int> (pow(2.0, nBIT * nFEAT), 0));
		nN.push_back(vector<int> (pow(2.0, nBIT * nFEAT), 0));
	}

	for (int i = 0; i < nTREES; i++) {
		for (unsigned int j = 0; j < WEIGHT[i].size(); j++) {
			WEIGHT[i].at(j) = 0;
			nP[i].at(j) = 0;
			nN[i].at(j) = 0;
		}
	}

	return;
}

Eigen::RowVectorXd fern2(Eigen::Matrix<double, 10, Eigen::Dynamic> const & X,
		Eigen::VectorXd const & Y, double margin,
		unsigned char bootstrap, Eigen::VectorXd const & idx) {

	int numX = X.cols();
	double thrP = margin * nTREES;

	int step = numX / 10;

	if (idx(0) == -1) {
		for (int j = 0; j < bootstrap; j++) {

			for (int i = 0; i < step; i++) {
				for (int k = 0; k < 10; k++) {

					int I = k * step + i;
					//double *x = X+nTREES*I;
					if (Y(I) == 1) {
						if (measure_forest(X.col(I)) <= thrP)
							update(X.col(I), 1, 1);
					} else {
						if (measure_forest(X.col(I)) >= thrN)
							update(X.col(I), 0, 1);
					}
				}
			}

		}
	} else {

		int nIdx = idx.size(); // ROWVECTOR!


		for (int j = 0; j < bootstrap; j++) {

			for (int i = 0; i < nIdx; i++) {
				int I = idx(i);
				//double *x = X+nTREES*I;
				if (Y(I) == 1) {
					if (measure_forest(X.col(I)) <= thrP)
						update(X.col(I), 1, 1);
				} else {
					if (measure_forest(X.col(I)) >= thrN)
						update(X.col(I), 0, 1);
				}
			}

		}
	}

	Eigen::MatrixXd out(1, numX);

	for (int i = 0; i < numX; i++) {
		out(0, i) = measure_forest(X.col(i));
	}

	return out;
}

Eigen::RowVectorXd fern3(Eigen::Matrix<double, 10, Eigen::Dynamic> const & nX2, int n) {

	int numX = n;
	Eigen::RowVectorXd out(numX);

	for (int i = 0; i < numX; i++)
		out(i) = measure_forest(nX2.col(i));

	return out;
}

void fern4(ImgType& img, double maxBBox, double minVar, Eigen::VectorXd& conf,
		Eigen::Matrix<double, 10, Eigen::Dynamic>& patt) {

	for (int i = 0; i < nBBOX; i++)
		conf(i) = -1;

	double probability = maxBBox;
	double nTest = nBBOX * probability;
	if (nTest > nBBOX)
		nTest = nBBOX;

	double pStep = (double) nBBOX / nTest;
	double pState = uniform() * pStep;

	iimg(img.input, IIMG, iHEIGHT, iWIDTH);
	iimg2(img.input, IIMG2, iHEIGHT, iWIDTH);

	unsigned int I = 0;

	while (1) {

		I = (unsigned int) floor(pState);
		pState += pStep;
		if (pState >= nBBOX)
			break;
		conf(I) = measure_bbox_offset(img.blur, I, minVar, patt);
	}

}

Eigen::Matrix<double, TLD_NTREES, Eigen::Dynamic> fern5(ImgType& img, std::vector<
		int>& idx, double var) {

	// bbox indexes
	int numIdx = idx.size();

	// minimal variance
	if (var > 0) {
		iimg(img.input, IIMG, iHEIGHT, iWIDTH);
		iimg2(img.input, IIMG2, iHEIGHT, iWIDTH);
	}

	// output patterns
	Eigen::MatrixXd patt(nTREES, numIdx);
	Eigen::MatrixXd status(nTREES, numIdx);

	for (int j = 0; j < numIdx; j++) {

		if (var > 0) {
			double bboxvar = bbox_var_offset(IIMG, IIMG2, BBOX + j * BBOX_STEP,
					iHEIGHT);
			if (bboxvar < var) {
				status(0, j) = 0;
				continue;
			}
		}

		status(0, j) = 1;
		for (int i = 0; i < nTREES; i++) {
			patt(i, j) = (double) measure_tree_offset(img.blur, idx[j], i);
		}
	}
	Eigen::MatrixXd outpattern(nTREES, numIdx * 2);
	outpattern << patt, status;


	return outpattern;

}

