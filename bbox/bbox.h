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

#ifndef BBOX_H_
#define BBOX_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>

using namespace cv;

#include "../tld/structs.h"

typedef struct {
	int width;
	int height;
} BBOXSIZE;

/*This function returns the width of a given bounding box*/
inline int bb_width(Eigen::Vector4d const & bb) {
	return (bb(2) - bb(0));
}
/*This function returns the height of a given bounding box */
inline int bb_height(Eigen::Vector4d const & bb) {
	return (bb(3) - bb(1));
}
/*This function returns the size of a given bounding box */
inline BBOXSIZE bb_size(Eigen::Vector4d const & bb) {
	int x = bb_width(bb);
	int y = bb_height(bb);
	BBOXSIZE Boxsize = { x, y };
	return Boxsize;
}

/*This function returns the center of a given bounding box */
inline Eigen::Vector2d bb_center(Eigen::Vector4d const & bb) {
	Eigen::Vector2d center;
	center(0) = (bb(0) + bb(2)) / 2;
	center(1) = (bb(1) + bb(3)) / 2;
	return center;
}

/*Returns the scale of a give bounding box.*/
inline int bb_scale(Eigen::Vector4d const & bb) {
	int scale = 0;
	scale = sqrt((bb(3) - bb(2) + 1) * (bb(1) - bb(0) + 1));
	return scale;
}

/* Converts a given bounding box to a square and returns it. */
inline Eigen::Vector4d bb_square(Eigen::Vector4d const & bb) {
	int s = 0;
	s = bb_scale(bb);
	Eigen::Vector2d c;
	c = bb_center(bb);
	Eigen::Vector4d bbout;
	bbout(0) = c(0) - s / 2;
	bbout(1) = c(1) - s / 2;
	bbout(2) = c(0) + s / 2;
	bbout(3) = c(1) + s / 2;
	return bbout;
}

/*Checks if a given bounding box fits into a given image size.*/
inline bool bb_isin(Eigen::Vector4d const & bb, Eigen::Vector2i const & imsize) {
	return ((bb(0) >= 0) && (bb(1) >= 0) && (bb(2) < imsize(0)) && (bb(3)
			< imsize(1)));
}
/*Checks if a given bounding box is well defined*/
inline bool bb_isdef(Eigen::VectorXd const & bb) {
	for (int i = 0; i < bb.cols(); i++) {
		if (std::isnan(bb(i)) || std::isinf(bb(i))) {
			return false;
		}
	}
	return true;
}
/*Checks if a given bounding box does not fit into a given image size.*/
inline bool bb_isout(Eigen::VectorXd const & bb, Eigen::Vector2i const & imsize) {
	return ((bb(0) < 0) || (bb(1) < 0) || (bb(2) > imsize(0)) || (bb(3)
			> imsize(1)));
}
/*Returns a bounding box, that contains the min/max x/y values of all bounding boxes*/
inline Eigen::Vector4d bb_hull(Eigen::MatrixXd const & bb0) {
	Eigen::Vector4d bb;
	bb(0) = bb0.row(0).minCoeff();
	bb(1) = bb0.row(1).minCoeff();
	bb(2) = bb0.row(2).maxCoeff();
	bb(3) = bb0.row(3).maxCoeff();
	return bb;
}
/*Rescales a given bounding box*/
inline Eigen::Vector4d bb_rescalerel(Eigen::Vector4d const & bb,
		Eigen::Vector2d & s) {
	if (isnan(s(1))) {
		s(1) = s(0);
	}
	int s1 = 0.5 * (s(0) - 1) * bb_width(bb);
	int s2 = 0.5 * (s(1) - 1) * bb_height(bb);
	Eigen::Vector4d bbout;
	bbout(0) = bb(0) + s1 * -1;
	bbout(1) = bb(1) + s2 * -1;
	bbout(2) = bb(2) + s1;
	bbout(3) = bb(3) + s2;
	return bbout;
}

/**
 * The function draws a rectangle with coordinates given by a vector and a defined color.
 */
inline void bb_draw(IplImage* img, Eigen::Vector4d const & bb, Scalar color,
		unsigned int linewidth) {
	if (bb.isZero()) {
		return;
	} else {
		if (bb(2) - bb(0) > 0 && bb(3) - bb(1) > 0) {
			cvRectangle(img, cvPoint(bb(0), bb(1)), cvPoint(bb(2), bb(3)),
					color, linewidth, 8, 0);
		} else {
			return;
		}
	}
}

/**
 * If no drawing color is set. It will be set to yellow.
 */
inline void bb_draw_add_color(IplImage* img, Eigen::Vector4d const & bb) {
	if (bb.isZero()) {
		return;
	} else {
		Scalar yellow = Scalar(0, 255, 255);
		bb_draw(img, bb, yellow, 2);
	}
}

Eigen::Matrix<double, 2, 150> bb_points(Eigen::VectorXd const & bb, int row,
		int col, int margin, int & size);

/*Predicts where the tracked bounding box will be in the next frame*/
Eigen::Vector4d bb_predict(Eigen::VectorXd const & bb,
		Eigen::MatrixXd const & xFI, Eigen::MatrixXd const & xFJ);

void bb_scan(TldStruct& tld, Eigen::Vector4d const & bb,
		Eigen::Vector2i imsize, int minwin);

Eigen::Matrix<double, 4, Eigen::Dynamic> bb_cluster_confidence(Eigen::Matrix<
		double, 4, 20> const & iBB, Eigen::Matrix<double, 20, 1> const & iConf,
		int nD);

/*Calculates the distance between two given bounding boxes*/
Eigen::MatrixXd bb_distance(Eigen::MatrixXd const & bb1,
		Eigen::MatrixXd const & bb2);

/*Calculates the distance between a given bounding box and all
 * other saved bounding boxes.
 */
Eigen::MatrixXd bb_distance(Eigen::MatrixXd const & bb1);

#endif /* BBOX_H_ */
