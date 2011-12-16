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

#include <iostream>

#include "tld/tld.h"

namespace tld {

/**
 *  Loop function of TLDC.
 *
 *  @param opt tldStruct structure with initial values and thresholds
 *  @param cfg stream settings and initial bounding box
 */
void tldExample(TldStruct& tldStruct, bool display) {

	srand(0);

	double t = (double)cv::getTickCount();

	/* INITIALIZATION -------------------------------------- */

	if(!tldInit(tldStruct))
		return; // No images.

	if(display)
		tldDisplay(0, 0, tldStruct, t);

	/* RUN-TIME -------------------------------------------- */

	unsigned long i = 1;

	while (i < 2500) {

		t = (double)cv::getTickCount();

		if(!tldProcessFrame(tldStruct))
			break; // Out of images.

		t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();

		if (display)
			tldDisplay(1, i, tldStruct, t);

		std::cout << "BB - xmin: " << tldStruct.currentBB(0) << " ymin: "
				<< tldStruct.currentBB(1) << " xmax: " << tldStruct.currentBB(2) << " ymax: "
				<< tldStruct.currentBB(3) << std::endl;

		i++;

	}

	if(display)
		cvDestroyWindow("Result");
}

} // namespace tld
