#include "ImageSource.h"

CvImage ImageSource::getImage() {
	return image;
}

CvImage ImageSource::getGrayImage() {
	return gray;
}

bool ImageSource::nextImage() {
	image = getNextImage();
	if(!image)
		return false;

	gray = CvImage(image.size(), IPL_DEPTH_8U, 1);
	cvCvtColor(image, gray, CV_BGR2GRAY);

	return true;
}



MemoryFeedImageSource::MemoryFeedImageSource() { }

void MemoryFeedImageSource::addImage(CvImage img) {
	images.push_back(img);
}

CvImage MemoryFeedImageSource::getNextImage() {
	if(images.empty())
		return CvImage();

	CvImage img = images.front();
	images.pop_front();
	return img;
}


CvCaptureImageSource::CvCaptureImageSource(CvCapture *source)
: source(source) { }

CvImage CvCaptureImageSource::getNextImage() {
	IplImage *raw = cvQueryFrame(source);
	if(!raw)
		return CvImage();

	// cvQueryFrame() returns a static buffer. Clone it so that the image
	// can be freed after usage.
	return CvImage(cvCloneImage(raw));
}
