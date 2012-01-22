#include "tld/ImageSource.h"

namespace tld {

cv::Mat ImageSource::getImage() {
	return image;
}

cv::Mat ImageSource::getGrayImage() {
	return gray;
}

bool ImageSource::nextImage() {
	image = getNextImage();
	if(image.empty())
		return false;

	// Create a new matrix so that the previous gray isn't overwritten.
	gray = cv::Mat();
	cv::cvtColor(image, gray, CV_BGR2GRAY);

	return true;
}



MemoryFeedImageSource::MemoryFeedImageSource() { }

void MemoryFeedImageSource::addImage(cv::Mat img) {
	images.push_back(img);
}

cv::Mat MemoryFeedImageSource::getNextImage() {
	if(images.empty())
		return cv::Mat();

	cv::Mat img = images.front();
	images.pop_front();
	return img;
}


CvCaptureImageSource::CvCaptureImageSource(cv::VideoCapture *source)
: source(source) { }

cv::Mat CvCaptureImageSource::getNextImage() {
	cv::Mat ret;
	(*source) >> ret;
	return ret;
}

} // namespace tld
