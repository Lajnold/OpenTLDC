#ifndef IMAGESOURCE_H_
#define IMAGESOURCE_H_

#include <list>

#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>

namespace tld {

class ImageSource {
public:
	bool nextImage();
	cv::Mat getImage();
	cv::Mat getGrayImage();

	virtual ~ImageSource() { };

protected:
	ImageSource() { }

	virtual cv::Mat getNextImage() = 0;

private:
	cv::Mat image;
	cv::Mat gray;
};


class MemoryFeedImageSource : public ImageSource {
public:
	MemoryFeedImageSource();

	void addImage(cv::Mat img);

protected:
	cv::Mat getNextImage();

private:
	std::list<cv::Mat> images;
};


class CvCaptureImageSource : public ImageSource {
public:
	CvCaptureImageSource(cv::VideoCapture *source);

protected:
	cv::Mat getNextImage();

private:
	cv::VideoCapture *source;
};

} // namespace tld

#endif
