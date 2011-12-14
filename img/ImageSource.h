#ifndef IMAGESOURCE_H_
#define IMAGESOURCE_H_

#include <list>

#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>

class ImageSource {
public:
	bool nextImage();
	CvImage getImage();
	CvImage getGrayImage();

	virtual ~ImageSource() { };

protected:
	ImageSource() { }

	virtual CvImage getNextImage() = 0;

private:
	CvImage image;
	CvImage gray;
};


class MemoryFeedImageSource : public ImageSource {
public:
	MemoryFeedImageSource();

	void addImage(CvImage img);

protected:
	CvImage getNextImage();

private:
	std::list<CvImage> images;
};


class CvCaptureImageSource : public ImageSource {
public:
	CvCaptureImageSource(CvCapture *source);

protected:
	CvImage getNextImage();

private:
	CvCapture *source;
};

#endif
