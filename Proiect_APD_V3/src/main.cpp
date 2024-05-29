#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <chrono>

// Declarations of the CUDA functions with extern "C"
extern "C" void resizeImage(const cv::Mat & input, cv::Mat & output, int newWidth, int newHeight);
extern "C" void convertToBlackAndWhite(const cv::Mat & input, cv::Mat & output);

using namespace cv;
using namespace std;
using namespace chrono;

int main(int argc, char** argv) {
	string inputPath = "../Proiect_CUDA/input/big_image.tif";
	string outputPath = "../Proiect_CUDA/output/";

	Mat img = imread(inputPath);
	if (img.empty()) {
		cout << "Error: The image could not be loaded" << endl;
		cout << "Check the file path: " << inputPath << endl;
		return -1;
	}

	int newWidth, newHeight;
	cout << "Please define the new size of the image\n";
	cout << "Width: ";
	cin >> newWidth;
	cout << "Height: ";
	cin >> newHeight;
	cout << endl;

	Mat resizedImg(newHeight, newWidth, CV_8UC3);
	Mat blackAndWhiteImg(newHeight, newWidth, CV_8UC1);

	auto start = high_resolution_clock::now();
	resizeImage(img, resizedImg, newWidth, newHeight);
	auto stop = high_resolution_clock::now();
	auto durationResize = duration_cast<milliseconds>(stop - start);
	cout << "Time taken to resize: " << durationResize.count() << " milliseconds" << endl;

	start = high_resolution_clock::now();
	convertToBlackAndWhite(resizedImg, blackAndWhiteImg);
	stop = high_resolution_clock::now();
	auto durationConvert = duration_cast<milliseconds>(stop - start);
	cout << "Time taken to convert to black and white: " << durationConvert.count() << " milliseconds" << endl;

	string outputFilename = outputPath + "Resized_small_image.tif";
	imwrite(outputFilename, resizedImg);
	outputFilename = outputPath + "BlackWhite_small_image.tif";
	imwrite(outputFilename, blackAndWhiteImg);

	return 0;
}
