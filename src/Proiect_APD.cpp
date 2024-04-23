#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

void resizeImagePixelByPixel(const Mat& input, Mat& output, int newWidth, int newHeight)
{
	output = Mat(newHeight, newWidth, input.type());

	double widthRatio = static_cast<double>(input.cols) / newWidth;
	double heightRatio = static_cast<double>(input.rows) / newHeight;

	for (int y = 0; y < newHeight; ++y)
	{
		for (int x = 0; x < newWidth; ++x)
		{
			int srcX = static_cast<int>(x * widthRatio);
			int srcY = static_cast<int>(y * heightRatio);

			output.at<Vec3b>(y, x) = input.at<Vec3b>(srcY, srcX);
		}
	}
}

void convertToBlackAndWhite(const Mat& input, Mat& output)
{
	output = Mat(input.size(), CV_8UC1);

	for (int y = 0; y < input.rows; ++y)
	{
		for (int x = 0; x < input.cols; ++x)
		{
			int blue = input.at<Vec3b>(y, x)[0];
			int green = input.at<Vec3b>(y, x)[1];
			int red = input.at<Vec3b>(y, x)[2];
			int intensity = 0.299 * red + 0.587 * green + 0.114 * blue;
			output.at<uchar>(y, x) = intensity;

		}
	}
}

int main()
{
	string inputPath = "input/Cat2_1920x1080.jpg";
	string outputPath = "output/";

	Mat img = imread(inputPath);
	if (img.empty())
	{
		cout << "Error: The image could not be loaded" << endl;
		return -1;
	}

	int newWidth, newHeight;
	cout << "Please define the new size of the image\n";
	cout << "Width: ";
	cin >> newWidth;
	cout << "Height: ";
	cin >> newHeight;
	cout << endl;
	Mat resizedImg;

	auto start = high_resolution_clock::now();
	resizeImagePixelByPixel(img, resizedImg, newWidth, newHeight);
	auto stop = high_resolution_clock::now();
	auto durationResize = duration_cast<milliseconds>(stop - start);
	cout << "Time taken to resize: " << durationResize.count() << " milliseconds" << endl;


	Mat blackAndWhiteImg;
	auto startConvert = high_resolution_clock::now();
	convertToBlackAndWhite(resizedImg, blackAndWhiteImg);
	auto stopConvert = high_resolution_clock::now();
	auto durationConvert = duration_cast<milliseconds>(stopConvert - startConvert);
	cout << "Time taken to convert to black and white: " << durationConvert.count() << " milliseconds" << endl;


	string outputFilename = outputPath + "Resized_Cat2_1920x1080.tif";
	imwrite(outputFilename, resizedImg);
	outputFilename = outputPath + "BlackWhite_Cat2_1920x1080.tif";
	imwrite(outputFilename, blackAndWhiteImg);
	waitKey(0);

	return 0;
}