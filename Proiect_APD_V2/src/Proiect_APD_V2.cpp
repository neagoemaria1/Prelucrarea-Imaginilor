#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <mpi.h>

using namespace cv;
using namespace std;
using namespace chrono;

void resizeImagePixelByPixel(const Mat& input, Mat& output, int newWidth, int newHeight, int startRow, int numRows)
{
	for (int y = startRow; y < startRow + numRows; ++y)
	{
		for (int x = 0; x < newWidth; ++x)
		{
			int srcX = static_cast<int>(x * static_cast<double>(input.cols) / newWidth);
			int srcY = static_cast<int>(y * static_cast<double>(input.rows) / newHeight);

			output.at<Vec3b>(y - startRow, x) = input.at<Vec3b>(srcY, srcX);
		}
	}
}

void convertToBlackAndWhite(const Mat& input, Mat& output, int startRow, int numRows)
{
	for (int y = startRow; y < startRow + numRows; ++y)
	{
		for (int x = 0; x < input.cols; ++x)
		{
			int blue = input.at<Vec3b>(y, x)[0];
			int green = input.at<Vec3b>(y, x)[1];
			int red = input.at<Vec3b>(y, x)[2];
			int intensity = 0.299 * red + 0.587 * green + 0.114 * blue;
			output.at<uchar>(y - startRow, x) = intensity;
		}
	}
}
void saveSubImage(const Mat& mat, const string& matName, int rank)
{
	string fileName = matName + "_process_" + to_string(rank) + ".png";
	imwrite(fileName, mat);
	cout << "Process " << rank << " saved " << matName << " as " << fileName << "\n\n";
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	string inputPath = "C:/Users/Percyka/Desktop/apd/Proiect_V2/Proiect_APD/input/big_image.tif";
	string outputPath = "C:/Users/Percyka/Desktop/apd/Proiect_V2/Proiect_APD/output/";

	Mat img;
	int newWidth, newHeight;

	if (rank == 0) {
		img = imread(inputPath);
		if (img.empty())
		{
			cout << "Error: The image could not be loaded" << endl;
			MPI_Abort(MPI_COMM_WORLD, -1);
			return -1;
		}
		cout << "Please define the new size of the image\n";
		cout << "Width: ";
		cin >> newWidth;
		cout << "Height: ";
		cin >> newHeight;
		cout << endl;
	}
	// Process 0 sends the dimensions to the other processes
	MPI_Bcast(&newWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&newHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Broadcast image size information to all processes
	int imgType;
	Size imgSize;
	if (rank == 0) {
		imgSize = img.size();
		imgType = img.type();
	}
	MPI_Bcast(&imgSize, sizeof(Size), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&imgType, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank != 0) {
		img = Mat(imgSize, imgType);
	}

	// Broadcast the image data
	MPI_Bcast(img.data, img.total() * img.elemSize(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	Mat resizedImg(newHeight, newWidth, CV_8UC3);
	Mat blackAndWhiteImg(newHeight, newWidth, CV_8UC1);

	// Calculation of the number of lines per process and the remaining lines
	int rowsPerProcess = newHeight / size;
	int remainingRows = newHeight % size;

	int startRow = rank * rowsPerProcess;
	int numRows = (rank == size - 1) ? (rowsPerProcess + remainingRows) : rowsPerProcess;

	Mat subResized(numRows, newWidth, CV_8UC3);
	Mat subBW(numRows, newWidth, CV_8UC1);

	MPI_Barrier(MPI_COMM_WORLD);
	auto start = high_resolution_clock::now();
	// Each process resizes a section of the image
	resizeImagePixelByPixel(img, subResized, newWidth, newHeight, startRow, numRows);

	auto stop = high_resolution_clock::now();
	auto durationResize = duration_cast<milliseconds>(stop - start);
	if (rank == 0) {
		cout << "Time taken to resize: " << durationResize.count() << " milliseconds" << endl;
	}
	saveSubImage(subResized, outputPath + "subResize", rank);

	MPI_Gather(subResized.data, numRows * newWidth * 3, MPI_UNSIGNED_CHAR,
		resizedImg.data, rowsPerProcess * newWidth * 3, MPI_UNSIGNED_CHAR,
		0, MPI_COMM_WORLD);

	// Handle remaining rows if there are any
	if (rank == size - 1 && remainingRows > 0) {
		MPI_Send(subResized.data + rowsPerProcess * newWidth * 3, remainingRows * newWidth * 3, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
	}

	if (rank == 0 && remainingRows > 0) {
		MPI_Recv(resizedImg.data + rowsPerProcess * size * newWidth * 3, remainingRows * newWidth * 3, MPI_UNSIGNED_CHAR, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	MPI_Bcast(resizedImg.data, resizedImg.total() * resizedImg.elemSize(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	auto startConvert = high_resolution_clock::now();

	convertToBlackAndWhite(resizedImg, subBW, startRow, numRows);

	auto stopConvert = high_resolution_clock::now();
	auto durationConvert = duration_cast<milliseconds>(stopConvert - startConvert);
	if (rank == 0) {
		cout << "Time taken to convert to black and white: " << durationConvert.count() << " milliseconds" << endl;
	}

	saveSubImage(subBW, outputPath + "subBW", rank);

	MPI_Gather(subBW.data, numRows * newWidth, MPI_UNSIGNED_CHAR,
		blackAndWhiteImg.data, rowsPerProcess * newWidth, MPI_UNSIGNED_CHAR,
		0, MPI_COMM_WORLD);

	// Handle remaining rows if there are any
	if (rank == size - 1 && remainingRows > 0) {
		MPI_Send(subBW.data + rowsPerProcess * newWidth, remainingRows * newWidth, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
	}

	if (rank == 0 && remainingRows > 0) {
		MPI_Recv(blackAndWhiteImg.data + rowsPerProcess * size * newWidth, remainingRows * newWidth, MPI_UNSIGNED_CHAR, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	if (rank == 0) {
		string outputFilename = outputPath + "Resized_big_image.tif";
		imwrite(outputFilename, resizedImg);
		outputFilename = outputPath + "BlackWhite_big_image.tif";
		imwrite(outputFilename, blackAndWhiteImg);
	}

	MPI_Finalize();
	return 0;
}
