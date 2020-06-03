#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

const char *pathOfDataFile = "data.xml";
const char *pathOfTest = "data/test/*.png";
const char *pathOfTrain = "data/train/*.png";

vector<vector<float>> extractFeatures(Mat image) {
	vector<Vec4i> hierarchy;
	vector<vector<float>> output;
	vector<vector<Point>> contours;

	findContours(image, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	if(contours.empty()) return output;

	for(int i = 0; i < contours.size(); i++) {
		Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1);
		drawContours(mask, contours, i, Scalar(1), FILLED, LINE_8, hierarchy, 1);

		float area = (float) sum(mask)[0];
		if(area > 500) {
			RotatedRect rect = minAreaRect(contours[i]);
			float width = rect.size.width;
			float height = rect.size.height;
			float aspectRatio = (width < height) ? height / width : width / height;

			vector<float> row;
			row.push_back(area);
			row.push_back(aspectRatio);
			output.push_back(row);
		}
	}
	return output;
}

void readAndExtractFeatures(const char *pathOfFolder, vector<float> &data, vector<int> &dataResponse) {
	vector<string> fileNames;
	glob(pathOfFolder, fileNames);

	cout << format("\n Reading started in the %s folder\n", pathOfFolder);
	for(int imageIndex = 0; imageIndex < fileNames.size(); imageIndex++) {
		Mat image = imread(fileNames.at(imageIndex), IMREAD_GRAYSCALE);
		threshold(image, image, 90, 255, THRESH_BINARY);

		vector<vector<float>> features = extractFeatures(image);
		for(int i = 0; i < features.size(); i++) {
			data.push_back(features[i][0]);
			data.push_back(features[i][1]);
			dataResponse.push_back(fileNames[imageIndex][fileNames[imageIndex].size() - 6] - 48);
		}
		cout << format("\r %d out of %d were process. (%.2f%%) ",
					   imageIndex + 1, fileNames.size(), 100 * ((float) imageIndex + 1) / fileNames.size());
	}
	cout << endl;
}

bool readFromFile(Mat &testMat, Mat &trainMat, Mat &testResponses, Mat &trainResponses) {
	FileStorage fileRead(pathOfDataFile, cv::FileStorage::READ);

	if(fileRead.isOpened()) {
		cout << format("\n Reading started in the %s data file\n", pathOfDataFile);

		fileRead["testMat"] >> testMat;
		fileRead["trainMat"] >> trainMat;
		fileRead["testResponses"] >> testResponses;
		fileRead["trainResponses"] >> trainResponses;

		return true;
	}
	return false;
}

void writeToFile(Mat &testMat, Mat &trainMat, Mat &testResponses, Mat &trainResponses) {
	cout << format("\n Writing started to %s data file\n", pathOfDataFile);

	FileStorage fileWrite("data.xml", cv::FileStorage::WRITE);
	fileWrite << "testMat" << testMat;
	fileWrite << "trainMat" << trainMat;
	fileWrite << "testResponses" << testResponses;
	fileWrite << "trainResponses" << trainResponses;

	cout << format("\n %s data file created\n", pathOfDataFile);
}

void trainAndTest() {
	Mat testMat;
	Mat trainMat;
	Mat testResponses;
	Mat trainResponses;

	if(!readFromFile(testMat, trainMat, testResponses, trainResponses)) {
		cout << "\n Failed to open data.xml data file\n";

		vector<float> test;
		vector<float> train;
		vector<int> testResponse;
		vector<int> trainResponse;

		readAndExtractFeatures(pathOfTest, test, testResponse);
		readAndExtractFeatures(pathOfTrain, train, trainResponse);

		testMat = Mat((int) test.size() / 2, 2, CV_32FC1, &test[0]);
		trainMat = Mat((int) train.size() / 2, 2, CV_32FC1, &train[0]);
		testResponses = Mat((int) testResponse.size(), 1, CV_32SC1, testResponse[0]);
		trainResponses = Mat((int) trainResponse.size(), 1, CV_32SC1, &trainResponse[0]);

		writeToFile(testMat, trainMat, testResponses, trainResponses);
	}

	cout << format("\n Size of Test Mat: %d"
				   "\n Size of Train Mat: %d"
				   "\n Size of Test Responses: %d"
				   "\n Size of Train Responses: %d",
				   testMat.size().height, trainMat.size().height,
				   testResponses.size().height, trainResponses.size().height);
}

int main(int argc, char **argv) {
	trainAndTest();
	return EXIT_SUCCESS;
}
