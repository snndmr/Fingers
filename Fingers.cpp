#include <iostream>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

const char *pathOfTest = "data/test/*.png";
const char *pathOfTrain = "data/train/*.png";

Mat preprocess(Mat image) {
	if(image.channels() == 3) cvtColor(image, image, COLOR_RGB2GRAY);
	threshold(image, image, 90, 255, THRESH_BINARY);
	return image;
}

vector<vector<float>> extractFeatures(Mat image) {
	vector<Vec4i> hierarchy;
	vector<vector<float>> output;
	vector<vector<Point>> contours;

	findContours(image, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

	if(contours.empty()) return output;

	for(int i = 0; i < contours.size(); i++) {
		Mat mask = Mat::zeros(image.rows, image.cols, CV_8UC1);
		drawContours(mask, contours, i, Scalar(1), FILLED, LINE_8, hierarchy, 1);

		float area = (double) sum(mask)[0];
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

void readAndExtractFeatures(const char *folderPath, vector<float> &data, vector<float> &dataResponse) {
	vector<string> fileNames;
	glob(folderPath, fileNames);

	cout << format("\n Reading in %s started.\n", folderPath);
	for(int imageIndex = 0; imageIndex < fileNames.size(); imageIndex++) {
		Mat image = imread(fileNames.at(imageIndex));

		vector<vector<float>> features = extractFeatures(preprocess(image));
		for(int i = 0; i < features.size(); i++) {
			data.push_back(features[i][0]);
			data.push_back(features[i][1]);
			dataResponse.push_back(fileNames[imageIndex][fileNames[imageIndex].size() - 6]);
			//cout << features[i][0] << " " << features[i][1] << " " << fileNames[imageIndex][fileNames[imageIndex].size() - 6] << endl;
		}
		cout << format("\r %d out of %d were read. (%.2lf%%) ",
					   imageIndex + 1, fileNames.size(), 100 * ((double) imageIndex + 1) / fileNames.size());
	}
}

void trainAndTest() {
	vector<float> test;
	vector<float> testResponse;
	vector<float> train;
	vector<float> trainResponse;

	readAndExtractFeatures(pathOfTest, test, testResponse);
	readAndExtractFeatures(pathOfTrain, train, trainResponse);

	cout << endl << format(" Number of train samples: %4zd", testResponse.size());
	cout << endl << format(" Number of test samples : %4zd", trainResponse.size());
}

int main(int argc, char **argv) {
	trainAndTest();
	return EXIT_FAILURE;
}