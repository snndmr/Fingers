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

void readAndExtractFeatures(const char *folderPath, vector<float> &data, vector<int> &dataResponse) {
	vector<string> fileNames;
	glob(folderPath, fileNames);

	cout << format("\n Reading in %s started.\n", folderPath);
	for(int imageIndex = 0; imageIndex < fileNames.size(); imageIndex++) {
		Mat image = imread(fileNames.at(imageIndex));

		vector<vector<float>> features = extractFeatures(preprocess(image));
		for(int i = 0; i < features.size(); i++) {
			data.push_back(features[i][0]);
			data.push_back(features[i][1]);
			int label = fileNames[imageIndex][fileNames[imageIndex].size() - 6] - 48;
			dataResponse.push_back(label);
		}
		cout << format("\r %d out of %d were process. (%.2lf%%) ",
					   imageIndex + 1, fileNames.size(), 100 * ((double) imageIndex + 1) / fileNames.size());
	}
	cout << endl;
}

Ptr<SVM> trainAndTest() {
	Mat testMat;
	Mat trainMat;
	Mat testResponses;
	Mat trainResponses;

	vector<float> test;
	vector<int> testResponse;
	vector<float> train;
	vector<int> trainResponse;

	//To read from data.xml file.
	FileStorage fileRead;
	fileRead.open("data.xml", cv::FileStorage::READ);

	if(!fileRead.isOpened()) {
		cout << "\n Failed to open data.xml\n";

		readAndExtractFeatures(pathOfTest, test, testResponse);
		readAndExtractFeatures(pathOfTrain, train, trainResponse);

		testMat = Mat((int) test.size() / 2, 2, CV_32FC1, &test[0]);
		trainMat = Mat((int) train.size() / 2, 2, CV_32FC1, &train[0]);
		testResponses = Mat((int) testResponse.size(), 1, CV_32SC1, testResponse[0]);
		trainResponses = Mat((int) trainResponse.size(), 1, CV_32SC1, &trainResponse[0]);

		// To write to data.xml file.
		FileStorage fileWrite("data.xml", cv::FileStorage::WRITE);
		fileWrite << "testMat" << testMat;
		fileWrite << "trainMat" << trainMat;
		fileWrite << "testResponses" << testResponses;
		fileWrite << "trainResponses" << trainResponses;
		cout << "\n data.xml created";
	} else {
		cout << "\n Opened data.xml\n";
		fileRead["testMat"] >> testMat;
		fileRead["trainMat"] >> trainMat;
		fileRead["testResponses"] >> testResponses;
		fileRead["trainResponses"] >> trainResponses;
	}

	Ptr<TrainData> trainData = TrainData::create(trainMat, ROW_SAMPLE, trainResponses);
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setNu(0.05);
	svm->setKernel(SVM::CHI2);
	svm->setDegree(1.0);
	svm->setGamma(2.0);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(trainData);
	cout << endl << "Test";
	if(testResponse.size() > 0) {
		// Test the ML Model
		Mat testPredict;
		svm->predict(testMat, testPredict);
		cout << testPredict.size() << endl;
		cout << testResponses.size();
		// Error calculation
		Mat errorMat = testPredict != testResponses;
		cout << endl << format(" Error: %.5lf%%", 100.0f * countNonZero(errorMat) / testResponse.size());
	}
	return svm;
}

int main(int argc, char **argv) {
	Ptr<SVM> svm = trainAndTest();

	VideoCapture capture;

	if(!capture.open(0)) {
		return EXIT_FAILURE;
	}

	Mat frame;
	while(true) {
		if(!capture.read(frame)) break;
		imshow("Window", frame);
		Mat specific(frame, Rect(100, 100, 200, 200));
		specific = preprocess(specific);
		imshow("Window 3", specific);

		vector<vector<float>> features = extractFeatures(specific);

		for(int i = 0; i < features.size(); i++) {
			Mat trainingDataMat(1, 2, CV_32FC1, &features[i][0]);
			float result = svm->predict(trainingDataMat);
			cout << result;
			/*if(result == 0) {
				cout << 0 << endl;
			} else if(result == 1) {
				cout << 1 << endl;
			} else if(result == 2) {
				cout << 2 << endl;
			} else if(result == 3) {
				cout << 3 << endl;
			} else if(result == 4) {
				cout << 4 << endl;
			} else if(result == 5) {
				cout << 5 << endl;
			}*/
		}
		waitKey(1);
	}
	return EXIT_SUCCESS;
}