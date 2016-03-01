#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <ctime>
#define DIR "train_photos/"
using namespace cv;
using namespace std;

typedef struct{
	string name;
	Mat img;
}YelpImage;

typedef struct{
	vector<YelpImage> y;
}YelpDataset;

Mat& ScanImageAndReduceC(Mat &I, const uchar *table);
Mat& ScanImageandReduceIterator(Mat &I, const uchar *table);
Mat& ScanImageAndReduceRandomAccess(Mat &I, const uchar *table);

int main(int argc, char **argv){
	time_t current_time, end_time;
	current_time = time(NULL);
	ifstream input;
	string line;
	YelpDataset ds;
	Mat I;
	input.open(argv[1]);
	if(!input.is_open()){
		cout << "Unable to read file!" << endl;
		exit(1);
	}
	while(getline(input, line)){
		YelpImage y_img;
		y_img.name = line;
		line = DIR + line;
		I = imread(line, 0);
		if(!I.data){
			cout << "No image data" << endl;
			return -1;
		}
		y_img.img = I;
		//namedWindow(line, CV_WINDOW_AUTOSIZE);
		//imshow(line, I);
		//waitKey(0);
		ds.y.push_back(y_img);
	}
	end_time = time(NULL);
	cout << "Total Run Time: " << end_time - current_time << endl;
	return 0;
}
