#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <ctime>
#include <unordered_map>
#define DIR "train_photos/"
#define TOTAL 2527877 //`ls DIR`

using namespace cv;
using namespace std;

typedef struct YelpImage{
	string filename;
	Mat image;
	YelpImage(string filename){
		this->filename = filename;
	}
	Mat descriptors;
	vector<KeyPoint> keypoints;
}YelpImage;

typedef struct YelpDataset{
	unordered_map<string, vector<YelpImage> >restaurantId_img_mapping;
}YelpDataset;

Mat& ScanImageAndReduceC(Mat &I, const uchar *table);
Mat& ScanImageandReduceIterator(Mat &I, const uchar *table);
Mat& ScanImageAndReduceRandomAccess(Mat &I, const uchar *table);

YelpDataset retrieveData(string filename){
	ifstream input;
	YelpDataset ds;
	string line;
	input.open(filename);
	if(!input.is_open()){
		cout << "Unable to read file!" << endl;
		return -1;
	}
	while(getline(input, line)){
		Mat I;
		YelpImage img = new YelpImage(line);
		line = DIR + line;
		I = imread(line, 0);
		if(!I.data){
			cout << "No image data" << endl;
			continue;
		}
		img.image = I;
		
		/* 
		

		Get train labels and IDs 



		*/
	}
	input.close();
	return (ds);
}

