#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int PercentileFilter(Mat r, int index);
int main()
{
    Mat test = imread("/home/syeda_urooj_fatima/Downloads/Images/po.png",1); //input image
    Mat gray; //grayscaled input image
    cvtColor(test,gray,CV_BGR2GRAY);
    Mat percentiled (gray.rows, gray.cols, CV_32S); //image for the results of percentile filters
    Mat out (gray.rows, gray.cols, CV_32S); //binary output image
    int win_size=2;
    int pad=win_size/2;

    Mat en_test; //extrapolated image with paddings around borders
    copyMakeBorder(gray,en_test,pad,pad,pad,pad,BORDER_REPLICATE);

    int p =80; //percentile rank
    int n = win_size*win_size; //number of pixels in windows
    int index = round(n*p/100); //index of required percentile rank pixel

    for (int i=0; i<en_test.rows-(2*pad); i++) //Applying percentile filter
    {
        for(int j=0; j<en_test.cols-(2*pad); j++)
        {
            Mat r(en_test,Rect(j,i,win_size,win_size));
            percentiled.at<int>(i,j) = PercentileFilter(r, index);
        }
    }

    namedWindow("ORIGINAL");
    imshow("ORIGINAL",gray);
    namedWindow("PERCENTILED");
    imshow("PERCENTILED",percentiled);


    waitKey(0);
    return 0;
}

int PercentileFilter(Mat r,int index)
{
    vector<int> array;
    MatIterator_<int> it, end;
    for(it = r.begin<int>(), end = r.end<int>(); it != end; ++it)
    {
        array.push_back(*it);
    }
    sort(array.begin(),array.end(),less<int>());

    return array[index];
}
