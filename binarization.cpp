#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int PercentileFilter(Mat r, int index);
int main()
{
    Mat test = imread("/home/syeda_urooj_fatima/Downloads/Images/po.png",1);
    Mat gray;
    cvtColor(test,gray,CV_BGR2GRAY);
    Mat out (test.rows, test.cols, CV_8U);

    int win_size=3;
    int pad=win_size/2;

    Mat en_test;
    copyMakeBorder(gray,en_test,pad,pad,pad,pad,BORDER_REPLICATE);

    int p =80;
    int n=win_size*win_size;
    int index=round(n*p/100);

    for (int i=0; i<en_test.rows-(2*pad); i++)
    {
        for(int j=0; j<en_test.cols-(2*pad); j++)
        {
            Mat r(en_test,Rect(j,i,win_size,win_size));
            out.at<int>(i,j) = PercentileFilter(r, index);
        }
    }

    namedWindow("ORIGINAL");
    imshow("ORIGINAL",test);
    namedWindow("PERCENTILED");
    imshow("PERCENTILED",out);
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
