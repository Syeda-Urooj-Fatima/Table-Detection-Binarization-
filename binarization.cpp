#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

uchar PercentileFilter(Mat r, int index);
int main()
{
    Mat test = imread("/home/syeda_urooj_fatima/Downloads/Images/po.png",1); //input image
    Mat gray; //grayscaled input image
    cvtColor(test,gray,CV_BGR2GRAY);
    Mat out (gray.rows, gray.cols, CV_8U); //binary output image
    int win_size = 5;
    int pad = win_size/2;

    Mat en_test; //extrapolated image with paddings around borders
    copyMakeBorder(gray,en_test,pad,pad,pad,pad,BORDER_REPLICATE);

    int p =80; //percentile rank
    int n = win_size*win_size; //number of pixels in windows
    int index = round(n*p/100); //index of required percentile rank pixel

    float t=0.3; //threshold factor;


    for (int i=0; i<en_test.rows-(2*pad); i++) //Applying percentile filter
    {

        for(int j=0; j<en_test.cols-(2*pad); j++)
        {
            Mat r(en_test,Rect(j,i,win_size,win_size));
            out.at<uchar>(i,j) = PercentileFilter(r, index);
        }
    }

    namedWindow("ORIGINAL");
    imshow("ORIGINAL",gray);
    namedWindow("PERCENTILED");
    imshow("PERCENTILED",out);

    for(int i=0; i<gray.rows; i++) //binarizing the gray scale image
    {
        for (int j=0; j<gray.cols; j++)
        {
            if(gray.at<uchar>(i,j) < t*out.at<uchar>(i,j))
                out.at<uchar>(i,j)=255;
            else
                out.at<uchar>(i,j)=0;
        }
    }

    namedWindow("BINARIZED");
    imshow("BINARIZED",out);
    waitKey(0);
    return 0;
}

uchar PercentileFilter(Mat r,int index) //function applying percentile filter on matrix r using "window search" method
{
    vector<uchar> A; //vector for pixels under consideration
    vector<uchar> A_s; //vector for pixels under the percentile rank
    MatIterator_<uchar> it, end;

    for(it = r.begin<uchar>(), end = r.end<uchar>(); it != end; ++it)
    {
        A.push_back(*it);
    }

    int M = A.size();
    int M_s=0; //size of A_s
    vector<uchar> prev; //vector to keep track of two or more pixels for the same percentile rank
    int M_p=index+1;
    while(M>1)
    {
        uchar pivot = A[rand()%M];

        for(int i=0;i<A.size();i++)
        {
            if(A[i]<=pivot)
            {
                A_s.push_back(A[i]);
                A.erase(A.begin()+i);
                i--;
                M_s++;
            }
        }

        M = A.size();

        if(M_s>=M_p)
        {
            if(prev==A_s)
            {
                return A_s[0];
            }
            A=A_s;
            prev=A;
            M=M_s;
        }
        else
        {
            M_p=M_p-M_s;
        }

        A_s.clear();
    }

    return A[0];
}



//Note: Window search method is mentioned in "Fast Percentile Filtering" paper