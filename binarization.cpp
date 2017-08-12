#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat PercentileFilter(Mat img, int window, int p);

int main()
{
    Mat test = imread("/home/syeda_urooj_fatima/Downloads/Images/page.jpg",1); //input image
    Mat gray; //grayscaled input image
    cvtColor(test,gray,CV_BGR2GRAY);
    gray.convertTo(gray,CV_32FC1);
    Mat out (gray.rows, gray.cols, CV_8U); //binary output image
    int win_size = 40;
    int p =90; //percentile rank


    float t=0.6; //threshold factor;
    out = PercentileFilter(gray,win_size,p);
    gray.convertTo(gray,CV_8U);
    //cout<<gray;
    //cout<<out;
    //out.convertTo(out,CV_8U);
    namedWindow("ORIGINAL");
    imshow("ORIGINAL",gray);
    namedWindow("PERCENTILED");
    imshow("PERCENTILED",out);
    imwrite("/home/syeda_urooj_fatima/Downloads/Images/MAT1.jpg",out);

    for(int i=0; i<gray.rows; i++) //binarizing the gray scale image
    {
        for (int j=0; j<gray.cols; j++)
        {
            if(gray.at<uchar>(i,j) < t*out.at<uchar>(i,j))
                out.at<uchar>(i,j)=0;
            else
                out.at<uchar>(i,j)=255;
        }
    }

    namedWindow("BINARIZED");
    imshow("BINARIZED",out);
    imwrite("/home/syeda_urooj_fatima/Downloads/Images/MAT2.jpg",out);
    waitKey(0);
    return 0;
}

Mat PercentileFilter(Mat img,int win_size, int p) //function applying percentile filter on matrix r using "window search" method
{
    bool nextline=false;
    int pad = win_size/2;
    int n = win_size*win_size; //number of pixels in windows
    int index = round(n*p/100); //index of required percentile rank pixel
    Mat en_test; //extrapolated image with paddings around borders
    copyMakeBorder(img,en_test,pad,pad,pad,pad,BORDER_REPLICATE);
    Mat out (img.rows, img.cols, img.type());

    Mat hist;
    int histsize=256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    int sum=0;
    Mat first;

    //cout<<en_test;

    for (int i=0; i<en_test.rows-(2*pad); i++) //Applying percentile filter
    {
        for(int j=0; j<en_test.cols-(2*pad); j++)
        {
            Mat r(en_test,Rect(j,i,win_size,win_size));
            if(i==0 && j==0)
            {
                calcHist(&r,1,0,Mat(),hist,1,&histsize,&histRange,true,true);
                for(int k=0; k<histsize; k++)
                {
                    sum+=hist.at<float>(k);
                    if(sum>=index+1)
                    {
                        out.at<float>(i,j)=k;
                        break;
                    }
                }
                hist.copyTo(first);
            }

            else if (!nextline)
            {
                for(int y=i; y<i+win_size; y++)
                {
                    hist.at<float>(en_test.at<float>(y,j-1))--;
                }

                for(int z=j+win_size-1, y=i; y<i+win_size; y++)
                {
                    hist.at<float>(en_test.at<float>(y,z))++;
                }

                for(int k=0; k<histsize; k++)
                {
                    sum+=hist.at<float>(k);
                    if(sum>=index+1)
                    {
                        out.at<float>(i,j)=k;
                        break;
                    }
                }
            }

            else
            {
                first.copyTo(hist);
                for(int z=j; z<j+win_size; z++)
                {
                    hist.at<float>(en_test.at<float>(i-1,z))--;
                }

                for(int y=i+win_size-1, z=j; z<j+win_size; z++)
                {
                    hist.at<float>(en_test.at<float>(y,z))++;
                }

                for(int k=0; k<histsize; k++)
                {
                    sum+=hist.at<float>(k);
                    if(sum>=index+1)
                    {
                        out.at<float>(i,j)=k;
                        break;
                    }
                }
                hist.copyTo(first);
                nextline=false;
            }

            sum=0;
        }
        nextline=true;
    }

    //cout<<out;
    out.convertTo(out,CV_8U);
    return out;
    //imshow("ORIGINAL",gray);
    //cout<<hist.type();
}


/*uchar PercentileFilter(Mat r,int index) //function applying percentile filter on matrix r using "window search" method
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
}*/
