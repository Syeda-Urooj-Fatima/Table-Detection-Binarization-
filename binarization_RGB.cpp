#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat PercentileFilter(Mat img, int window, int p);

int main()
{
    Mat test = imread("/home/syeda_urooj_fatima/Downloads/Images/test1.jpg",1); //input image
    vector<Mat> input(3); //grayscaled input image
    vector<Mat> output(3);
    Mat out;
    split(test,input); //split the image into channels

   /* imshow("INPUT",test);
    cvtColor(test, test, CV_BGR2YCrCb); //change the color image from BGR to YCrCb format

    split(test,input); //split the image into channels

    equalizeHist(input[0], input[0]); //equalize histogram on the 1st channel (Y)

    merge(input,out); //merge 3 channels including the modified 1st channel into one image

    cvtColor(out, out, CV_YCrCb2BGR);*/

    int win_size = 7;
    int p =80; //percentile rank
    int allowance=30;
    /*int allowance_g=25;
    int allowance_r=25;*/

    input[0].convertTo(input[0],CV_32FC1);
    input[1].convertTo(input[1],CV_32FC1);
    input[2].convertTo(input[2],CV_32FC1);
    output[0] = PercentileFilter(input[0],win_size,p);
    output[1] = PercentileFilter(input[1],win_size,p);
    output[2] = PercentileFilter(input[2],win_size,p);

    output[0].convertTo(output[0],CV_8U);
    output[1].convertTo(output[1],CV_8U);
    output[2].convertTo(output[2],CV_8U);
    //cout<<gray;
    //cout<<out;
    //out.convertTo(out,CV_8U);
    merge(output,out);
    namedWindow("ORIGINAL");
    imshow("ORIGINAL",test);
    namedWindow("PERCENTILED");
    imshow("PERCENTILED",out);
    //imwrite("/home/syeda_urooj_fatima/Downloads/Images/MAT3.jpg",out);

    for(int i=0; i<out.rows; i++) //binarizing the gray scale image
    {
        for (int j=0; j<out.cols; j++)
        {
            float b=saturate_cast<float>(out.at<Vec3b>(i,j).val[0]);
            float g=saturate_cast<float>(out.at<Vec3b>(i,j).val[1]);
            float r=saturate_cast<float>(out.at<Vec3b>(i,j).val[2]);
            float b1=saturate_cast<float>(test.at<Vec3b>(i,j).val[0]);
            float g1=saturate_cast<float>(test.at<Vec3b>(i,j).val[1]);
            float r1=saturate_cast<float>(test.at<Vec3b>(i,j).val[2]);
            //float avg=(b+g+r)/3;
            int difb = abs(b1-b);
            int difg = abs(g1-g);
            int difr = abs(r1-r);
            if(difb<=allowance && difg<=allowance &&difr<=allowance)
                test.at<Vec3b>(i,j)={255,255,255};
            /*else
                out.at<Vec3b>(i,j)={255,255,255};*/


        }
    }

    Mat element = getStructuringElement(MORPH_RECT,Size(2,2));
    //erode(test,test,element,Point(-1,-1));
    //dilate(test,test,element,Point(-1,-1));
    namedWindow("BINARIZED");
    imshow("BINARIZED",test);
    //imwrite("/home/syeda_urooj_fatima/Downloads/Images/MAT4.jpg",test);
    waitKey(0);
    return 0;
}

Mat PercentileFilter(Mat img,int win_size, int p) //function applying percentile filter on matrix r using "histogram search" method
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
