#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <dirent.h>

using namespace std;
using namespace cv;
using boost::property_tree::ptree;

struct box
{
    int x;
    int y;
    int xmax;
    int ymax;
};

void RGB2GrayBinary(Mat& im, Mat& out, int& var_threshold);
vector<box> read(istream & is);
void findConnectedComponents(const Mat &binary, vector <vector<Point2i>> &blobs, vector <Rect> &blobRects);
float intersect_over_union(Rect box, struct box box_org);

int main()
{
    string im_dir_path="/home/syeda_urooj_fatima/Downloads/highlight_images";
    string label_dir_path="/home/syeda_urooj_fatima/Downloads/highlight_labels";
    string output_path="/home/syeda_urooj_fatima/Downloads/Images/OUTPUT";
    int i=0; //for output names
    int right=0;
    int wrong=0;
    int total=0;
    int var_threshold=2300;
    DIR *pDIR;
    struct dirent *entry;
    if( pDIR=opendir(im_dir_path.c_str()) )
    {
        while(entry = readdir(pDIR))
        {
            if( strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0 )
            {
                i++;
                vector<vector<Point2i>> blobs;
                vector<Rect> blobRects;
                float maxblob=0;
                string name (entry->d_name);
                string im_file_path=im_dir_path+"/"+name;
                name = name.substr(0,name.find("."));
                string label_file_path = label_dir_path+"/"+name+".xml";

                /*cout<<im_file_path<<"\n";
                cout<<label_file_path<<"\n\n";*/

                Mat test = imread(im_file_path, 1); //input image
                Mat out(test.rows,test.cols,CV_8U);

                RGB2GrayBinary(test,out, var_threshold);
                findConnectedComponents(out, blobs, blobRects);

                //imshow("Output 2300",out);
                //imwrite("/home/syeda_urooj_fatima/Downloads/Images/OUTPUT.jpg", out);

                for (int i=0; i<blobRects.size(); i++)
                {
                    float area = blobRects[i].width*blobRects[i].height;
                    if (area>maxblob)
                    {
                        maxblob=area;
                    }
                }

                for(int i=0; i<blobRects.size(); i++)
                {
                    float area=blobRects[i].width*blobRects[i].height;
                    if(area<0.07*maxblob)
                    {
                        blobRects.erase(blobRects.begin()+i);
                        i--;
                    }
                }

                vector<box> boxes;
                filebuf fb;
                if (fb.open (label_file_path.c_str(),ios::in))
                {
                    istream is(&fb);
                    if(is)
                        boxes = read(is);
                }

                bool incorrect=false;
                for(int i=0; i<blobRects.size(); i++)
                {
                    for(int j=0; j<boxes.size(); j++)
                    {
                        float iou = intersect_over_union(blobRects[i],boxes[j]);
                        if(iou>0.8) {
                            rectangle(test, blobRects[i], Scalar(0, 255, 0), 4);
                            right++;
                            total++;
                            incorrect=false;
                            break;
                        }

                        else {
                            incorrect = true;
                        }
                    }
                    if(incorrect) {
                        wrong++;
                        total++;
                        rectangle(test, blobRects[i], Scalar(0, 0, 255), 4);
                    }
                }

                //imshow("IMAGE With Boxes", test);
                imwrite(output_path+to_string(i)+".jpg", test);

            }
        }
        closedir(pDIR);
    }

    cout<<"TOTAL: "<<total<<"\n";
    cout<<"RIGHT: "<<right<<"\n";
    cout<<"WRONG: "<<wrong<<"\n";
    cout<<"ACCURACY: "<<(float(right)/total)*100;

    return 0;
}

void RGB2GrayBinary(Mat& test, Mat& out, int& var_threshold)
{
    for (int i = 0; i < test.rows; i++) {

        for (int j = 0; j < test.cols; j++) {
            float b = saturate_cast<float>(test.at<Vec3b>(i, j).val[0]);
            float g = saturate_cast<float>(test.at<Vec3b>(i, j).val[1]);
            float r = saturate_cast<float>(test.at<Vec3b>(i, j).val[2]);

            float m = (b + g + r) / 3;
            float variance = (r - m) * (r - m) + (b - m) * (b - m) + (g - m) * (g - m);
            if (variance < var_threshold)
                out.at<uchar>(i, j) = 255;
            else
                out.at<uchar>(i, j) = 0;
        }
    }
}

vector<box> read(istream & is)
{
    ptree pt;
    read_xml(is, pt);

    vector<box> boxes;
    BOOST_FOREACH(const auto & obj, pt.get_child("annotation"))
                {
                    if(obj.first == "object")
                    {
                        BOOST_FOREACH(const auto & attrib, obj.second)
                                    {
                                        if(attrib.first=="bndbox")
                                        {
                                            box b;
                                            b.x = attrib.second.get<int>("xmin");
                                            b.y = attrib.second.get<int>("ymin");
                                            b.xmax = attrib.second.get<int>("xmax");
                                            b.ymax = attrib.second.get<int>("ymax");
                                            boxes.push_back(b);
                                        }
                                    }
                    }
                }

    return boxes;
}

void findConnectedComponents(const Mat &binary, vector <vector<Point2i>> &blobs, vector <Rect> &blobRects)
{
    Mat image2=binary.clone();
    image2=image2/255;
    image2=1-image2;

    Mat label_image;
    image2.convertTo(label_image, CV_32FC1); // weird it doesn't support CV_32S!

    int label_count = 2;

    for (int y = 0; y < image2.rows; y++) {
        for (int x = 0; x < image2.cols; x++) {
            if ((int) label_image.at<float>(y, x) != 1) {
                continue;
            }

            Rect rect;
            floodFill(label_image, Point(x, y), Scalar(label_count), &rect, Scalar(0), Scalar(0), 4);

            vector <Point2i> blob;

            for (int i = rect.y; i < (rect.y + rect.height); i++) {
                for (int j = rect.x; j < (rect.x + rect.width); j++) {
                    if ((int) label_image.at<float>(i, j) != label_count) {
                        continue;
                    }

                    blob.push_back(Point2i(j, i));
                }
            }
            blobs.push_back(blob);

            label_count++;
        }
    }

    for (size_t i = 0; i < blobs.size(); i++) {
        int top = 10000, bottom = -1, left = 10000, right = -1;
        for (size_t j = 0; j < blobs[i].size(); j++) {
            int x = blobs[i][j].x;
            int y = blobs[i][j].y;
            if (x < left)
                left = x;
            if (x > right)
                right = x;
            if (y < top)
                top = y;
            if (y > bottom)
                bottom = y;
        }
        int w = right - left;
        int h = bottom - top;
        blobRects.push_back(Rect(left, top, w, h));
    }
}

float intersect_over_union(Rect box, struct box box_org)
{
     /*float xA = max(box.x, box_org.x);
     float yA = max(box.y, box_org.y);
     float xB = min(box.x+box.width-1, box_org.xmax);
     float yB = min(box.y+box.height-1, box_org.ymax);

     float interArea = (xB - xA + 1) * (yB - yA + 1);*/


    Rect box_org2(box_org.x, box_org.y, box_org.xmax-box_org.x+1, box_org.ymax-box_org.y+1);
    int interArea = (box & box_org2).area();

    float boxAArea = (box.width) * (box.height);
    float boxBArea = (box_org.xmax-box_org.x+1) * (box_org.ymax-box_org.y+1);

    float iou = interArea / (boxAArea + boxBArea - interArea);

    return iou;
}
