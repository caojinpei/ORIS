#include <iostream>
#include<opencv2/opencv.hpp>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;

int main()
{
	Mat srcimage1 = imread("1.jpg");
	Mat srcimage2 = imread("2.jpg");

	if (!srcimage1.data || !srcimage2.data)
	{
		cout << "ͼƬ��������!" << endl;
		return -1;
	}

	//ʹ��ORB��������
	Ptr<ORB> detector = ORB::create(60000);
	vector<KeyPoint> keypoint1, keypoint2;

	//��Դͼ�����ҳ������㲢�����vector��
	detector->detect(srcimage1, keypoint1);
	detector->detect(srcimage2, keypoint2);


	//������������
	Ptr<ORB> extractor = ORB::create();
	Mat descriptors1, descriptors2;
	extractor->compute(srcimage1, keypoint1, descriptors1);
	extractor->compute(srcimage2, keypoint2, descriptors2);
	//����ƥ��
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
	cout << "��ƥ���" << knn_matches.size() << endl;

	//��������ƥ��������ֵ�����ʵ�
	const float ratio_thresh = 0.8f;
	std::vector<DMatch> good_matches;
	good_matches.clear();
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance / knn_matches[i][1].distance < ratio_thresh)
		{

			good_matches.push_back(knn_matches[i][0]);

		}
	}
	//����ƥ������
	cout << "����ƥ����" << good_matches.size() << endl;

	//��ȡ����ƥ����У�ͼ��1��ͼ��2��������
	vector<KeyPoint> R_keypoint01, R_keypoint02;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		R_keypoint01.push_back(keypoint1[good_matches[i].queryIdx]);
		R_keypoint02.push_back(keypoint2[good_matches[i].trainIdx]);
	}

	//��ȡ����ƥ����У�ͼ��1��ͼ��2������������
	vector<Point2f>p01, p02;
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		p01.push_back(R_keypoint01[i].pt);
		p02.push_back(R_keypoint02[i].pt);
	}

	//�����������ʹ��RANSAC�㷨
	Mat  H12;
	H12 = findHomography(Mat(p01), Mat(p02), CV_RANSAC);
	//���¶���ؼ���RR_KP��RR_matches���洢�µĹؼ���ͻ�������ͨ��RansacStatus��ɾ����ƥ���	
	cout << "�任����Ϊ��\n" << H12 << endl << endl; //���ӳ�����

	Mat imageTransform1, imageTransform2;
	cv::warpPerspective(srcimage1, imageTransform1, H12, Size(srcimage2.cols*1.3, srcimage2.rows*1.8));
	//����Ӱ����׼�任ͼ
	namedWindow("��׼", 0);
	imshow("��׼", imageTransform1);
	imwrite("��׼.jpg", imageTransform1);

	//�����ͼ��׼���ͼ
	int dst_width = imageTransform1.cols;  //ȡ���ҵ�ĳ���Ϊƴ��ͼ�ĳ���
	int dst_height = imageTransform1.rows;
	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);
	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	srcimage2.copyTo(dst(Rect(0, 0, srcimage2.cols, srcimage2.rows)));
	namedWindow("b_dst", 0);
	imshow("b_dst", dst);
	//����Ӱ��dst.jpg���ͼ
	imwrite("dst.jpg", dst);

	waitKey();
	return 0;
}


