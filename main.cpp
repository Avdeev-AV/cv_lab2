#include <iostream>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

int main() {
	cv::Mat source_image;

	std::cout << std::endl << "Enter full name of the file: " << std::endl;
	std::string imagename = "";
	std::cin >> imagename;

	try {
		source_image = cv::imread(cv::samples::findFile(imagename), cv::IMREAD_COLOR);
		if (source_image.empty()) {
			std::cout << "Image cannot be loaded!" << std::endl;
			exit(0);
		}
	}
	catch (cv::Exception) {
		std::cout << "Image cannot be loaded!" << std::endl;
		exit(0);
	}
		//------------------------------Task 1------------------------------------------
		cv::Mat BGRchannels[3];
		cv::split(source_image, BGRchannels);
		std::cout << source_image.rows << '\t' << source_image.cols << std::endl;

		cv::Mat kernels[5][3];
		for (int i = 0; i < 5; i++)
			for(int j = 0; j < 3; j++) {
				kernels[i][j] = cv::Mat(3, 3, CV_32SC1, 0.f);
				cv::randu(kernels[i][j], -27, 27);
				std::cout << kernels[i][j] << std::endl;
			}

		cv::Mat conv_image_ch[3], conv_image_res[5];
		for (int i = 0; i < 5; i++)	{
			cv::filter2D(BGRchannels[0], conv_image_ch[0], -1, kernels[i][0]);
			cv::filter2D(BGRchannels[1], conv_image_ch[1], -1, kernels[i][1]);
			cv::filter2D(BGRchannels[2], conv_image_ch[2], -1, kernels[i][2]);
			conv_image_res[i] = conv_image_ch[0] + conv_image_ch[1] + conv_image_ch[2];
		}

		std::cout << conv_image_res[0].rows << '\t' << conv_image_res[0].cols << std::endl;

		//------------------------------Task 2------------------------------------------
		cv::Mat norm_image_res[5];
		for (int i = 0; i < 5; i++)	{
			cv::normalize(conv_image_res[i], norm_image_res[i], 0, 1, 32, CV_32FC3);
		}

		//------------------------------Task 3------------------------------------------
		cv::Mat relu_image_res[5];
		for (int i = 0; i < 5; i++)
			relu_image_res[i] = cv::max(norm_image_res[i], 0);

		//------------------------------Task 4------------------------------------------
		cv::Mat maxPooling_res[5];

		for (int a = 0; a < 5; a++)	{
			maxPooling_res[a] = cv::Mat::zeros(relu_image_res[a].size(), CV_32SC1);
			maxPooling_res[a].rows = relu_image_res[a].rows / 2;
			maxPooling_res[a].cols = relu_image_res[a].cols / 2;

			for (int i = 0; i < maxPooling_res[a].rows; i++)
				for (int j = 0; j < maxPooling_res[a].cols; j++)
					for (int k = 0; k < 2; k++)
						for (int l = 0; l < 2; l++)
							maxPooling_res[a].at<float>(i, j) = cv::max(relu_image_res[a].at<float>(i * 2 + k, j * 2 + l), maxPooling_res[a].at<float>(i, j));
		}

		std::cout << maxPooling_res[0].rows << '\t' << maxPooling_res[0].cols << std::endl;

		//------------------------------Task 5------------------------------------------
		cv::Mat softmax_channels[5];

		for (int a = 0; a < 5; a++)
			for (int i = 0; i < maxPooling_res[a].rows; i++)
				for (int j = 0; j < maxPooling_res[a].cols; j++) {
					softmax_channels[a] = cv::Mat::zeros(maxPooling_res[a].size(), CV_32SC1);
					softmax_channels[a].at<float>(i, j) = std::exp(maxPooling_res[a].at<float>(i, j)) / cv::sum(std::exp(maxPooling_res[a].at<float>(i, j)))[0];
				}
		std::cout << softmax_channels[0].rows << '\t' << softmax_channels[0].cols << std::endl;
					
		cv::waitKey(0);
		cv::destroyAllWindows();
}