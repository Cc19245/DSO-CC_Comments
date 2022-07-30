/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/

#include <sstream>
#include <fstream>
#include <iostream>

#include <Eigen/Core>
#include <iterator>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "IOWrapper/ImageDisplay.h"
#include "IOWrapper/ImageRW.h"
#include "util/Undistort.h"

namespace dso
{
	/********************************
	 * @ function: 光度矫正构造函数
	 *   疑问：其中的光度标定文件到底是G还是G^-1？
     * //! 7.30：目前认为是G，因为它的值有255，肯定就是一个能量值传进来，然后把它投影成像素为0-255，否则下面代码中为什么会有255呢？
	 * @ param: 	file          	响应函数参数文件
	 *		 		noiseImage   	没用，最后传入的是空
	 *		 		vignetteImage 	辐射衰减图像
	 *				w_, h_			图像大小
	*******************************/
	PhotometricUndistorter::PhotometricUndistorter(
		std::string file,
		std::string noiseImage,
		std::string vignetteImage,
		int w_, int h_)
	{
		valid = false;
		vignetteMap = 0;
		vignetteMapInv = 0;
		w = w_;
		h = h_;
		output = new ImageAndExposure(w, h); //; 辐照和时间，注意这里是辐照，不是简单的像素灰度了
		if (file == "" || vignetteImage == "")
		{
			printf("NO PHOTOMETRIC Calibration!\n");
		}

		// Step 1 光度仿射变换函数G，注意这个是从辐照B到光度I的变换，而我们最终是要去仿射变换，所以要求逆
		// read G.
		std::ifstream f(file.c_str());
		printf("Reading Photometric Calibration from file %s\n", file.c_str());
		if (!f.good())
		{
			printf("PhotometricUndistorter: Could not open file!\n");
			return;
		}

		// 得到响应函数的逆变换
        //! 疑问：这里哪里有逆变换了？扯淡
		{
			std::string line;
			std::getline(f, line);
			std::istringstream l1i(line);
			// begin迭代器, end迭代器来初始化
			//; 这个写法牛皮啊，直接把所有数据读到vector中了
			std::vector<float> Gvec = std::vector<float>(std::istream_iterator<float>(l1i), std::istream_iterator<float>());

			GDepth = Gvec.size(); //; 正常应该是256个数，并且最小0，最大255（注意是255不是256）

			if (GDepth < 256)
			{
				printf("PhotometricUndistorter: invalid format!
					 got %d entries in first line, expected at least 256!\n", (int)Gvec.size());
				return;
			}

			//; 响应函数的逆变换，直接赋值（注意参数文件给的就是U=G^-1，也就是逆变换）
            //! 疑问：这里存疑，到底是正变换还是逆变换
			for (int i = 0; i < GDepth; i++)
				G[i] = Gvec[i];

            //; 标定的响应函数值要是单调的，这里就是需要是递增的，这个也很符合常理
			for (int i = 0; i < GDepth - 1; i++)
			{
				if (G[i + 1] <= G[i])
				{
					printf("PhotometricUndistorter: G invalid! it has to be strictly increasing, but it isnt!\n");
					return;
				}
			}
			// 对响应值进行标准化，因为标定的响应值可能不是恰好在0-255之间，可能稍微差一点，这里就把他强行归一化到0-255之间
			float min = G[0];
			float max = G[GDepth - 1];
			for (int i = 0; i < GDepth; i++)
				G[i] = 255.0 * (G[i] - min) / (max - min); // make it to 0..255 => 0..255.
		}

		// 如果没有标定值, 则初始化为0-255 线性分布值
		//; 注意这里默认是2，也就是有标定的gamma文件
		if (setting_photometricCalibration == 0)
		{
			for (int i = 0; i < GDepth; i++)
				G[i] = 255.0f * i / (float)(GDepth - 1);
		}

		// Step 2 读取镜头渐晕
		// 读取图像, 为什么要读一个16位, 一个8位的???
		printf("Reading Vignette Image from %s\n", vignetteImage.c_str());
		MinimalImage<unsigned short> *vm16 = IOWrap::readImageBW_16U(vignetteImage.c_str());
		MinimalImageB *vm8 = IOWrap::readImageBW_8U(vignetteImage.c_str());
		vignetteMap = new float[w * h];
		vignetteMapInv = new float[w * h];

		//; 上面读取两种方式，应该就是备份双保险，以适应不同编码格式的渐晕图片
		if (vm16 != 0)
		{
			if (vm16->w != w || vm16->h != h)
			{
				printf("PhotometricUndistorter: Invalid vignette image size! got %d x %d, expected %d x %d\n",
					   vm16->w, vm16->h, w, h);
				if (vm16 != 0)
					delete vm16;
				if (vm8 != 0)
					delete vm8;
				return;
			}
			// 使用最大值来归一化
            //; 对渐晕图像进行归一化，因为渐晕是0-1之间的值
			float maxV = 0;
			for (int i = 0; i < w * h; i++)
				if (vm16->at(i) > maxV)
					maxV = vm16->at(i);
			for (int i = 0; i < w * h; i++)
				vignetteMap[i] = vm16->at(i) / maxV;
		}
		else if (vm8 != 0)
		{
			if (vm8->w != w || vm8->h != h)
			{
				printf("PhotometricUndistorter: Invalid vignette image size! got %d x %d, expected %d x %d\n",
					   vm8->w, vm8->h, w, h);
				if (vm16 != 0)
					delete vm16;
				if (vm8 != 0)
					delete vm8;
				return;
			}

			float maxV = 0;
			for (int i = 0; i < w * h; i++)
				if (vm8->at(i) > maxV)
					maxV = vm8->at(i);

			for (int i = 0; i < w * h; i++)
				vignetteMap[i] = vm8->at(i) / maxV;
		}
		else
		{
			printf("PhotometricUndistorter: Invalid vignette image\n");
			if (vm16 != 0)
				delete vm16;
			if (vm8 != 0)
				delete vm8;
			return;
		}

		if (vm16 != 0)
			delete vm16;
		if (vm8 != 0)
			delete vm8;

		// 求逆
        //; 渐晕的逆
		for (int i = 0; i < w * h; i++)
			vignetteMapInv[i] = 1.0f / vignetteMap[i];

		printf("Successfully read photometric calibration!\n");
		valid = true;
	}

	PhotometricUndistorter::~PhotometricUndistorter()
	{
		if (vignetteMap != 0)
			delete[] vignetteMap;
		if (vignetteMapInv != 0)
			delete[] vignetteMapInv;
		delete output;
	}

	//@ 给图像加上响应函数
	void PhotometricUndistorter::unMapFloatImage(float *image)
	{
		int wh = w * h;
		for (int i = 0; i < wh; i++)
		{
			float BinvC;
			float color = image[i];

			if (color < 1e-3) // 小置零
				BinvC = 0.0f;
			else if (color > GDepth - 1.01f) // 大最大值
				BinvC = GDepth - 1.1;
			else // 中间对响应函数插值
			{
				int c = color;
				float a = color - c;
				BinvC = G[c] * (1 - a) + G[c + 1] * a;
			}

			float val = BinvC;
			if (val < 0)
				val = 0;
			image[i] = val;
		}
	}

	/**
	 * @brief  去除光度畸变：去掉非线性响应函数G、渐晕V，得到辐照B
	 * 
	 * @tparam T 
	 * @param[in] image_in        原始输入图像
	 * @param[in] exposure_time   图像曝光时间
	 * @param[in] factor          默认1
	 */
	template <typename T>
	void PhotometricUndistorter::processFrame(T *image_in, float exposure_time, float factor)
	{
		int wh = w * h;
		float *data = output->image; //; 这里直接输出到类成员变量的图像中
		assert(output->w == w && output->h == h);
		assert(data != 0);

		// 没有光度模型
		//; valid=false说明没有成功读取gamma响应/渐晕文件，也就无法进行去光度畸变
		if (!valid || exposure_time <= 0 || setting_photometricCalibration == 0) // disable full photometric calibration.
		{
			for (int i = 0; i < wh; i++)
			{
				data[i] = factor * image_in[i];
			}
			output->exposure_time = exposure_time;
			output->timestamp = 0;
		}
		//; 否则可以正常进行去光度畸变
		else
		{
			for (int i = 0; i < wh; i++)
			{
				//! 重点：这里可以看出来，G就是gamma响应的逆变换，输入像素值I，得到t*V*B
				data[i] = G[image_in[i]]; // 去掉响应函数
			}
			//; 如果设置中还要求去掉渐晕，这里就再除以渐晕（也就是乘以渐晕的逆）
			if (setting_photometricCalibration == 2) // 去掉衰减系数
			{
				for (int i = 0; i < wh; i++)
					data[i] *= vignetteMapInv[i];
			}
			output->exposure_time = exposure_time; // 设置曝光时间
			output->timestamp = 0;
		}

		//; 如果设置中不使用曝光时间，那么这里就把曝光时间统一赋值成1ms
        //; 这里色值是false, 所以不会执行
		if (!setting_useExposure)
			output->exposure_time = 1;
	}
	// 模板特殊化, 指定两个类型
	template void PhotometricUndistorter::processFrame<unsigned char>(unsigned char *image_in, float exposure_time, float factor);
	template void PhotometricUndistorter::processFrame<unsigned short>(unsigned short *image_in, float exposure_time, float factor);


	//! 光度畸变类 结束， 畸变基类开始
	//! --------------------------  分割线  -----------------------------------------
	//! --------------------------  分割线  -----------------------------------------
	//! --------------------------  分割线  -----------------------------------------
	//! --------------------------  分割线  -----------------------------------------
	//******************************** 矫正基类, 包括几何和光度 ************************************
	Undistort::~Undistort()
	{
		if (remapX != 0)
			delete[] remapX;
		if (remapY != 0)
			delete[] remapY;
	}

	/**
	 * @brief 从传入的内参文件、矫正文件中建立相机去畸变模型，最后返回一个去畸变的类指针
	 *          注意如果没有光度矫正文件，传入的就是一个空字符串
	 * @param[in] configFilename    相机内参畸变校正文件
	 * @param[in] gammaFilename     相机非线性响应文件，一行，把[0-256]映射成某些数值
	 * @param[in] vignetteFilename  渐晕图像
	 * @return Undistort*  相机矫正基类指针，注意内部会根据不同的派生类(不同相机模型)调用派生类的方法，这也就是多态
	 */
	Undistort* Undistort::getUndistorterForFile(std::string configFilename, std::string gammaFilename, std::string vignetteFilename)
	{
		printf("Reading Calibration from file %s", configFilename.c_str());
		std::ifstream f(configFilename.c_str());
		if (!f.good())
		{
			f.close();
			printf(" ... not found. Cannot operate without calibration, shutting down.\n");
			f.close();
			return 0;
		}
		printf(" ... found!\n");
		std::string l1;
		std::getline(f, l1);  //; 读取畸变文件中的第一行到字符串l1中
		f.close();
		float ic[10];

		// Step 1 校正相机几何畸变
		Undistort* u; // 矫正基类, 作为返回值, 其他的类型继承自它

		//* 下面三种具体模型, 是针对没有指明模型名字的, 只给了参数
		//; 为了向后兼容？什么意思？
		//; 解答：注意看sscanf的读取方式，有的前面有前缀，有的前面没有前缀，所以就是有没有前缀都可以读取
		// for backwards-compatibility: Use RadTan model for 8 parameters.
		if (std::sscanf(l1.c_str(), "%f %f %f %f %f %f %f %f",
						&ic[0], &ic[1], &ic[2], &ic[3],
						&ic[4], &ic[5], &ic[6], &ic[7]) == 8)
		{
			printf("found RadTan (OpenCV) camera model, building rectifier.\n");
			//; 新建相机模型类：这里面最重要的操作就是计算正常图像对应畸变图像的像素位置
			u = new UndistortRadTan(configFilename.c_str(), true);
			if (!u->isValid())
			{
				delete u;
				return 0;
			}
		}
		// for backwards-compatibility: Use Pinhole / FoV model for 5 parameter.
		else if (std::sscanf(l1.c_str(), "%f %f %f %f %f",
							 &ic[0], &ic[1], &ic[2], &ic[3], &ic[4]) == 5)
		{
			if (ic[4] == 0) // 没有FOV的畸变参数, 只有pinhole
			{
				printf("found PINHOLE camera model, building rectifier.\n");
				u = new UndistortPinhole(configFilename.c_str(), true);
				if (!u->isValid())
				{
					delete u;
					return 0;
				}
			}
			//; 对于sequence01序列，进这里，因为是鱼眼镜头
			//! 靠，怎么换了一个图片看起来不是鱼眼镜头(sequence42)的，畸变参数还是这个模型？
			else // pinhole + FOV , atan
			{
				printf("found ATAN camera model, building rectifier.\n");
				//; 在构造函数中，会计算 输出图像 和 有效归一化平面范围 之间的内参，然后计算
				//; 输出图像和输入图像之间的像素坐标对应关系，存储在remapX和Y中
				u = new UndistortFOV(configFilename.c_str(), true);
				if (!u->isValid())
				{
					delete u;
					return 0;
				}
			}
		}
		//; 以下是指明了相机模型的几种选择，也就是在畸变参数文件中最开始明确写了是什么畸变模型
		// clean model selection implementation.
		else if (std::sscanf(l1.c_str(), "KannalaBrandt %f %f %f %f %f %f %f %f",
							 &ic[0], &ic[1], &ic[2], &ic[3],
							 &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
		{
			u = new UndistortKB(configFilename.c_str(), false);
			if (!u->isValid())
			{
				delete u;
				return 0;
			}
		}
		else if (std::sscanf(l1.c_str(), "RadTan %f %f %f %f %f %f %f %f",
							 &ic[0], &ic[1], &ic[2], &ic[3],
							 &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
		{
			u = new UndistortRadTan(configFilename.c_str(), false);
			if (!u->isValid())
			{
				delete u;
				return 0;
			}
		}
		else if (std::sscanf(l1.c_str(), "EquiDistant %f %f %f %f %f %f %f %f",
							 &ic[0], &ic[1], &ic[2], &ic[3],
							 &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
		{
			u = new UndistortEquidistant(configFilename.c_str(), false);
			if (!u->isValid())
			{
				delete u;
				return 0;
			}
		}
		else if (std::sscanf(l1.c_str(), "FOV %f %f %f %f %f",
							 &ic[0], &ic[1], &ic[2], &ic[3],
							 &ic[4]) == 5)
		{
			u = new UndistortFOV(configFilename.c_str(), false);
			if (!u->isValid())
			{
				delete u;
				return 0;
			}
		}
		else if (std::sscanf(l1.c_str(), "Pinhole %f %f %f %f %f",
							 &ic[0], &ic[1], &ic[2], &ic[3],
							 &ic[4]) == 5)
		{
			u = new UndistortPinhole(configFilename.c_str(), false);
			if (!u->isValid())
			{
				delete u;
				return 0;
			}
		}
		else
		{
			printf("could not read calib file! exit.");
			exit(1);
		}

		// Step 2 相机光度标定参数：包括非线性仿射变换和图像渐晕
		// 读入相机的光度标定参数
        //; 注意这里的函数调用，是利用上面创建的几何畸变类，去生成光度畸变类。
        //;  就相当于以几何畸变类为主体，内部包含了光度畸变类
		u->loadPhotometricCalibration(gammaFilename, "", vignetteFilename);

		return u;
	}

	//* 得到光度矫正类
	void Undistort::loadPhotometricCalibration(std::string file, std::string noiseImage, std::string vignetteImage)
	{
		photometricUndist = new PhotometricUndistorter(file, noiseImage, vignetteImage, getOriginalSize()[0], getOriginalSize()[1]);
	}

	/**
	 * @brief  得到去除光度参数的图像, 并添加几何和光度噪声（添加噪声是作者做实验用的）
	 * 
	 * @tparam T 
	 * @param[in] image_raw  从数据集中读取出来的原始的输入图像
	 * @param[in] exposure   从数据集中读取出来的曝光时间
	 * @param[in] timestamp  时间戳
	 * @param[in] factor     什么参数？默认实参是1
	 * @return ImageAndExposure*  去掉光度之后的辐照和曝光时间组成的图像
	 */
	template <typename T>
	ImageAndExposure* Undistort::undistort(const MinimalImage<T> *image_raw, float exposure, double timestamp, float factor) const
	{
		if (image_raw->w != wOrg || image_raw->h != hOrg)
		{
			printf("Undistort::undistort: wrong image size (%d %d instead of %d %d) \n", image_raw->w, image_raw->h, w, h);
			exit(1);
		}

		// Step 1 使用光度畸变类去除光度参数的影响：去掉gamma响应、渐晕，结果保存到类成员变量ImageAndExposure指针中
		//; 注意：这个是对 输入图像 进行整个去光度畸变操作
        //;  该函数执行结束后：它的类成员变量中直接存储了t*B, 以及曝光时间t
		photometricUndist->processFrame<T>(image_raw->data, exposure, factor); // 去除光度参数影响
		ImageAndExposure *result = new ImageAndExposure(w, h, timestamp);
		photometricUndist->output->copyMetaTo(*result); // 只复制了曝光时间

		// Step 2 对于输出图像，根据去几何畸变的remapX/Y，找到对应位置的输入图像的辐照B并赋值，从而得到输出的辐照图像
		//; 一般情况下, 只要图像裁切方式不是none, passthrough就是false, 所以进入这个分支
		if (!passthrough)
		{
			//; 这里是指针赋值，所以操作out_data就相当于操作result->image
			float *out_data = result->image; // 复制的图像做输出
			//; 注意这个in_data实际是上面的去光度畸变之后的图像，也就是t*B
			float *in_data = photometricUndist->output->image; // 输入图像

			//[ ***step 2*** ] 如果定义了噪声值, 设置随机几何噪声大小, 并且添加到输出图像
			float *noiseMapX = 0;
			float *noiseMapY = 0;

			//; 默认这个是0，所以不满足
			if (benchmark_varNoise > 0)
			{
				int numnoise = (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
				noiseMapX = new float[numnoise];
				noiseMapY = new float[numnoise];
				memset(noiseMapX, 0, sizeof(float) * numnoise);
				memset(noiseMapY, 0, sizeof(float) * numnoise);

				for (int i = 0; i < numnoise; i++)
				{
					noiseMapX[i] = 2 * benchmark_varNoise * (rand() / (float)RAND_MAX - 0.5f);
					noiseMapY[i] = 2 * benchmark_varNoise * (rand() / (float)RAND_MAX - 0.5f);
				}
			}

			for (int idx = w * h - 1; idx >= 0; idx--)
			{
				// get interp. values
				//; 输出图像去畸变，得到对应输入图像中的 像素位置
				float xx = remapX[idx];
				float yy = remapY[idx];

				//; 默认配置=0，所以不执行
				if (benchmark_varNoise > 0)
				{
					//? 具体怎么算的?
					float deltax = getInterpolatedElement11BiCub(noiseMapX, 4 + (xx / (float)wOrg) * benchmark_noiseGridsize, 4 + (yy / (float)hOrg) * benchmark_noiseGridsize, benchmark_noiseGridsize + 8);
					float deltay = getInterpolatedElement11BiCub(noiseMapY, 4 + (xx / (float)wOrg) * benchmark_noiseGridsize, 4 + (yy / (float)hOrg) * benchmark_noiseGridsize, benchmark_noiseGridsize + 8);
					float x = idx % w + deltax;
					float y = idx / w + deltay;
					if (x < 0.01)
						x = 0.01;
					if (y < 0.01)
						y = 0.01;
					if (x > w - 1.01)
						x = w - 1.01;
					if (y > h - 1.01)
						y = h - 1.01;

					xx = getInterpolatedElement(remapX, x, y, w);
					yy = getInterpolatedElement(remapY, x, y, w);
				}

				// 插值得到带有几何噪声的输出图像
				// 正常去畸变没有问题的话，这里if应该不会满足
				if (xx < 0)
					out_data[idx] = 0;
				//; 正常进入这个分支
				else
				{
					// get integer and rational parts
					int xxi = xx;
					int yyi = yy;
					xx -= xxi;
					yy -= yyi;
					float xxyy = xx * yy;

					// get array base pointer
					//; 得到原图这个位置的整数位置（因为像素只能是整数）对应的在数组中的指针
					const float *src = in_data + xxi + yyi * wOrg;

					// interpolate (bilinear)
					//; 双线性插值得到去畸变后的图像的辐照值（注意这里已经是辐照了，因为上面已经去掉光度了）
					//; 注意这里操作out_data就是在操作result->image
					out_data[idx] = xxyy * src[1 + wOrg] + (yy - xxyy) * src[wOrg] + (xx - xxyy) * src[1] + (1 - xx - yy + xxyy) * src[0];
				}
			}
			// 默认不执行
			if (benchmark_varNoise > 0)
			{
				delete[] noiseMapX;
				delete[] noiseMapY;
			}
		}
		else
		{
			memcpy(result->image, photometricUndist->output->image, sizeof(float) * w * h);
		}

		//[ ***step 3*** ]	添加光度噪声
		// 这个也是作者做实验用的，所以这里进入之后直接就return了
		applyBlurNoise(result->image);

		return result;
	}
	template ImageAndExposure *Undistort::undistort<unsigned char>(const MinimalImage<unsigned char> *image_raw, float exposure, double timestamp, float factor) const;
	template ImageAndExposure *Undistort::undistort<unsigned short>(const MinimalImage<unsigned short> *image_raw, float exposure, double timestamp, float factor) const;

	//* 添加图像高斯噪声
	void Undistort::applyBlurNoise(float *img) const
	{
		if (benchmark_varBlurNoise == 0)
			return; // 不添加噪声

		int numnoise = (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
		float *noiseMapX = new float[numnoise];
		float *noiseMapY = new float[numnoise];
		float *blutTmp = new float[w * h];

		if (benchmark_varBlurNoise > 0)
		{
			for (int i = 0; i < numnoise; i++)
			{
				noiseMapX[i] = benchmark_varBlurNoise * (rand() / (float)RAND_MAX);
				noiseMapY[i] = benchmark_varBlurNoise * (rand() / (float)RAND_MAX);
			}
		}

		// 高斯分布
		float gaussMap[1000];
		for (int i = 0; i < 1000; i++)
			gaussMap[i] = expf((float)(-i * i / (100.0 * 100.0)));

		// 对 X-Y 添加高斯噪声
		// x-blur.
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++)
			{
				float xBlur = getInterpolatedElement11BiCub(noiseMapX,
															4 + (x / (float)w) * benchmark_noiseGridsize,
															4 + (y / (float)h) * benchmark_noiseGridsize,
															benchmark_noiseGridsize + 8);

				if (xBlur < 0.01)
					xBlur = 0.01;

				int kernelSize = 1 + (int)(1.0f + xBlur * 1.5);
				float sumW = 0;
				float sumCW = 0;
				for (int dx = 0; dx <= kernelSize; dx++)
				{
					int gmid = 100.0f * dx / xBlur + 0.5f;
					if (gmid > 900)
						gmid = 900;
					float gw = gaussMap[gmid];

					if (x + dx > 0 && x + dx < w)
					{
						sumW += gw;
						sumCW += gw * img[x + dx + y * this->w];
					}

					if (x - dx > 0 && x - dx < w && dx != 0)
					{
						sumW += gw;
						sumCW += gw * img[x - dx + y * this->w];
					}
				}

				blutTmp[x + y * this->w] = sumCW / sumW;
			}

		// y-blur.
		for (int x = 0; x < w; x++)
			for (int y = 0; y < h; y++)
			{
				float yBlur = getInterpolatedElement11BiCub(noiseMapY,
															4 + (x / (float)w) * benchmark_noiseGridsize,
															4 + (y / (float)h) * benchmark_noiseGridsize,
															benchmark_noiseGridsize + 8);

				if (yBlur < 0.01)
					yBlur = 0.01;

				int kernelSize = 1 + (int)(0.9f + yBlur * 2.5);
				float sumW = 0;
				float sumCW = 0;
				for (int dy = 0; dy <= kernelSize; dy++)
				{
					int gmid = 100.0f * dy / yBlur + 0.5f;
					if (gmid > 900)
						gmid = 900;
					float gw = gaussMap[gmid];

					if (y + dy > 0 && y + dy < h)
					{
						sumW += gw;
						sumCW += gw * blutTmp[x + (y + dy) * this->w];
					}

					if (y - dy > 0 && y - dy < h && dy != 0)
					{
						sumW += gw;
						sumCW += gw * blutTmp[x + (y - dy) * this->w];
					}
				}
				img[x + y * this->w] = sumCW / sumW;
			}

		delete[] noiseMapX;
		delete[] noiseMapY;
	}

	/**
	 * @brief 对于crop图像的方式，纠正图像，计算 输出图像 和 有效归一化平面范围 之间的投影参数
	 * 
	 */
	void Undistort::makeOptimalK_crop()
	{
		printf("finding CROP optimal new model!\n");
		K.setIdentity();

		// 1. stretch the center lines as far as possible, to get initial coarse quess.
		float *tgX = new float[100000];
		float *tgY = new float[100000];
		float minX = 0;
		float maxX = 0;
		float minY = 0;
		float maxY = 0;

		// Step 1 先粗略在归一化平面上寻找裁切范围
		// Step 1.1 沿着水平轴寻找水平方向的最大裁切范围
		// -5 ~ 5分成10万份，分辨率0.0001
		for (int x = 0; x < 100000; x++)
		{
			tgX[x] = (x - 50000.0f) / 10000.0f; // -5 ~ 5 ?
			tgY[x] = 0;
		}
		//; 注意：对于自己下载的dso的几个数据集(sequence 1 19 42)应该都是FOV模型，所以这里对应就是FOV子类的函数
		distortCoordinates(tgX, tgY, tgX, tgY, 100000); // 加畸变
		
        for (int x = 0; x < 100000; x++)
		{
			//; 从小到大遍历，只要图像落在原图像内，那么就更新minX和maxX
			if (tgX[x] > 0 && tgX[x] < wOrg - 1)
			{
				if (minX == 0)
					minX = (x - 50000.0f) / 10000.0f;
				maxX = (x - 50000.0f) / 10000.0f;
			}
		}

		// Step 1.2 沿着竖直轴寻找竖直方向的裁切范围
		for (int y = 0; y < 100000; y++)
		{
			tgY[y] = (y - 50000.0f) / 10000.0f;
			tgX[y] = 0; //; 注意这里又把x全部归0了
		}
		distortCoordinates(tgX, tgY, tgX, tgY, 100000);
		for (int y = 0; y < 100000; y++)
		{
			if (tgY[y] > 0 && tgY[y] < hOrg - 1)
			{
				if (minY == 0)
					minY = (y - 50000.0f) / 10000.0f;
				maxY = (y - 50000.0f) / 10000.0f;
			}
		}
		delete[] tgX;
		delete[] tgY;

		//! 问题：什么操作？
		//; 解答：这个应该是再稍微放宽一点这个边界位置，注意maxX和Y都是*1.01很好理解，为什么minX和Y也是*1.01，感觉
		//;   不应该是乘以一个<1的数吗？比如0.99。其实是因为minX和minY都是<0的，如果想要放大图像范围，应该让他们
		//;   负数变得更加负，所以就是扩大负数的绝对值，也就是应该乘以>1的数而不是
		minX *= 1.01;
		maxX *= 1.01;
		minY *= 1.01;
		maxY *= 1.01;
		printf("initial range: x: %.4f - %.4f; y: %.4f - %.4f!\n", minX, maxX, minY, maxY);

		// Step 2 再精细地裁切，把边界上有一些无效像素的全部裁切掉
		// 2. while there are invalid pixels at the border: shrink square at the side that
		// has invalid pixels, if several to choose from, shrink the wider dimension.
		// 然而边界有无效像素：在有 无效像素的一侧缩小正方形，如果有多个可供选择，则缩小较宽的尺寸。
		bool oobLeft = true, oobRight = true, oobTop = true, oobBottom = true;
		int iteration = 0;
		while (oobLeft || oobRight || oobTop || oobBottom)
		{
			oobLeft = oobRight = oobTop = oobBottom = false;

			// Step 2.1 对水平x方向范围裁切
			// X 赋值最大最小, Y 赋值是根据坐标从小到大映射在minY 到 minY 之间
			//! 这样保证每个Y坐标都分别对应最大x, 最小x, 以便求出边界
			for (int y = 0; y < h; y++)
			{
				remapX[y * 2] = minX;
				remapX[y * 2 + 1] = maxX;
				//; 这里根据输出图像的高度来划分，很巧妙，其实就是在算图像左右边界的位置，经过畸变后有没有超出输入图像范围的
				remapY[y * 2] = minY + (maxY - minY) * (float)y / ((float)h - 1.0f);
				remapY[y * 2 + 1] = minY + (maxY - minY) * (float)y / ((float)h - 1.0f);
			}
			distortCoordinates(remapX, remapY, remapX, remapY, 2 * h); // 加畸变变换到当前图像
			// 如果还有不在图像范围内的, 则继续缩减
			for (int y = 0; y < h; y++)
			{
				// 最小的值即左侧要收缩
				if (!(remapX[2 * y] > 0 && remapX[2 * y] < wOrg - 1))
					oobLeft = true;
				// 最大值, 即右侧要收缩
				if (!(remapX[2 * y + 1] > 0 && remapX[2 * y + 1] < wOrg - 1))
					oobRight = true;
			}

			// Step 2.2 对竖直y方向范围裁切
			//! 保证每个 X 坐标都和Y坐标的最大最小, 构成对应坐标
			for (int x = 0; x < w; x++)
			{
				remapY[x * 2] = minY;
				remapY[x * 2 + 1] = maxY;
				remapX[x * 2] = minX + (maxX - minX) * (float)x / ((float)w - 1.0f);
				remapX[x * 2 + 1] = minX + (maxX - minX) * (float)x / ((float)w - 1.0f);
			}
			distortCoordinates(remapX, remapY, remapX, remapY, 2 * w);
			// 如果还有不在图像范围内的, 则继续缩减
			for (int x = 0; x < w; x++)
			{
				if (!(remapY[2 * x] > 0 && remapY[2 * x] < hOrg - 1))
					oobTop = true;
				if (!(remapY[2 * x + 1] > 0 && remapY[2 * x + 1] < hOrg - 1))
					oobBottom = true;
			}

			//! 如果上下, 左右都超出去, 也只缩减最大的一侧
			if ((oobLeft || oobRight) && (oobTop || oobBottom))
			{
				if ((maxX - minX) > (maxY - minY))
					oobBottom = oobTop = false; // only shrink left/right
				else
					oobLeft = oobRight = false; // only shrink top/bottom
			}

			// 缩减
			if (oobLeft)
				minX *= 0.995;
			if (oobRight)
				maxX *= 0.995;
			if (oobTop)
				minY *= 0.995;
			if (oobBottom)
				maxY *= 0.995;

			iteration++;

			printf("iteration %05d: range: x: %.4f - %.4f; y: %.4f - %.4f!\n", iteration, minX, maxX, minY, maxY);
			if (iteration > 500) // 迭代次数太多
			{
				printf("FAILED TO COMPUTE GOOD CAMERA MATRIX - SOMETHING IS SERIOUSLY WRONG. ABORTING \n");
				exit(1);
			}
		}

		// Step 3 图像裁剪完成，计算裁剪之后的相机内参
		//; 1.从这里也可以看出来，其实min/max X/Y就是归一化平面上的坐标
		//; 2.这里为什么w h要-1？ 因为我们最终的图像序号是从0开始的，即0~w-1 一共才是w个像素
		//! 思考这里新计算的内参有什么用？
		//! 解答：如果你有一个归一化平面上的点，我可以用这个内参计算像素坐标（注意归一化平面上的点是没有畸变的）
		//!      但是如果你有一个原图中的像素点，没办法转成这个新内参的像素坐标，因为畸变是非线性的
		K(0, 0) = ((float)w - 1.0f) / (maxX - minX);
		K(1, 1) = ((float)h - 1.0f) / (maxY - minY);
		K(0, 2) = -minX * K(0, 0);
		K(1, 2) = -minY * K(1, 1);
	}

	void Undistort::makeOptimalK_full()
	{
		// todo
		assert(false);
	}

	/**
	 * @brief   从参数文件中读取相机内参和畸变模型，主要就是两个步骤：
     *     1. 计算 输出图像 和 有效归一化平面 范围之间的投影参数矩阵，存到类成员变量 K 中
     *     2. 建立输出图像和输入图像像素对应关系，存到类成员变量 remapX/Y 中
	 *  该函数也比较有意思，它是基类中的函数，但是在其中调用了派生类的函数(也就是基类中的虚函数)，
     *  从而针对不同类型的相机畸变模型进行了校正
	 * @param[in] configFileName   图像内参文件名称
	 * @param[in] nPars   相机内参的参数个数，5个或者8个
	 * @param[in] prefix 
	 */
	void Undistort::readFromFile(const char *configFileName, int nPars, std::string prefix)
	{
		photometricUndist = 0;
		valid = false;
		passthrough = false;
		remapX = 0;
		remapY = 0;

		float outputCalibration[5];

		//; typedef Eigen::Matrix<double,Eigen::Dynamic,1> VecX;
		parsOrg = VecX(nPars); // 相机原始参数

		// read parameters
		std::ifstream infile(configFileName);
		assert(infile.good());

		std::string l1, l2, l3, l4; //; 读取4行
		std::getline(infile, l1);
		std::getline(infile, l2);
		std::getline(infile, l3);
		std::getline(infile, l4);

		// Step 1 读取内参和原始分辨率
		//* 第一行, 相机模型参数; 第二行, 原始输入图片大小
		// l1 & l2
		if (nPars == 5) // fov model
		{
			char buf[1000];
			// 复制 prefix 最大1000个, 以%s格式, 到 buf (char) 数组, %%表示输出%
			// 因此buf中是 "fov%lf %lf %lf %lf %lf"
			snprintf(buf, 1000, "%s%%lf %%lf %%lf %%lf %%lf", prefix.c_str());

			// 使用buf做格式控制, 将l1输出到这5个参数
			if (std::sscanf(l1.c_str(), buf, &parsOrg[0], &parsOrg[1], &parsOrg[2], &parsOrg[3], &parsOrg[4]) == 5 &&
				std::sscanf(l2.c_str(), "%d %d", &wOrg, &hOrg) == 2) // 得到像素大小
			{
				printf("Input resolution: %d %d\n", wOrg, hOrg); //; 原始图像分辨率
				printf("In: %f %f %f %f %f\n",
					   parsOrg[0], parsOrg[1], parsOrg[2], parsOrg[3], parsOrg[4]);
			}
			else
			{
				printf("Failed to read camera calibration (invalid format?)\nCalibration file: %s\n", configFileName);
				infile.close();
				return;
			}
		}
		else if (nPars == 8) // KB, equi & radtan model
		{
			char buf[1000];
			snprintf(buf, 1000, "%s%%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf", prefix.c_str());
			// line1:相机参数， line2: 图像宽高
			if (std::sscanf(l1.c_str(), buf,
							&parsOrg[0], &parsOrg[1], &parsOrg[2], &parsOrg[3], &parsOrg[4],
							&parsOrg[5], &parsOrg[6], &parsOrg[7]) == 8 &&
				std::sscanf(l2.c_str(), "%d %d", &wOrg, &hOrg) == 2)
			{
				printf("Input resolution: %d %d\n", wOrg, hOrg);
				printf("In: %s%f %f %f %f %f %f %f %f\n",
					   prefix.c_str(),
					   parsOrg[0], parsOrg[1], parsOrg[2], parsOrg[3], parsOrg[4], parsOrg[5], parsOrg[6], parsOrg[7]);
			}
			else
			{
				printf("Failed to read camera calibration (invalid format?)\nCalibration file: %s\n", configFileName);
				infile.close();
				return;
			}
		}
		else
		{
			printf("called with invalid number of parameters.... forgot to implement me?\n");
			infile.close();
			return;
		}

		// cx, cy 小于1, 则说明是个相对值, 乘上图像大小
		if (parsOrg[2] < 1 && parsOrg[3] < 1)
		{
			printf("\n\nFound fx=%f, fy=%f, cx=%f, cy=%f.\n I'm assuming this is the \"relative\" calibration file format,"
				   "and will rescale this by image width / height to fx=%f, fy=%f, cx=%f, cy=%f.\n\n",
				   parsOrg[0], parsOrg[1], parsOrg[2], parsOrg[3],
				   parsOrg[0] * wOrg, parsOrg[1] * hOrg, parsOrg[2] * wOrg - 0.5, parsOrg[3] * hOrg - 0.5);

			//?? 0.5 还是不是很理解, 为了使用积分来近似像素强度
			// rescale and substract 0.5 offset.
			// the 0.5 is because I'm assuming the calibration is given such that the pixel at (0,0)
			// contains the integral over intensity over [0,0]-[1,1], whereas I assume the pixel (0,0)
			// to contain a sample of the intensity ot [0,0], which is best approximated by the integral over
			// [-0.5,-0.5]-[0.5,0.5]. Thus, the shift by -0.5.
			//; 0.5 是因为我假设给出了校准，使得 (0,0) 处的像素包含 [0,0]-[1,1] 上的积分强度，而我假设像素 (0,0)
			//;   包含 [0,0] 处的强度样本，最好通过 [-0.5,-0.5]-[0.5,0.5] 上的积分来近似。 因此，偏移 -0.5。
			//! 这个地方具体的解释在github的readme中有解释:
			/* 那个奇怪的“0.5”偏移：在内部，DSO 使用图像中整数位置(1,1) 的像素，即第二行和第二列中的像素，
			 包含从 (0.5 ,0.5) 到 (1.5,1.5)，即近似于 (1.0, 1.0) 处的连续图像函数的“点样本”。 反过来，
			 校准工具箱之间似乎没有统一的约定，整数位置 (1,1) 处的像素是否包含 (0.5,0.5) 到 (1.5,1.5) 
			 上的积分，或者 (1,1) 到 (2,2)上的积分。 上述转换假定校准文件中的给定校准使用后一种约定，
			 因此应用 -0.5 校正。 请注意，在创建比例金字塔时也会考虑到这一点（请参阅 globalCalib.cpp）。
			*/
			parsOrg[0] = parsOrg[0] * wOrg;
			parsOrg[1] = parsOrg[1] * hOrg;
			parsOrg[2] = parsOrg[2] * wOrg - 0.5;
			parsOrg[3] = parsOrg[3] * hOrg - 0.5;
		}

		// Step 2 读取纠正图像的方式（是否裁切）、想要的输出图像大小
		//* 第三行, 相机图像类别, 是否裁切
		//? 注意这部分在github的readme中也有说，即如何矫正图像：
		//; 1.裁剪图像：会自动裁剪图像到最大的矩形、定义明确的区域（也就是去畸变后的图像没有黑边）
		if (l3 == "crop")
		{
			outputCalibration[0] = -1;
			printf("Out: Rectify Crop\n");
		}
		//; 2.保留完整的原始视野：主要用于可视化调试，因为他会在未定义的图像区域中创建黑色边框（对后续算法有影响）
		else if (l3 == "full")
		{
			outputCalibration[0] = -2;
			printf("Out: Rectify Full\n");
		}
		//! 3.啥也不干：这个情况应该是不存在的，github上作者也没有提
		else if (l3 == "none")
		{
			outputCalibration[0] = -3;
			printf("Out: No Rectification\n");
		}
		//; 4.矫正图像到用户定义的针孔相机模型参数上，针孔相机参数fx fy cx cy 0
		else if (std::sscanf(l3.c_str(), "%f %f %f %f %f", &outputCalibration[0], &outputCalibration[1], &outputCalibration[2], &outputCalibration[3], &outputCalibration[4]) == 5)
		{
			//! 这里为啥没有赋值outputCalibration[0]的结果？
			printf("Out: %f %f %f %f %f\n",
				   outputCalibration[0], outputCalibration[1], outputCalibration[2], outputCalibration[3], outputCalibration[4]);
		}
		else
		{
			printf("Out: Failed to Read Output pars... not rectifying.\n");
			infile.close();
			return;
		}

		//* 第四行, 图像的大小, 会根据设置进行裁切...
		// l4
		//? 输出的分辨率确实被读取到类成员变量w和h中了！
		//! 问题：这里的输出大小代表什么意思呢？配置文件第3行如果是crop，会自动裁剪图像变成去畸变后的无黑边的
		//!  最大图像，这里又有一个输出大小。
		//! 解答：github issue   https://github.com/JakobEngel/dso/issues/71
		//!  里面说这是在矫正图像的时候，允许对输出图像进行resize的操作，让输出图像大小和自己想要的大小一样
		if (std::sscanf(l4.c_str(), "%d %d", &w, &h) == 2)
		{
			// 如果有代码中固定设置的大小
			//; 这个参数是在settings.cpp中定义的，默认就是0，所以这个分支并没有用
			//; github上作者说不要动这些参数：most of which you shouldn't touch
			if (benchmarkSetting_width != 0)
			{
				w = benchmarkSetting_width;
				if (outputCalibration[0] == -3)
					outputCalibration[0] = -1; // crop instead of none, since probably resolution changed.
			}
			if (benchmarkSetting_height != 0)
			{
				h = benchmarkSetting_height;
				if (outputCalibration[0] == -3)
					outputCalibration[0] = -1; // crop instead of none, since probably resolution changed.
			}
			printf("Output resolution: %d %d\n", w, h);
		}
		else
		{
			printf("Out: Failed to Read Output resolution... not rectifying.\n");
			valid = false;
		}

		//; 无畸变的图像和有畸变的图像之间，像素关系的映射
		remapX = new float[w * h];
		remapY = new float[w * h];

		// Step 3 根据配置文件中对图像裁剪的设置，对图像进行不同的矫正
		if (outputCalibration[0] == -1)
			//; 对归一化平面上的位置进行裁切，找到畸变之后仍然全部在输入图像范围内的归一化平面范围（定义为 有效归一化平面范围）。
			//;  同时会计算出这个归一化平面范围到设置的 输出图像 像素坐标之间的对应关系，也就是新的内参K
			makeOptimalK_crop();
		else if (outputCalibration[0] == -2)
			makeOptimalK_full();
		else if (outputCalibration[0] == -3)
		{
			if (w != wOrg || h != hOrg)
			{
				printf("ERROR: rectification mode none requires input and output dimenstions to match!\n\n");
				exit(1);
			}
			K.setIdentity();
			K(0, 0) = parsOrg[0];
			K(1, 1) = parsOrg[1];
			K(0, 2) = parsOrg[2];
			K(1, 2) = parsOrg[3];
			//; 这个passthrough其实是一个是否直接 经过 的标志，实际上就是如果裁切方式是none
			//; 那么输入图像和输出图像大小必须设置一样，就不进行去畸变操作，输入图像像素值直接就是
			//; 输出图像像素值
			passthrough = true;
		}
		else
		{
			// 标定输出错误
			if (outputCalibration[2] > 1 || outputCalibration[3] > 1)
			{
				printf("\n\n\nWARNING: given output calibration (%f %f %f %f) seems wrong. It needs to be relative to image width / height!\n\n\n",
					   outputCalibration[0], outputCalibration[1], outputCalibration[2], outputCalibration[3]);
			}

			// 相对于长宽的比例值（TUMmono这样）
			K.setIdentity();
			K(0, 0) = outputCalibration[0] * w;
			K(1, 1) = outputCalibration[1] * h;
			K(0, 2) = outputCalibration[2] * w - 0.5;
			K(1, 2) = outputCalibration[3] * h - 0.5;
		}

		// 设置了fx和fy，则取最好的
		//; 这个是setting.cpp中的参数，默认是0，所以这里不满足
		if (benchmarkSetting_fxfyfac != 0)
		{
			K(0, 0) = fmax(benchmarkSetting_fxfyfac, (float)K(0, 0));
			K(1, 1) = fmax(benchmarkSetting_fxfyfac, (float)K(1, 1));
			passthrough = false; // cannot pass through when fx / fy have been overwritten.
		}

		//* 计算图像矫正的remapX和remapY
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				remapX[x + y * w] = x;
				remapY[x + y * w] = y;
			}
		}

		// Step 4 正向添加畸变，得到输出的去畸变图像对应畸变图像的位置
		//; 经过上面的代码阅读之后，就会发现这个函数的巧妙之处了！
		//;  1.上面裁剪归一化平面之后，得到了 有效归一化平面范围（即该范围内的点经过畸变后，全部都能落到输入图像中）
		//;     同时根据设置的输出图像大小，可以计算有效归一化平面范围和输出图像像素坐标之间的内参K，
		//;  2.这里再把输出图像的像素坐标输入这个函数，进入后函数内部会先利用内参K把图像投影到归一化平面上，这个内参
		//;     K正好就是上面我们求得那个内参。然后步骤就和第1步一样了，给归一化平面上的点加畸变，然后用输入图像
		//;     的内参投影到输入图像中，得到对应于输入图像的像素坐标位置。由于我们在第1步骤确定了这个归一化平面的范围
		//;     投影后一定在输入图像范围内，所以能够保证输出图像的像素点全部都能找到输入图像中对应的像素点
		distortCoordinates(remapX, remapY, remapX, remapY, h * w);

		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				// make rounding resistant.  抗舍入
				float ix = remapX[x + y * w];
				float iy = remapY[x + y * w];

				if (ix == 0)
					ix = 0.001;
				if (iy == 0)
					iy = 0.001;
				if (ix == wOrg - 1)
					ix = wOrg - 1.001;
				if (iy == hOrg - 1)
					ix = hOrg - 1.001;

				if (ix > 0 && iy > 0 && ix < wOrg - 1 && iy < wOrg - 1)
				{
					//; 靠，这个和之前不是一样的吗？
					remapX[x + y * w] = ix;
					remapY[x + y * w] = iy;
				}
				else
				{
					//; 标志-1说明图像出界，没有对应的畸变像素。这个应该不会发生，可以打印看看效果
					//TODO 添加打印
					remapX[x + y * w] = -1;
					remapY[x + y * w] = -1;
				}
			}
		}

		valid = true;

		printf("\nRectified Kamera Matrix:\n");
		std::cout << K << "\n\n";
	}


	/**
	 * @brief 相机几何畸变模型的构造函数，注意在这个构造函数中调用readFromFile就完成了相机几何畸变校正
	 * 
	 * @param[in] configFileName 
	 * @param[in] noprefix 
	 */
	UndistortFOV::UndistortFOV(const char *configFileName, bool noprefix)
	{
		printf("Creating FOV undistorter\n");

		//; 默认调用的都是这个分支
		if (noprefix)
			readFromFile(configFileName, 5);
		else
			readFromFile(configFileName, 5, "FOV ");
	}
	UndistortFOV::~UndistortFOV()
	{
	}


    //********************* 以下都是加畸变的算法 *******************************
	// FOV加畸变：dso的数据集中最常用到的！
	void UndistortFOV::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const
	{
		float dist = parsOrg[4]; //; 配置文件中的最后一个畸变系数，就是w
		float d2t = 2.0f * tan(dist / 2.0f);

		// current camera parameters
        //; 输入图像 到 有效归一化平面 之间的投影矩阵，即真实的相机内参
		float fx = parsOrg[0];
		float fy = parsOrg[1];
		float cx = parsOrg[2];
		float cy = parsOrg[3];

		//; 输出图像 到 有效归一化平面 之间的投影矩阵，也就是我们利用输出图像大小计算的虚拟的相机内参
		float ofx = K(0, 0); // 1
		float ofy = K(1, 1); // 1
		float ocx = K(0, 2); // 0
		float ocy = K(1, 2); // 0

		//; 遍历100000个点
		for (int i = 0; i < n; i++)
		{
			float x = in_x[i]; //; x从-5变到5
			float y = in_y[i]; //; y始终是0
			//; 这里想要做的应该就是把(x, y)这个点投到归一化平面上
			float ix = (x - ocx) / ofx;
			float iy = (y - ocy) / ofy;

			//; 计算畸变
			float r = sqrtf(ix * ix + iy * iy);
			float fac = (r == 0 || dist == 0) ? 1 : atanf(r * d2t) / (dist * r);

			//; 施加畸变后，对应在输入图像上的位置
			ix = fx * fac * ix + cx;
			iy = fy * fac * iy + cy;

			//; 这样就得到，图像上这个点，对应在畸变后的图像上的位置
			out_x[i] = ix;
			out_y[i] = iy;
		}
	}

	UndistortRadTan::UndistortRadTan(const char *configFileName, bool noprefix)
	{
		printf("Creating RadTan undistorter\n");

		// 默认这个输入参数都是true
		if (noprefix)
			readFromFile(configFileName, 8);
		else
			readFromFile(configFileName, 8, "RadTan ");
	}
	UndistortRadTan::~UndistortRadTan()
	{
	}

	void UndistortRadTan::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const
	{
		// RADTAN
		float fx = parsOrg[0];
		float fy = parsOrg[1];
		float cx = parsOrg[2];
		float cy = parsOrg[3];
		float k1 = parsOrg[4];
		float k2 = parsOrg[5];
		float r1 = parsOrg[6];
		float r2 = parsOrg[7];

		float ofx = K(0, 0);
		float ofy = K(1, 1);
		float ocx = K(0, 2);
		float ocy = K(1, 2);

		for (int i = 0; i < n; i++)
		{
			float x = in_x[i];
			float y = in_y[i];

			// RADTAN
			//; 一张正常的图片，通过反投影到归一化平面上，得到归一化平面上的坐标
			float ix = (x - ocx) / ofx;
			float iy = (y - ocy) / ofy;
			//; 下面对归一化平面上的点，利用radtan模型计算畸变
			float mx2_u = ix * ix;
			float my2_u = iy * iy;
			float mxy_u = ix * iy;
			float rho2_u = mx2_u + my2_u;
			float rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
			float x_dist = ix + ix * rad_dist_u + 2.0 * r1 * mxy_u + r2 * (rho2_u + 2.0 * mx2_u);
			float y_dist = iy + iy * rad_dist_u + 2.0 * r2 * mxy_u + r1 * (rho2_u + 2.0 * my2_u);
			//; 畸变之后的点投影到像素平面上的位置
			float ox = fx * x_dist + cx;
			float oy = fy * y_dist + cy;

			//; out_xy就是无畸变的图像，对应畸变的图像的索引
			out_x[i] = ox;
			out_y[i] = oy;
		}
	}

	UndistortEquidistant::UndistortEquidistant(const char *configFileName, bool noprefix)
	{
		printf("Creating Equidistant undistorter\n");

		if (noprefix)
			readFromFile(configFileName, 8);
		else
			readFromFile(configFileName, 8, "EquiDistant ");
	}
	UndistortEquidistant::~UndistortEquidistant()
	{
	}

	void UndistortEquidistant::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const
	{
		// EQUI
		float fx = parsOrg[0];
		float fy = parsOrg[1];
		float cx = parsOrg[2];
		float cy = parsOrg[3];
		float k1 = parsOrg[4];
		float k2 = parsOrg[5];
		float k3 = parsOrg[6];
		float k4 = parsOrg[7];

		float ofx = K(0, 0);
		float ofy = K(1, 1);
		float ocx = K(0, 2);
		float ocy = K(1, 2);

		for (int i = 0; i < n; i++)
		{
			float x = in_x[i];
			float y = in_y[i];

			// EQUI
			float ix = (x - ocx) / ofx;
			float iy = (y - ocy) / ofy;
			float r = sqrt(ix * ix + iy * iy);
			float theta = atan(r);
			float theta2 = theta * theta;
			float theta4 = theta2 * theta2;
			float theta6 = theta4 * theta2;
			float theta8 = theta4 * theta4;
			float thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);
			float scaling = (r > 1e-8) ? thetad / r : 1.0;
			float ox = fx * ix * scaling + cx;
			float oy = fy * iy * scaling + cy;

			out_x[i] = ox;
			out_y[i] = oy;
		}
	}

	UndistortKB::UndistortKB(const char *configFileName, bool noprefix)
	{
		printf("Creating KannalaBrandt undistorter\n");

		if (noprefix)
			readFromFile(configFileName, 8);
		else
			readFromFile(configFileName, 8, "KannalaBrandt ");
	}
	UndistortKB::~UndistortKB()
	{
	}

	void UndistortKB::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const
	{
		const float fx = parsOrg[0];
		const float fy = parsOrg[1];
		const float cx = parsOrg[2];
		const float cy = parsOrg[3];
		const float k0 = parsOrg[4];
		const float k1 = parsOrg[5];
		const float k2 = parsOrg[6];
		const float k3 = parsOrg[7];

		const float ofx = K(0, 0);
		const float ofy = K(1, 1);
		const float ocx = K(0, 2);
		const float ocy = K(1, 2);

		for (int i = 0; i < n; i++)
		{
			float x = in_x[i];
			float y = in_y[i];

			// RADTAN
			float ix = (x - ocx) / ofx;
			float iy = (y - ocy) / ofy;

			const float Xsq_plus_Ysq = ix * ix + iy * iy;
			const float sqrt_Xsq_Ysq = sqrtf(Xsq_plus_Ysq);
			const float theta = atan2f(sqrt_Xsq_Ysq, 1);
			const float theta2 = theta * theta;
			const float theta3 = theta2 * theta;
			const float theta5 = theta3 * theta2;
			const float theta7 = theta5 * theta2;
			const float theta9 = theta7 * theta2;
			const float r = theta + k0 * theta3 + k1 * theta5 + k2 * theta7 + k3 * theta9;

			if (sqrt_Xsq_Ysq < 1e-6)
			{
				out_x[i] = fx * ix + cx;
				out_y[i] = fy * iy + cy;
			}
			else
			{
				out_x[i] = (r / sqrt_Xsq_Ysq) * fx * ix + cx;
				out_y[i] = (r / sqrt_Xsq_Ysq) * fy * iy + cy;
			}
		}
	}

	UndistortPinhole::UndistortPinhole(const char *configFileName, bool noprefix)
	{
		if (noprefix)
			readFromFile(configFileName, 5);
		else
			readFromFile(configFileName, 5, "Pinhole ");
	}
	UndistortPinhole::~UndistortPinhole()
	{
	}

	void UndistortPinhole::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const
	{
		// current camera parameters
		float fx = parsOrg[0];
		float fy = parsOrg[1];
		float cx = parsOrg[2];
		float cy = parsOrg[3];

		float ofx = K(0, 0);
		float ofy = K(1, 1);
		float ocx = K(0, 2);
		float ocy = K(1, 2);

		for (int i = 0; i < n; i++)
		{
			float x = in_x[i];
			float y = in_y[i];
			float ix = (x - ocx) / ofx;
			float iy = (y - ocy) / ofy;
			ix = fx * ix + cx;
			iy = fy * iy + cy;
			out_x[i] = ix;
			out_y[i] = iy;
		}
	}

}
