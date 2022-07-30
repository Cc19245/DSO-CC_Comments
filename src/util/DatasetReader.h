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

#pragma once
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "util/Undistort.h"
#include "IOWrapper/ImageRW.h"

#if HAS_ZIPLIB
#include "zip.h"
#endif

#include <boost/thread.hpp>

using namespace dso;

//; 直接根据文件夹读取图片名称，而不是压缩包
inline int getdir(std::string dir, std::vector<std::string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(dir.c_str())) == NULL)
	{
		return -1;
	}

	while ((dirp = readdir(dp)) != NULL)
	{
		std::string name = std::string(dirp->d_name);

		if (name != "." && name != "..")
			files.push_back(name);
	}
	closedir(dp);

	std::sort(files.begin(), files.end());

	if (dir.at(dir.length() - 1) != '/')
		dir = dir + "/";
	for (unsigned int i = 0; i < files.size(); i++)
	{
		if (files[i].at(0) != '/')
			files[i] = dir + files[i];
	}

	return files.size();
}

struct PrepImageItem
{
	int id;
	bool isQueud;
	ImageAndExposure *pt;

	inline PrepImageItem(int _id)
	{
		id = _id;
		isQueud = false;
		pt = 0;
	}

	inline void release()
	{
		if (pt != 0)
			delete pt;
		pt = 0;
	}
};

class ImageFolderReader
{
public:
	ImageFolderReader(std::string path, std::string calibFile, std::string gammaFile, std::string vignetteFile)
	{
		this->path = path;
		this->calibfile = calibFile;

#if HAS_ZIPLIB
		ziparchive = 0;
		databuffer = 0;
#endif
		// Step 1 读取压缩包内所有图片的名称，存入 成员变量files
		// 判断输入图像压缩包名称的合法性
		isZipped = (path.length() > 4 && path.substr(path.length() - 4) == ".zip");
		//; 如果传入的路径是一个压缩包，那么用ziplib库读取压缩包
		if (isZipped)
		{
#if HAS_ZIPLIB
			int ziperror = 0;
			// 读取zip文件，存放到ziparchive中
			ziparchive = zip_open(path.c_str(), ZIP_RDONLY, &ziperror);
			if (ziperror != 0)
			{
				printf("ERROR %d reading archive %s!\n", ziperror, path.c_str());
				exit(1);
			}

			files.clear();
			int numEntries = zip_get_num_entries(ziparchive, 0);
			// 遍历zip中的每一个文件
			for (int k = 0; k < numEntries; k++)
			{
				const char *name = zip_get_name(ziparchive, k, ZIP_FL_ENC_STRICT);
				std::string nstr = std::string(name);
				if (nstr == "." || nstr == "..")
					continue;
				files.push_back(name); //; 保存zip中每张图片的名称
			}
			printf("got %d entries and %d files!\n", numEntries, (int)files.size());
			std::sort(files.begin(), files.end());
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
		//; 如果传入的图片路径不是压缩包，那么直接读取文件夹
		else
			getdir(path, files);

		// 去畸变的参数
		// Step 2 重要：这里面主要是把相机的参数加载进来，执行的操作如下：
		//;  1.建立相机畸变模型，其中包括计算正常图像和畸变图像的像素之间的对应关系
		//;  2.建立相机光度模型，包括辐照B和光度I的非线性仿射函数G、镜头渐晕V
		undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);

        // Step 2.1 把Undistort类中的成员变量，拷贝复制到当前ImageFolderReader类中
		//; 内参文件第2行：原始图像大小
		widthOrg = undistort->getOriginalSize()[0];
		heightOrg = undistort->getOriginalSize()[1];
		//; 内参文件第4行：输出图像大小
		width = undistort->getSize()[0];
		height = undistort->getSize()[1];

		// load timestamps if possible.
        // Step 3 如果提供了图像时间戳和曝光时间，那么也读到类的成员变量中
        //; 所以说这个函数重要的地方就在于它对曝光时间的读取
		loadTimestamps(); // 需要有times.txt文件
		printf("ImageFolderReader: got %d files in %s!\n", (int)files.size(), path.c_str());
	}
	~ImageFolderReader()
	{
#if HAS_ZIPLIB
		if (ziparchive != 0)
			zip_close(ziparchive);
		if (databuffer != 0)
			delete databuffer;
#endif

		delete undistort;
	};

	Eigen::VectorXf getOriginalCalib()
	{
		return undistort->getOriginalParameter().cast<float>();
	}
	Eigen::Vector2i getOriginalDimensions()
	{
		return undistort->getOriginalSize();
	}

	void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
	{
		K = undistort->getK().cast<float>();
		w = undistort->getSize()[0];
		h = undistort->getSize()[1];
	}

	void setGlobalCalibration()
	{
		int w_out, h_out;
		Eigen::Matrix3f K;
		// 1.获取前面根据相机内参文件建立的内参矩阵
		getCalibMono(K, w_out, h_out);

		// 2.建立金字塔，各层金字塔之间的比例为2，并且计算各层金字塔的内参
		//   根据图像输出大小确定金字塔层数，原始图像为第0层，最高层为pyrLevelsUsed-1层。
		//   注意：最高层图像的高和宽要大于100,并且pyrLevelsUsed要大于等于3。
		setGlobalCalib(w_out, h_out, K);
	}

	int getNumImages()
	{
		return files.size();
	}

	double getTimestamp(int id)
	{
		if (timestamps.size() == 0)
			return id * 0.1f;
		if (id >= (int)timestamps.size())
			return 0;
		if (id < 0)
			return 0;
		return timestamps[id];
	}

	void prepImage(int id, bool as8U = false)
	{
	}

	MinimalImageB *getImageRaw(int id)
	{
		return getImageRaw_internal(id, 0);
	}

    /**
     * @brief 根据输入的图片，对其校正光度，得到最后计算使用的 辐照*曝光时间
     * 
     * @param[in] id  要读取的图片在图片文件夹中的索引
     * @param[in] forceLoadDirectly  没用
     * @return ImageAndExposure* 
     */
	ImageAndExposure *getImage(int id, bool forceLoadDirectly = false)
	{
		return getImage_internal(id, 0);
	}


	inline float *getPhotometricGamma()
	{
		//; 如果几何畸变或者光度畸变模型有一个没有建立，就返回空指针
		if (undistort == 0 || undistort->photometricUndist == 0)
			return 0;
		//; 否则返回响应函数的逆变换
		return undistort->photometricUndist->getG();
	}

	//; 这个应该是一个类成员变量，是一个Undistort类型的指针
	// undistorter. [0] always exists, [1-2] only when MT is enabled.
	Undistort* undistort;

private:
	//; 根据图像Id（其实就是压缩包中第几张图像）读取对应的图像
	MinimalImageB *getImageRaw_internal(int id, int unused)
	{
		if (!isZipped)
		{
			// CHANGE FOR ZIP FILE
			return IOWrap::readImageBW_8U(files[id]);
		}
		else
		{
#if HAS_ZIPLIB
			if (databuffer == 0)
				databuffer = new char[widthOrg * heightOrg * 6 + 10000];
			zip_file_t *fle = zip_fopen(ziparchive, files[id].c_str(), 0);
			long readbytes = zip_fread(fle, databuffer, (long)widthOrg * heightOrg * 6 + 10000);

			if (readbytes > (long)widthOrg * heightOrg * 6)
			{
				printf("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes, (long)widthOrg * heightOrg * 6 + 10000, files[id].c_str());
				delete[] databuffer;
				databuffer = new char[(long)widthOrg * heightOrg * 30];
				fle = zip_fopen(ziparchive, files[id].c_str(), 0);
				readbytes = zip_fread(fle, databuffer, (long)widthOrg * heightOrg * 30 + 10000);

				if (readbytes > (long)widthOrg * heightOrg * 30)
				{
					printf("buffer still to small (read %ld/%ld). abort.\n", readbytes, (long)widthOrg * heightOrg * 30 + 10000);
					exit(1);
				}
			}

			return IOWrap::readStreamBW_8U(databuffer, readbytes);
#else
			printf("ERROR: cannot read .zip archive, as compile without ziplib!\n");
			exit(1);
#endif
		}
	}

	/**
	 * @brief 读取输入图片，对其校正光度畸变和几何畸变，然后返回 辐照值*曝光时间 
	 * 
	 * @param[in] id  输入的图片在文件夹中的索引
	 * @param[in] unused 
	 * @return ImageAndExposure*  
	 */
	ImageAndExposure *getImage_internal(int id, int unused)
	{
		// Step 1 从原始压缩文件中读取对应图像，存到自定义图像结构中
		MinimalImageB *minimg = getImageRaw_internal(id, 0);

		// Step 2 得到去畸变(包括光度畸变和几何畸变)后的辐照图像 ImageAndExposure：
		//; 1.对整个 输入图片 去除光度畸变，包括非线性响应、渐晕的光度畸变
		//; 2.对 输出图片 ，根据畸变的remapX和Y找到对应的原图中的位置，得到输出图片的辐照，就是在去几何畸变
		ImageAndExposure *ret2 = undistort->undistort<unsigned char>(
			minimg, //; 读取的数据集中的原始图像
			(exposures.size() == 0 ? 1.0f : exposures[id]),
			(timestamps.size() == 0 ? 0.0 : timestamps[id]));
		delete minimg;
		return ret2;
	}

	inline void loadTimestamps()
	{
		std::ifstream tr;
		std::string timesFile = path.substr(0, path.find_last_of('/')) + "/times.txt";
		tr.open(timesFile.c_str());
		while (!tr.eof() && tr.good())
		{
			std::string line;
			char buf[1000];
			tr.getline(buf, 1000);

			int id;
			double stamp;
			float exposure = 0;

			if (3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}

			else if (2 == sscanf(buf, "%d %lf", &id, &stamp))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}
		}
		tr.close();

		// check if exposures are correct, (possibly skip)
		bool exposuresGood = ((int)exposures.size() == (int)getNumImages());
		for (int i = 0; i < (int)exposures.size(); i++)
		{
			if (exposures[i] == 0)
			{
				// fix!
				float sum = 0, num = 0;
				if (i > 0 && exposures[i - 1] > 0)
				{
					sum += exposures[i - 1];
					num++;
				}
				if (i + 1 < (int)exposures.size() && exposures[i + 1] > 0)
				{
					sum += exposures[i + 1];
					num++;
				}

				if (num > 0)
					exposures[i] = sum / num;
			}

			if (exposures[i] == 0)
				exposuresGood = false;
		}

		if ((int)getNumImages() != (int)timestamps.size())
		{
			printf("set timestamps and exposures to zero!\n");
			exposures.clear();
			timestamps.clear();
		}

		if ((int)getNumImages() != (int)exposures.size() || !exposuresGood)
		{
			printf("set EXPOSURES to zero!\n");
			exposures.clear();
		}

		printf("got %d images and %d timestamps and %d exposures.!\n", (int)getNumImages(), (int)timestamps.size(), (int)exposures.size());
	}

	//------ 私有成员变量
	std::vector<ImageAndExposure *> preloadedImages;
	std::vector<std::string> files; //; zip包中每张图片的名称
	std::vector<double> timestamps; //; 数据集中的图像时间戳
	std::vector<float> exposures;	//; 数据集中的图像曝光时间

	int width, height;
	int widthOrg, heightOrg;

	std::string path;	   //; 图片源文件的文件夹路径
	std::string calibfile; //; 相机内参文件

	bool isZipped;

#if HAS_ZIPLIB
	zip_t *ziparchive;
	char *databuffer;
#endif
};
