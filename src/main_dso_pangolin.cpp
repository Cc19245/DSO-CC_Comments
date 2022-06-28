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

#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"

#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"

#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

std::string vignette = "";	 // 镜头渐晕png图片
std::string gammaCalib = ""; // gamma响应函数校准
std::string source = "";	 // 图像源文件，zip压缩包
std::string calib = "";		 // 相机内参文件
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start = 0;
int end = 100000;
bool prefetch = false;
float playbackSpeed = 0; // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload = false;
bool useSampleOutput = false;

int mode = 0; // 是否有光度校准，0：有光度校准文件， 1：没有光度校准文件，自己估计a和b， 2：图片已经进行过光度校准

bool firstRosSpin = false;

using namespace dso;

void my_exit_handler(int s)
{
	printf("Caught signal %d\n", s);
	exit(1);
}

//; Ctrl+C 退出程序
void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin = true;
	while (true)
		pause();
}

/**
 * @brief 根据preset设置一些运行参数，主要是提取的点个数、是否实时运行
 * 
 * @param[in] preset 
 *   0: 默认设置，2k个点，不强制实时运行
 *   1: 默认设置，2k个点，强制1倍速实时运行
 *   2: 快速的设置，800个点，不强制实时运行（会把图片分辨率调整成424x320）
 *   3: 快速的设置，800个点，强制5倍速实时运行（会把图片分辨率调整成424x320）
 */
void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if (preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
			   "- %s real-time enforcing\n"
			   "- 2000 active points\n"
			   "- 5-7 active frames\n"
			   "- 1-6 LM iteration each KF\n"
			   "- original image resolution\n",
			   preset == 0 ? "no " : "1x");

		playbackSpeed = (preset == 0 ? 0 : 1);
		preload = preset == 1;
		setting_desiredImmatureDensity = 1500;
		setting_desiredPointDensity = 2000;
		setting_minFrames = 5;
		setting_maxFrames = 7;
		setting_maxOptIterations = 6;
		setting_minOptIterations = 1;

		setting_logStuff = false;
	}

	if (preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
			   "- %s real-time enforcing\n"
			   "- 800 active points\n"
			   "- 4-6 active frames\n"
			   "- 1-4 LM iteration each KF\n"
			   "- 424 x 320 image resolution\n",
			   preset == 0 ? "no " : "5x");

		playbackSpeed = (preset == 2 ? 0 : 5);
		preload = preset == 3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations = 4;
		setting_minOptIterations = 1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}

/**
 * @brief 读取命令行输入的文件
 * 
 * @param[in] arg 
 */
void parseArgument(char *arg)
{
	int option;
	float foption;
	char buf[1000];

	// sscanf是C语言库函数从字符串读取格式化输入
	//; 输出一些例子到命令行
	if (1 == sscanf(arg, "sampleoutput=%d", &option))
	{
		if (option == 1)
		{
			useSampleOutput = true;
			printf("USING SAMPLE OUTPUT WRAPPER!\n");
		}
		return;
	}
	//; 关闭大部分的命令行输出（对算法运行有利）
	if (1 == sscanf(arg, "quiet=%d", &option))
	{
		if (option == 1)
		{
			setting_debugout_runquiet = true;
			printf("QUIET MODE, I'll shut up!\n");
		}
		return;
	}

	// preset设定DSO运行时的参数，比如选取像素点的个数等。preset=3是preset=0的5倍速度运行DSO
	if (1 == sscanf(arg, "preset=%d", &option))
	{
		settingsDefault(option);
		return;
	}

	if (1 == sscanf(arg, "rec=%d", &option))
	{
		if (option == 0)
		{
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
		return;
	}

	if (1 == sscanf(arg, "noros=%d", &option))
	{
		if (option == 1)
		{
			disableROS = true;
			disableReconfigure = true;
			printf("DISABLE ROS (AND RECONFIGURE)!\n");
		}
		return;
	}
	//; 关闭一些日志输出：比如特征值的输出（这样对算法的表现有利）
	if (1 == sscanf(arg, "nolog=%d", &option))
	{
		if (option == 1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}
	//; 数据集是否倒放
	if (1 == sscanf(arg, "reverse=%d", &option))
	{
		if (option == 1)
		{
			reverse = true;
			printf("REVERSE!\n");
		}
		return;
	}
	//; 是否显示gui界面（关闭对算法表现有好处）
	if (1 == sscanf(arg, "nogui=%d", &option))
	{
		if (option == 1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if (1 == sscanf(arg, "nomt=%d", &option))
	{
		if (option == 1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	//; 是否实现把所有图片载入内存，这样后面运行就不用每次都从硬盘读取新的图片了
	if (1 == sscanf(arg, "prefetch=%d", &option))
	{
		if (option == 1)
		{
			prefetch = true;
			printf("PREFETCH!\n");
		}
		return;
	}
	//; 读取起始和结束的帧的位置，应该是可以对数据集中间的一段进行播放
	if (1 == sscanf(arg, "start=%d", &option))
	{
		start = option;
		printf("START AT %d!\n", start);
		return;
	}
	if (1 == sscanf(arg, "end=%d", &option))
	{
		end = option;
		printf("END AT %d!\n", start);
		return;
	}

	//; 读取图像文件, source
	if (1 == sscanf(arg, "files=%s", buf))
	{
		source = buf;
		printf("loading data from %s!\n", source.c_str());
		return;
	}
	//; 相机内参
	if (1 == sscanf(arg, "calib=%s", buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}
	//; 渐晕图 png
	if (1 == sscanf(arg, "vignette=%s", buf))
	{
		vignette = buf;
		printf("loading vignette from %s!\n", vignette.c_str());
		return;
	}
	//; 响应函数校准：这里可以看出来，非线性响应函数、gamma函数、伽马校准说的都是一件事
	if (1 == sscanf(arg, "gamma=%s", buf))
	{
		gammaCalib = buf;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
		return;
	}

	if (1 == sscanf(arg, "rescale=%f", &foption))
	{
		rescale = foption;
		printf("RESCALE %f!\n", rescale);
		return;
	}
	//; 运行速度，如果是0不强制实时运行，如果不是0，那么强制以几倍速运行
	if (1 == sscanf(arg, "speed=%f", &foption))
	{
		playbackSpeed = foption;
		printf("PLAYBACK SPEED %f!\n", playbackSpeed);
		return;
	}
	//; 是否保存图片
	if (1 == sscanf(arg, "save=%d", &option))
	{
		if (option == 1)
		{
			debugSaveImages = true;
			//; 在程序中直接调用命令行，优秀！
			if (42 == system("rm -rf images_out"))
				printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if (42 == system("mkdir images_out"))
				printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if (42 == system("rm -rf images_out"))
				printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if (42 == system("mkdir images_out"))
				printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

	//; 模式：是否进行了光度校准
	if (1 == sscanf(arg, "mode=%d", &option))
	{
		mode = option;
		//; 有光度校准文件，比如TUM monoVO dataset
		if (option == 0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		//; 没有光度校准文件，那么就需要自己估计参数a和b，比如ETH EuRoC MAV dataset
		if (option == 1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0; //; 配置不进行光度矫正
			setting_affineOptModeA = 0;			//-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0;			//-1: fix. >=0: optimize (with prior, if > 0).
		}
		//; 图像就已经去掉了gamma响应、渐晕等，所以直接不包括光度校准部分。比如合成的数据集（仿真数据集）
		if (option == 2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_minGradHistAdd = 3;
		}
		return;
	}
	printf("could not parse argument \"%s\"!!!!\n", arg);
}

int main(int argc, char **argv)
{
	//setlocale(LC_ALL, "");
	// Step 1 ：读取命令行运行时的输入参数
	for (int i = 1; i < argc; i++)
		parseArgument(argv[i]);

	// hook crtl+C.
	boost::thread exThread = boost::thread(exitThread);

	// Step 2 ：读取相机参数文件，在构造函数中执行两个步骤：
	//; 1. 根据相机参数建立相机畸变模型  2.根据光度参数得到非线性仿射函数G、镜头渐晕
	ImageFolderReader *reader = new ImageFolderReader(source, calib, gammaCalib, vignette);
	//; 2.这里github上readme中作者说了，需要在初始化之前设置相机内参和视频分辨率，虽然可能不是最方便的方式
	//;  2.1. 根据上面的图像去畸变类，得到输出图像的宽、高、投影矩阵
	//;  2.2. 计算能够构成的图像金字塔，并且计算各层的宽、高、投影矩阵
	reader->setGlobalCalibration();

	//; 1.前面的setting_photometricCalibration是2，可以认为是代码要求进行gamma和渐晕矫正？
	//; 2.后面的参数表示是否有gamma矫正的文件，如果没有这个文件，你还要求使用光度矫正，那么这里就报错退出
	if (setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
	{
		printf("ERROR: dont't have photometric calibation. 
			Need to use commandline options mode=1 or mode=2 ");
		exit(1);
	}

	//; 靠，还能倒放？骚操作
	int lstart = start;
	int lend = end;
	int linc = 1; //; 正放还是倒放，增加的方向不一样
	if (reverse)
	{
		printf("REVERSE!!!!");
		lstart = end - 1;
		if (lstart >= reader->getNumImages())
			lstart = reader->getNumImages() - 1;
		lend = start;
		linc = -1;
	}

	// Step 3 new一个系统类
	FullSystem *fullSystem = new FullSystem();
	//! 设置非线性响应函数，注意其中会给类成员变量 Hcalib 赋值
	fullSystem->setGammaFunction(reader->getPhotometricGamma()); //; 设置非线性响应函数
	fullSystem->linearizeOperation = (playbackSpeed == 0);		 //; 如果=0，不强制实时执行

	IOWrap::PangolinDSOViewer *viewer = 0;
	if (!disableAllDisplay)
	{
		viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
		fullSystem->outputWrapper.push_back(viewer);
	}

	if (useSampleOutput)
		fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());

	// Step 4 运行线程
	// to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
	std::thread runthread(
		[&]()
		{
			// Step 4.1. 读取图像的时间戳，以及根据要播放的速度设置什么时候处理这些图像
			std::vector<int> idsToPlay;
			std::vector<double> timesToPlayAt;
			for (int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i += linc)
			{
				idsToPlay.push_back(i);
				if (timesToPlayAt.size() == 0)
				{
					timesToPlayAt.push_back((double)0);
				}
				else
				{
					double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
					double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
					//; 可以看到这个时间值得是程序什么时刻处理下一张图片的时间，因为最后/playbackSpeed，可以倍速播放
					timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / playbackSpeed);
				}
			}

			// Step 4.2 预先读取所有图像，这样可以在运行的时候降低读取图像的额外耗时
			std::vector<ImageAndExposure *> preloadedImages;
			//; preload在设置中默认是true
			if (preload)
			{
				printf("LOADING ALL IMAGES!\n");
				for (int ii = 0; ii < (int)idsToPlay.size(); ii++)
				{
					int i = idsToPlay[ii];
					//; 读取图像：去光度畸变和几何畸变，得到输出图像size的辐照图ImageAndExposure
					preloadedImages.push_back(reader->getImage(i));
				}
			}

			struct timeval tv_start;
			gettimeofday(&tv_start, NULL);
			clock_t started = clock();
			double sInitializerOffset = 0;

			// Step 4.3 遍历每一帧图像，开启DSO算法的跟踪过程
			for (int ii = 0; ii < (int)idsToPlay.size(); ii++)
			{
				if (!fullSystem->initialized) // if not initialized: reset start time.
				{
					gettimeofday(&tv_start, NULL);
					started = clock();					  //; 这个应该是程序运行到这里时的系统时间
					sInitializerOffset = timesToPlayAt[ii]; //; 计算的播放的每一帧相对以第一帧的播放时间偏置
				}

				int i = idsToPlay[ii]; //; i就是要处理的图像在数据集文件夹中的索引

				//; 如果上面已经提前把所有图像都读取进来了，那么直接从存储的变量中拿出来一张图像即可
				ImageAndExposure *img;
				if (preload)
					img = preloadedImages[ii];
				else
					img = reader->getImage(i); //; 注意getImage函数里面会进行光度校准

				//; 判断是否要跳过这一帧，应该是抽帧之类的操作？
				bool skipFrame = false;
				// 播放速度不为0，也就是要强制倍速执行
				if (playbackSpeed != 0)
				{
					struct timeval tv_now;
					gettimeofday(&tv_now, NULL);
					double sSinceStart = sInitializerOffset +
										((tv_now.tv_sec - tv_start.tv_sec) + (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));

					if (sSinceStart < timesToPlayAt[ii])
						usleep((int)((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
					else if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2))
					{
						printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
						skipFrame = true;
					}
				}

				//! 重要：DSO系统的入口函数！
				if (!skipFrame)
					fullSystem->addActiveFrame(img, i); //; i就是要处理的图像在数据集文件夹中的索引

				delete img;

				if (fullSystem->initFailed || setting_fullResetRequested)
				{
					if (ii < 250 || setting_fullResetRequested)
					{
						printf("RESETTING!\n");

						std::vector<IOWrap::Output3DWrapper *> wraps = fullSystem->outputWrapper;
						delete fullSystem;

						for (IOWrap::Output3DWrapper *ow : wraps)
							ow->reset();

						fullSystem = new FullSystem();
						fullSystem->setGammaFunction(reader->getPhotometricGamma());
						fullSystem->linearizeOperation = (playbackSpeed == 0);

						fullSystem->outputWrapper = wraps;

						setting_fullResetRequested = false;
					}
				}

				if (fullSystem->isLost)
				{
					printf("LOST!!\n");
					break;
				}
			} // 结束，整个数据集的所有图像都跟踪完毕，输出最终结果

			fullSystem->blockUntilMappingIsFinished();
			clock_t ended = clock();
			struct timeval tv_end;
			gettimeofday(&tv_end, NULL);

			fullSystem->printResult("result.txt");

			int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
			double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0]) - reader->getTimestamp(idsToPlay.back()));
			double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC);
			double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
			printf("\n======================"
					"\n%d Frames (%.1f fps)"
					"\n%.2fms per frame (single core); "
					"\n%.2fms per frame (multi core); "
					"\n%.3fx (single core); "
					"\n%.3fx (multi core); "
					"\n======================\n\n",
					numFramesProcessed, numFramesProcessed / numSecondsProcessed,
					MilliSecondsTakenSingle / numFramesProcessed,
					MilliSecondsTakenMT / (float)numFramesProcessed,
					1000 / (MilliSecondsTakenSingle / numSecondsProcessed),
					1000 / (MilliSecondsTakenMT / numSecondsProcessed));
			//fullSystem->printFrameLifetimes();
			if (setting_logStuff)
			{
				std::ofstream tmlog;
				tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
				tmlog << 1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC * reader->getNumImages()) << " "
					<< ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) / (float)reader->getNumImages() << "\n";
				tmlog.flush();
				tmlog.close();
			}
		});

	if (viewer != 0)
		viewer->run();

	runthread.join();

	for (IOWrap::Output3DWrapper *ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}

	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;

	printf("DELETE READER!\n");
	delete reader;

	printf("EXIT NOW!\n");
	return 0;
}
