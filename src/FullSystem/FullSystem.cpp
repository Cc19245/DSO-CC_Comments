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

/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"

#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

namespace dso
{
	// Hessian矩阵计数, 有点像 shared_ptr
	int FrameHessian::instanceCounter = 0;
	int PointHessian::instanceCounter = 0;
	int CalibHessian::instanceCounter = 0;


	FullSystem::FullSystem()
	{
		int retstat = 0;
		if (setting_logStuff)
		{
			//shell命令删除旧的文件夹, 创建新的
			retstat += system("rm -rf logs");
			retstat += system("mkdir logs");

			retstat += system("rm -rf mats");
			retstat += system("mkdir mats");

			// 打开读写log文件
			calibLog = new std::ofstream();
			calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
			calibLog->precision(12);

			numsLog = new std::ofstream();
			numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
			numsLog->precision(10);

			coarseTrackingLog = new std::ofstream();
			coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
			coarseTrackingLog->precision(10);

			eigenAllLog = new std::ofstream();
			eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
			eigenAllLog->precision(10);

			eigenPLog = new std::ofstream();
			eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
			eigenPLog->precision(10);

			eigenALog = new std::ofstream();
			eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
			eigenALog->precision(10);

			DiagonalLog = new std::ofstream();
			DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
			DiagonalLog->precision(10);

			variancesLog = new std::ofstream();
			variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
			variancesLog->precision(10);

			nullspacesLog = new std::ofstream();
			nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
			nullspacesLog->precision(10);
		}
		else
		{
			nullspacesLog = 0;
			variancesLog = 0;
			DiagonalLog = 0;
			eigenALog = 0;
			eigenPLog = 0;
			eigenAllLog = 0;
			numsLog = 0;
			calibLog = 0;
		}

		assert(retstat != 293847); // shell正常执行结束返回这么个值,填充8~15位bit, 有趣

		selectionMap = new float[wG[0] * hG[0]];

		coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
        coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
		pixelSelector = new PixelSelector(wG[0], hG[0]);

        //; 有意思，tracker类有两个对象
        coarseTracker = new CoarseTracker(wG[0], hG[0]);
		coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);


		statistics_lastNumOptIts = 0;
		statistics_numDroppedPoints = 0;
		statistics_numActivatedPoints = 0;
		statistics_numCreatedPoints = 0;
		statistics_numForceDroppedResBwd = 0;
		statistics_numForceDroppedResFwd = 0;
		statistics_numMargResFwd = 0;
		statistics_numMargResBwd = 0;

		lastCoarseRMSE.setConstant(100); //5维向量都=100

		currentMinActDist = 2;
		initialized = false;

        //; 后端求解的大boss
		ef = new EnergyFunctional();
		ef->red = &this->treadReduce;

		isLost = false;
		initFailed = false;

		needNewKFAfter = -1;

		linearizeOperation = true;  // 默认强制实时执行，如果有配置，会在调用构造函数之后再单独配置
		runMapping = true;

        //! 竟然没发现，这里还TM有个建图线程
		mappingThread = boost::thread(&FullSystem::mappingLoop, this); // 建图线程单开
		
        lastRefStopID = 0;

		minIdJetVisDebug = -1;
		maxIdJetVisDebug = -1;
		minIdJetVisTracker = -1;
		maxIdJetVisTracker = -1;
	}

	FullSystem::~FullSystem()
	{
		blockUntilMappingIsFinished();

		// 删除new的ofstream
		if (setting_logStuff)
		{
			calibLog->close();
			delete calibLog;
			numsLog->close();
			delete numsLog;
			coarseTrackingLog->close();
			delete coarseTrackingLog;
			//errorsLog->close(); delete errorsLog;
			eigenAllLog->close();
			delete eigenAllLog;
			eigenPLog->close();
			delete eigenPLog;
			eigenALog->close();
			delete eigenALog;
			DiagonalLog->close();
			delete DiagonalLog;
			variancesLog->close();
			delete variancesLog;
			nullspacesLog->close();
			delete nullspacesLog;
		}

		delete[] selectionMap;

		for (FrameShell *s : allFrameHistory)
			delete s;
		for (FrameHessian *fh : unmappedTrackedFrames)
			delete fh;

		delete coarseDistanceMap;
		delete coarseTracker;
		delete coarseTracker_forNewKF;
		delete coarseInitializer;
		delete pixelSelector;
		delete ef;
	}

	void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
	{
	}


	//* 设置相机响应函数
    /**
     * @brief BInv到底是响应函数G还是响应函数的逆变换G^-1？
     *  妈的，为什么要弄两个名字啊？又是B又是G的
     * 
     * @param[in] BInv 
     */
	void FullSystem::setGammaFunction(float *BInv)
	{
		if (BInv == 0)
			return;

		// copy BInv.
		//; Hcalib是相机内参Hessian，Binv是gamma响应的逆？
		memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);

		// invert.
		//! 首先注意这个求逆并不是求一个倒数那么简单，而是求一个参数表的对应关系
		for (int i = 1; i < 255; i++)
		{
			// find val, such that Binv[val] = i.
			// I dont care about speed for this, so do it the stupid way.

			for (int s = 1; s < 255; s++)
			{
				if (BInv[s] <= i && BInv[s + 1] >= i)
				{
					Hcalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
					break;
				}
			}
		}
		Hcalib.B[0] = 0;
		Hcalib.B[255] = 255;
	}


	void FullSystem::printResult(std::string file)
	{
		boost::unique_lock<boost::mutex> lock(trackMutex);
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

		std::ofstream myfile;
		myfile.open(file.c_str());
		myfile << std::setprecision(15);

		for (FrameShell *s : allFrameHistory)
		{
			if (!s->poseValid)
				continue;

			if (setting_onlyLogKFPoses && s->marginalizedAt == s->id)
				continue;

			myfile << s->timestamp << " " << s->camToWorld.translation().transpose() << " " << s->camToWorld.so3().unit_quaternion().x() << " " << s->camToWorld.so3().unit_quaternion().y() << " " << s->camToWorld.so3().unit_quaternion().z() << " " << s->camToWorld.so3().unit_quaternion().w() << "\n";
		}
		myfile.close();
	}

 
    /**  
     * @brief 这是前端跟踪的唯一函数：传入最新的一帧图像，对上一帧进行跟踪
     * 	//@ 使用确定的运动模型对新来的一帧进行跟踪, 得到位姿和光度参数
     *  对于本函数讲解的非常好的博客：https://blog.csdn.net/wubaobao1993/article/details/104022702
     * @param[in] fh  传入的当前帧图像
     * @return Vec4 
     */
	Vec4 FullSystem::trackNewCoarse(FrameHessian *fh)
	{
		assert(allFrameHistory.size() > 0);
		// set pose initialization.

		for (IOWrap::Output3DWrapper *ow : outputWrapper)
			ow->pushLiveFrame(fh);

        //; 获取参考帧的信息
		FrameHessian *lastF = coarseTracker->lastRef; // 参考帧
		AffLight aff_last_2_l = AffLight(0, 0);

		// Step 1. 设置不同的运动状态的估计
		std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
		printf("size: %d \n", lastF_2_fh_tries.size());
        //; 如果历史关键帧数量只有两帧，则设置所有的相对位姿为单位阵
        //! 疑问：为啥？？？
		if (allFrameHistory.size() == 2)
        {
            for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
			{
                lastF_2_fh_tries.push_back(SE3()); //? 这个size()不应该是0么
            }
        }	
		else
		{
			FrameShell *slast = allFrameHistory[allFrameHistory.size() - 2];	// 上一帧
			FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3]; // 大上一帧
			SE3 slast_2_sprelast;
			SE3 lastF_2_slast;
			{ // lock on global pose consistency!
				boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
                //; 注意：这里有参考帧和上一帧的区别，感觉就是关键帧和非关键帧之间的区别？
				slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;	// 上一帧和大上一帧的运动
				lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld; // 参考帧到上一帧运动
				aff_last_2_l = slast->aff_g2l;  //; 上一帧的光度值
			}
            // 当前帧到上一帧 = 上一帧和大上一帧的
			SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast. 

			// 尝试不同的运动
			// get last delta-movement.
            //; 匀速模型：当前帧到上一帧 * 上一帧到上一个关键帧
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);						 // assume constant motion.
			//; 倍速模型：当前帧到上一帧*当前帧到上一帧 * 上一帧到上一个关键帧
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast); // assume double motion (frame skipped)
			//; 半速模型
            lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast);	 // assume half motion.
			//; 零速模型：当前帧到上一帧为0，因此当前帧到上一个关键帧 = 上一帧到上一个关键帧
            lastF_2_fh_tries.push_back(lastF_2_slast);												 // assume zero motion.
			//; 不动模型(从参考帧静止)：因此当前帧到上一个关键帧的位姿就是单位阵
            lastF_2_fh_tries.push_back(SE3());														 // assume zero motion FROM KF.

			// 尝试不同的旋转变动
            //   只需尝试大量不同的初始化（所有旋转）。 最后，如果他们不工作，他们只会在最粗略的水平上进行尝试，
            //    无论如何这都是超级快的。 另外，如果我们在这里的轨道松动，所以我们真的非常想避免这种情况。
			// just try a TON of different initializations (all rotations). In the end,
			// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
			// also, if tracking rails here we loose, so we really, really want to avoid that.
            //; 下面又在匀速模型的基础上尝试了26种旋转运动，进一步扩大对运动模型的假设
			for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta++)
			{
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));					// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));					// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));					// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));					// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));					// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));					// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0))); // assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
				lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			}

			if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) // 有不和法的
			{
				lastF_2_fh_tries.clear();
				lastF_2_fh_tries.push_back(SE3());
			}
		}

		Vec3 flowVecs = Vec3(100, 100, 100);
		SE3 lastF_2_fh = SE3();
		AffLight aff_g2l = AffLight(0, 0);

		// as long as maxResForImmediateAccept is not reached, I'll continue through the options.
		// I'll keep track of the so-far best achieved residual for each level in achievedRes.
		// 把到目前为止最好的残差值作为每一层的阈值
		// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.
		// 粗层的能量值大, 也不继续优化了, 来节省时间

		Vec5 achievedRes = Vec5::Constant(NAN);
		bool haveOneGood = false;
		int tryIterations = 0;
		// Step 2 遍历所有运动状态的估计，进行跟踪
		for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
		{
			AffLight aff_g2l_this = aff_last_2_l; // 上一帧的赋值当前帧
			SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

            // Step 2.1 使用此函数对当前帧估计的位姿进行跟踪，从金字塔最高层到最低层跟踪
			bool trackingIsGood = coarseTracker->trackNewestCoarse(
                    fh, lastF_2_fh_this, aff_g2l_this,
                    pyrLevelsUsed - 1,
                    achievedRes); // in each level has to be at least as good as the last try.
			tryIterations++;

			if (i != 0)
			{
				printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					   i,
					   i, pyrLevelsUsed - 1,
					   aff_g2l_this.a, aff_g2l_this.b,
					   achievedRes[0],
					   achievedRes[1],
					   achievedRes[2],
					   achievedRes[3],
					   achievedRes[4],
					   coarseTracker->lastResiduals[0],
					   coarseTracker->lastResiduals[1],
					   coarseTracker->lastResiduals[2],
					   coarseTracker->lastResiduals[3],
					   coarseTracker->lastResiduals[4]);
			}

			// Step 2.2 如果跟踪成功, 并且0层残差比之前达到的最好的估计结果还好，则留下这次的位姿, 并且保存最好的每一层的能量值
			// do we have a new winner?
			if (trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >= achievedRes[0]))
			{
				//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
				flowVecs = coarseTracker->lastFlowIndicators;
				aff_g2l = aff_g2l_this;
				lastF_2_fh = lastF_2_fh_this;
				haveOneGood = true;  //; 有一次跟踪比较好的结果了
			}

			// take over achieved res (always).
			if (haveOneGood)
			{
				for (int i = 0; i < 5; i++)
				{
                    // take over if achievedRes is either bigger or NAN.
					if (!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])
                    {
                        //; 各层最小的能量阈值更新
                        achievedRes[i] = coarseTracker->lastResiduals[i];	// 里面保存的是各层得到的能量值
                    }	
				}
			}

			// Step 2.3 第0层残差小于阈值，则本帧跟踪成功，其他运动猜测也都不用尝试了
            // 如果当次的优化结果是上一帧结果的N倍之内，认为这就是最优的
            //; 如果本次的金字塔第0层误差水平在  **上一帧**  的误差的1.5倍之内，那么就认为本次运动假设的跟踪就是最优的，可以退出了
			if (haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
				break;
		}

        // Step 3 最后，如果上面三个步骤都能如期运行，那么我们就已经跟踪上了前一个关键帧；
        // Step    但是如果上面的步骤并没有给出一个很好的结果，那么算法将匀速假设作为最好的假设并设置为当前的位姿。
		if (!haveOneGood)
		{
			printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
			flowVecs = Vec3(0, 0, 0);
			aff_g2l = aff_last_2_l;
			lastF_2_fh = lastF_2_fh_tries[0];
		}

        //; 最后就是讲当前的误差水平更新为保存变量供之后的过程使用
		//! 把这次得到的最好值给下次用来当阈值
		lastCoarseRMSE = achievedRes;


		// Step 4 此时shell在跟踪阶段, 没人使用, 设置值
        //; 本帧跟踪成功，把本帧加入到shell中
		// no lock required, as fh is not used anywhere yet.
		fh->shell->camToTrackingRef = lastF_2_fh.inverse();
		fh->shell->trackingRef = lastF->shell;
		fh->shell->aff_g2l = aff_g2l;
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

		if (coarseTracker->firstCoarseRMSE < 0)
			coarseTracker->firstCoarseRMSE = achievedRes[0]; // 第一次跟踪的平均能量值

		if (!setting_debugout_runquiet)
			printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);

		if (setting_logStuff)
		{
			(*coarseTrackingLog) << std::setprecision(16)
								 << fh->shell->id << " "
								 << fh->shell->timestamp << " "
								 << fh->ab_exposure << " "
								 << fh->shell->camToWorld.log().transpose() << " "
								 << aff_g2l.a << " "
								 << aff_g2l.b << " "
								 << achievedRes[0] << " "
								 << tryIterations << "\n";
		}

		return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
	}


    /**
     * @brief  利用新的帧 fh 对关键帧中的ImmaturePoint进行更新
     *   做法很简单，就是遍历所有关键帧中的所有未成熟点，然后在当前帧中寻找这些未成熟点的投影匹配点，如果
     *   匹配成功，那么就可以利用本帧的投影匹配点对未成熟点的逆深度范围进行更新，也就是逆深度滤波
     * @param[in] fh 传入的新帧
     */
	void FullSystem::traceNewCoarse(FrameHessian *fh)
	{
		boost::unique_lock<boost::mutex> lock(mapMutex);

		int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, 
            trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

		Mat33f K = Mat33f::Identity();
		K(0, 0) = Hcalib.fxl();
		K(1, 1) = Hcalib.fyl();
		K(0, 2) = Hcalib.cxl();
		K(1, 2) = Hcalib.cyl();

		// 遍历关键帧
		for (FrameHessian *host : frameHessians) // go through all active frames
		{
            //; 当前帧到host关键帧的 位姿 变换关系
			SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
			Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
			Vec3f Kt = K * hostToNew.translation().cast<float>();
            //; 当前帧到host关键帧的 光度 变换关系
			Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

            //; 遍历host关键帧中所有的未成熟点进行极线搜索
			for (ImmaturePoint *ph : host->immaturePoints)
			{
                // 极线搜索，在当前帧中匹配未成熟点，如果匹配成功，则利用当前帧的观测跟新未成熟点的逆深度范围    
				ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
					trace_good++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
					trace_badcondition++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB)
					trace_oob++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
					trace_out++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED)
					trace_skip++;
				if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
					trace_uninitialized++;
				trace_total++;
			}
		}
	}


	//@ 处理挑选出来待激活的点
	void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian *> *optimized,
		std::vector<ImmaturePoint *> *toOptimize,
		int min, int max, Vec10 *stats, int tid)
	{
		ImmaturePointTemporaryResidual *tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
		for (int k = min; k < max; k++)
		{
			(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
		}
		delete[] tr;
	}


	//@ 激活未成熟点, 加入优化
    /**
     * @brief 激活滑窗中各个关键帧上的未成熟点变成地图点，加入优化中
     * 
     */
	void FullSystem::activatePointsMT()
	{
		// Step 1 阈值计算, 通过距离地图来控制提取的激活点的数目和密度
		//currentMinActDist 初值为 2
		//* 这太牛逼了.....参数
        //; 下面这些参数都是实际工程中调出来的，目的就是根据设定的点的密度值和当前帧中包含的点的个数对比
        //; 如果当前帧点个数特别多，那么我下面设定的距离肯定就要大一些，让新激活的点更少一点，从而控制总的点个数
        //; 而使用这个距离地图，就是保证了点的分布比较均匀
		if (ef->nPoints < setting_desiredPointDensity * 0.66)
			currentMinActDist -= 0.8;
		if (ef->nPoints < setting_desiredPointDensity * 0.8)
			currentMinActDist -= 0.5;
		else if (ef->nPoints < setting_desiredPointDensity * 0.9)
			currentMinActDist -= 0.2;
		else if (ef->nPoints < setting_desiredPointDensity)
			currentMinActDist -= 0.1;

		if (ef->nPoints > setting_desiredPointDensity * 1.5)
			currentMinActDist += 0.8;
		if (ef->nPoints > setting_desiredPointDensity * 1.3)
			currentMinActDist += 0.5;
		if (ef->nPoints > setting_desiredPointDensity * 1.15)
			currentMinActDist += 0.2;
		if (ef->nPoints > setting_desiredPointDensity)
			currentMinActDist += 0.1;

		if (currentMinActDist < 0)
			currentMinActDist = 0;
		if (currentMinActDist > 4)
			currentMinActDist = 4;

		if (!setting_debugout_runquiet)
        {
			printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
				   currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);
        }
        //; 取出最新的关键帧
		FrameHessian *newestHs = frameHessians.back();

		// make dist map.
		coarseDistanceMap->makeK(&Hcalib);  //; 取出相机内参，因为相机内参在被优化，所以每次都要重新拿出来内参
        // Step 2 把所有关键帧上的未成熟点投影到最新的关键帧上，生成距离地图
		coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

		//coarseTracker->debugPlotDistMap("distMap");

		std::vector<ImmaturePoint *> toOptimize;
		toOptimize.reserve(20000); // 待激活的点

		// Step 3 处理未成熟点, 激活/删除/跳过
		for (FrameHessian *host : frameHessians) // go through all active frames
		{
			if (host == newestHs)
				continue;

			SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
			// 第0层到1层
			Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
			Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

			for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1)
			{
				ImmaturePoint *ph = host->immaturePoints[i];
				ph->idxInImmaturePoints = i;

				// delete points that have never been traced successfully, or that are outlier on the last trace.
				if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
				{
					//	immature_invalid_deleted++;
					// remove point.
					delete ph;
					host->immaturePoints[i] = 0; // 指针赋零
					continue;
				}

				//* 未成熟点的激活条件
				// can activate only if this is true.
				bool canActivate = (ph->lastTraceStatus == IPS_GOOD || ph->lastTraceStatus == IPS_SKIPPED || ph->lastTraceStatus == IPS_BADCONDITION || ph->lastTraceStatus == IPS_OOB) && ph->lastTracePixelInterval < 8 && ph->quality > setting_minTraceQuality && (ph->idepth_max + ph->idepth_min) > 0;

				// if I cannot activate the point, skip it. Maybe also delete it.
				if (!canActivate)
				{
					//* 删除被边缘化帧上的, 和OOB点
					// if point will be out afterwards, delete it instead.
					if (ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
					{
						// immature_notReady_deleted++;
						delete ph;
						host->immaturePoints[i] = 0;
					}
					// immature_notReady_skipped++;
					continue;
				}

				// see if we need to activate point due to distance map.
				Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
				int u = ptp[0] / ptp[2] + 0.5f;
				int v = ptp[1] / ptp[2] + 0.5f;

				if ((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
				{
					// 距离地图 + 小数点
					float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] + (ptp[0] - floorf((float)(ptp[0])));

					if (dist >= currentMinActDist * ph->my_type) // 点越多, 距离阈值越大
					{
						coarseDistanceMap->addIntoDistFinal(u, v);
						toOptimize.push_back(ph);
					}
				}
				else
				{
					delete ph;
					host->immaturePoints[i] = 0;
				}
			}
		}

        // Step 4 优化上一步挑出来的未成熟点, 进行逆深度优化, 并得到pointhessian
		std::vector<PointHessian *> optimized;
		optimized.resize(toOptimize.size());

		if (multiThreading)
			treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

		else
			activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


		// Step 5 把PointHessian加入到能量函数, 删除收敛的未成熟点, 或不好的点
        //; 遍历上一步优化得到的未成熟点，判断是否能够生成地图点
		for (unsigned k = 0; k < toOptimize.size(); k++)
		{
			PointHessian *newpoint = optimized[k];
			ImmaturePoint *ph = toOptimize[k];

			if (newpoint != 0 && newpoint != (PointHessian *)((long)(-1)))
			{
				newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;
				newpoint->host->pointHessians.push_back(newpoint);
				ef->insertPoint(newpoint); // 能量函数中插入点

                //; 重点：把新生成的点的残差加入到后端优化的能量函数中
                //; 注意：newpoint是取出的数组optimized的某个元素，而这个数组在上面activatePointsMT_Reductor优化的 
                //;      时候函数内部就建立了这个残差了
				for (PointFrameResidual *r : newpoint->residuals)
                {
					ef->insertResidual(r); // 能量函数中插入残差
                }
				assert(newpoint->efPoint != 0);
				delete ph;
			}
			else if (newpoint == (PointHessian *)((long)(-1)) || ph->lastTraceStatus == IPS_OOB)
			{
				// bug: 原来的顺序错误
				ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
				delete ph;
			}
			else
			{
				assert(newpoint == 0 || newpoint == (PointHessian *)((long)(-1)));
			}
		}

		// Step 6 把删除的点丢掉
		for (FrameHessian *host : frameHessians)
		{
			for (int i = 0; i < (int)host->immaturePoints.size(); i++)
			{
				if (host->immaturePoints[i] == 0)
				{
					//bug 如果back的也是空的呢
					host->immaturePoints[i] = host->immaturePoints.back(); // 没有顺序要求, 直接最后一个给空的
					host->immaturePoints.pop_back();
					i--;
				}
			}
		}
	}


	void FullSystem::activatePointsOldFirst()
	{
		assert(false);
	}


	//@ 标记要移除点的状态, 边缘化or丢掉
    /**
     * @brief 
     * 
     */
	void FullSystem::flagPointsForRemoval()
	{
		assert(EFIndicesValid);

		std::vector<FrameHessian *> fhsToKeepPoints;  // 实际没用
		std::vector<FrameHessian *> fhsToMargPoints;  //; 要被边缘化掉的帧

		//if(setting_margPointVisWindow>0)
		{ 
            //bug 又是不用的一条语句
            //! 靠，确实是bug啊，这个i指向最新帧索引，还要>=最新帧个数，肯定不满足啊？？
			for (int i = ((int)frameHessians.size()) - 1; i >= 0 && i >= ((int)frameHessians.size()); i--)
			{
                //TODO 这里可以加个assert看看能不能运行到这里
            	if (!frameHessians[i]->flaggedForMarginalization)
                {
					fhsToKeepPoints.push_back(frameHessians[i]);
                }
            }
            // Step 1 遍历所有帧，之前被标记要margin的帧存入数组中。判断margin的帧在本次优化之间就判断了
			for (int i = 0; i < (int)frameHessians.size(); i++)
			{
                if (frameHessians[i]->flaggedForMarginalization)
                {
					fhsToMargPoints.push_back(frameHessians[i]);
                }
            }
		}

		//ef->setAdjointsF();
		//ef->setDeltaF(&Hcalib);
		int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

        // Step 2 遍历所有关键帧上持有的地图点，标记要边缘化/删除的点
		for (FrameHessian *host : frameHessians) // go through all active frames
		{
            //; 遍历这个关键帧持有的所有点
			for (unsigned int i = 0; i < host->pointHessians.size(); i++)
			{
				PointHessian *ph = host->pointHessians[i];
				if (ph == 0)
                {
					continue;
                }
                
                // Step 2.1 不正常的点直接丢掉：前面优化完了也没有判断点的深度是否正常，所以在这里判断
				//* 丢掉相机后面, 没有残差的点
                //; 后面残差为0的这个条件，在之前移除外点的时候就是用它判断的，怎么可能还存在？
                //TODO 我感觉确实后面这个条件不会满足了，可以进入if之后使用assert试一下
				if (ph->idepth_scaled < 0 || ph->residuals.size() == 0)
				{
					host->pointHessiansOut.push_back(ph);  //; 跟删除点一样，把点存到删除的数组里
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;  //; 标记后端能量点
					host->pointHessians[i] = 0;  //; 指针置零，但是没有把后边的指针放到置零的这里，我感觉可以放吧？
					flag_nores++;
				}
                // Step 2.2 边缘化的帧上持有的所有点，好的点则边缘化掉，不好的点则直接丢掉
				//* 把边缘化的帧上的点, 以及受影响较大的点标记为边缘化or删除
                //; OOB是Out Of Boundary，也就是点投影下来没有落到视场内，因此没有构成残差项 
                //! 7.28增：
                //; 如果这个点 是   或者 是边缘化帧上的点
                //; 从上面来看，这里fhsToKeepPoints是空的vector啊？--> 解答：实际上isOOB这个函数内部也没有使用这个变量
				//TODO: 这里看看isOOB有没有可能是true
                else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
				{
					flag_oob++;
					// Step 2.2.1 如果是一个内点, 则把残差在当前状态线性化, 并计算到零点残差
                    //; 这里并不是内点那么简单，而是根据这个点的残差个数进行判断的
					if (ph->isInlierNew())
					{
						flag_in++;
						int ngoodRes = 0;
                        //; 遍历这个点的所有残差进行线性化，因为要对这个点进行边缘化了
						for (PointFrameResidual *r : ph->residuals)
						{
							r->resetOOB();
                            //; 计算雅克比，并且传给后端优化
							r->linearize(&Hcalib);
							r->efResidual->isLinearized = false;  //; 注意这里给了false，就离谱，感觉大家全是false
							r->applyRes(true);
                            //; 这里的判断是有道理的，也就是之前是激活的残差，那么在这里就要把他线性化固定住
                            //; 如果不是激活的残差(即已经线性化过的残差)，这里就不用重复线性化了
							// 如果是激活(可参与优化)的残差, 则给fix住, 计算res_toZeroF
							if (r->efResidual->isActive())
							{
                                //! 注意：上面说的大家都是false是错的，下面这个函数里面就会把相关的点isLinearized=true
								//; 这个函数内部做的主要工作就是把这个点的当前残差值 - J*dx，恢复到线性化点的那个值
                                r->efResidual->fixLinearizationF(ef);
								ngoodRes++;
							}
						}
						//* 如果逆深度的协方差很大直接扔掉, 小的边缘化掉
                        //; hessian大于阈值，说明协方差足够小，那么就可以边缘化
						if (ph->idepth_hessian > setting_minIdepthH_marg)
						{
							flag_inin++;
							ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
							host->pointHessiansMarginalized.push_back(ph);
						}
						else
						{
                            //; 否则hessian太小，协方差太大，点不准确，就直接丢掉
							ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
							host->pointHessiansOut.push_back(ph);
						}
					}
					// Step 2.2.2 不是内点直接扔掉
					else
					{
						host->pointHessiansOut.push_back(ph);
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
					}

					host->pointHessians[i] = 0; // 把点给删除
				}
                else
                {
                    // CC : 空，这个else是自己加的，为了代码更加规范
                    //! 疑问：万一上面两个都不满足呢？进来这个分支咋办？
                    //TODO： 这里看看有没有可能会进来这个分支
                    //! 7.29增：
                    //; CC解答：其实这个分支就是什么都不干，因为上面是判断了边缘化或者丢掉的点，然后对他们进行处理。
                    //;   如果经过判断这个点不属于被边缘化的点或者被删除的点，那么它就是继续存在于滑窗中的点，所以
                    //;   不需要进行任何处理。
                }
			} //!  -----------  遍历这个关键帧持有的所有点结束  ---------------------

			//* 删除边缘化或者删除的点
            //; 靠，放在上面哪个循环里不好吗？先用一个变量把帧上所有的点的pointHessians[]的size读出来，
            //;    然后就可以遍历了啊？
            //; CC解答：感觉就是写法的原因，因为这里弹出点需要pop_back,放在上面的话尽管可以一开始就把size
            //;    读出来，但是过程中一直在pop，所以最后也不知道到底size是多少了。
            //;    但是好像既然在pop，那么每次使用size访问就正好？anyway，无伤大雅
			for (int i = 0; i < (int)host->pointHessians.size(); i++)
			{
				if (host->pointHessians[i] == 0)
				{
					host->pointHessians[i] = host->pointHessians.back();
					host->pointHessians.pop_back();
					i--;
				}
			}
		}
	}


	/**
	 * @brief 整个DSO系统的入口函数
	 * 
	 * @param[in] image   标定后的辐照度和曝光时间
	 * @param[in] id      图像对应数据集文件夹中的第几张图片	  
	 */
	void FullSystem::addActiveFrame(ImageAndExposure *image, int id)
	{
		// Step 1 track线程锁
		if (isLost)
			return;
		boost::unique_lock<boost::mutex> lock(trackMutex);

		// Step 2 创建FrameHessian和FrameShell, 并进行相应初始化, 并存储所有帧
		// =========================== add into allFrameHistory =========================
		FrameHessian *fh = new FrameHessian(); //; 包含了大部分的内容，比如H、状态
		FrameShell *shell = new FrameShell();  //; Frame的简化，位姿等变量
		//; 下面这个无用，在上面构造函数中就已经设置了
        shell->camToWorld = SE3();			   // no lock required, as fh is not used anywhere yet.
		shell->aff_g2l = AffLight(0, 0);	   //; 绝对光度系数a, b
		shell->marginalizedAt = shell->id = allFrameHistory.size();  //; shell->id是真正处理的帧的序号
		shell->timestamp = image->timestamp;
		shell->incoming_id = id; //; 这个id是图像在数据集文件夹中的序号
		fh->shell = shell;		 //; FrameHessian持有FrameShell
		//; allFrameHistory是历史上的所有帧的轨迹，也就是这次数据集跑完之后的结果
		allFrameHistory.push_back(shell); // 只把简略的shell存起来

		// Step 3 得到曝光时间, 生成金字塔, 计算金字塔各层图像的梯度
		// =========================== make Images / derivatives etc. =========================
		fh->ab_exposure = image->exposure_time;  //; ms单位的图像曝光时间
		// 计算当前图像帧各层金字塔的像素灰度值及梯度
		fh->makeImages(image->image, &Hcalib);   //; Hcalib是相机内参，但是主要还是相机的辐照参数

		// Step 4 进行初始化
		if (!initialized)
		{
			// use initializer!
			// Step 4.1 加入第一帧：主要是建立图像金字塔，为每一层金字塔提取特征点，并寻找最近邻点和父点
			if (coarseInitializer->frameID < 0) // first frame set. fh is kept by coarseInitializer.
			{
				coarseInitializer->setFirst(&Hcalib, fh);
				//! 至此，整个系统传入的第1帧图像处理结束，等待传入下一帧图像
			}
			//! 整个系统第2 3 4 5...帧图像进入，用这些图像对系统的 第1帧（注意都是对第1帧） 图像进行跟踪
			else if (coarseInitializer->trackFrame(fh, outputWrapper)) // if SNAPPED
			{
				// Step 4.2 跟踪成功, 完成初始化
				initializeFromInitializer(fh);   //; 初始化成功，把当前帧加入到系统滑窗中
				lock.unlock();

				deliverTrackedFrame(fh, true);   //; 把最新的帧作为关键帧，进行处理
			}
			//; 如果第2 3 4 5...帧跟踪第1帧没有成功，那么就把这些帧删掉。但是注意这些帧的shell还是保留的
			else
			{
				// if still initializing
				fh->shell->poseValid = false;
				delete fh;
			}
			return;
		}
		else // do front-end operation.
		{ 
			// Step 5  对新来的帧进行跟踪, 得到位姿光度, 判断跟踪状态
			// =========================== SWAP tracking reference?. =========================
			if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
			{
				// 交换参考帧和当前帧的coarseTracker
                //; 定义了两个coarseTracker对象，主要是别的线程中对coarseTracker中的内容有修改，所以为了
                //; 方便多线程操作，这里直接就在定义一个对象，然后交换对象
				boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
				CoarseTracker *tmp = coarseTracker;
				coarseTracker = coarseTracker_forNewKF;
				coarseTracker_forNewKF = tmp;
			}

			//TODO 使用旋转和位移对像素移动的作用比来判断运动状态
            //! 重要：前端跟踪函数内容全在这里面
			Vec4 tres = trackNewCoarse(fh);
			if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
			{
				printf("Initial Tracking failed: LOST!\n");
				isLost = true;
				return;
			}

			// Step 6  判断是否插入关键帧
			bool needToMakeKF = false;
            //; 如果设置了没隔几帧插入一个关键帧
			if (setting_keyframesPerSecond > 0) // 每隔多久插入关键帧
			{
				needToMakeKF = allFrameHistory.size() == 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f / setting_keyframesPerSecond;
			}
            //; 正常是这种情况，也就是根据运行状态自行判断是否要插入关键帧
			else
			{
				Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
														   coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

				// BRIGHTNESS CHECK
				needToMakeKF = allFrameHistory.size() == 1 ||
                    // 平移像素位移
					setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double)tres[1]) / (wG[0] + hG[0]) +		  
                    //TODO 旋转像素位移, 设置为0???
					setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double)tres[2]) / (wG[0] + hG[0]) +  
                    // 旋转+平移像素位移
					setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0]) + 
					// 光度变化大
                    setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 ||		
                    // 误差能量变化太大(最初的两倍)								 
					2 * coarseTracker->firstCoarseRMSE < tres[0]; 
			}

			for (IOWrap::Output3DWrapper *ow : outputWrapper)
            {
				ow->publishCamPose(fh->shell, &Hcalib);
            }

			// Step 7 把该帧发布出去
			lock.unlock();
            //; 根据当前帧是否是关键帧，把当前帧发布给建图线程
			deliverTrackedFrame(fh, needToMakeKF);
			return;
		}
	}


    /**
     * @brief 把跟踪的帧, 给到建图线程, 设置成关键帧或非关键帧
     *   本函数的参考博客：https://blog.csdn.net/xxxlinttp/article/details/90640350?spm=1001.2014.3001.5502
     * @param[in] fh  当期帧
     * @param[in] needKF  是否是关键帧的标志
     */
	void FullSystem::deliverTrackedFrame(FrameHessian *fh, bool needKF)
	{
		//! 顺序执行
		//; 这里linearizeOperation是 是否强制实时执行的标志，如果是true，那么不强制实时执行，也就是正常顺序执行即可
		if (linearizeOperation)
		{
			if (goStepByStep && lastRefStopID != coarseTracker->refFrameID)
			{
				MinimalImageF3 img(wG[0], hG[0], fh->dI);
				IOWrap::displayImage("frameToTrack", &img);
				while (true)
				{
					char k = IOWrap::waitKey(0);
					if (k == ' ')
						break;
					handleKey(k);
				}
				lastRefStopID = coarseTracker->refFrameID;
			}
			else
            {
				handleKey(IOWrap::waitKey(1));
            }

			//; 根据输入的帧是不是关键帧，对其进行不同的处理	
			if (needKF)
				makeKeyFrame(fh);
			else
				makeNonKeyFrame(fh);
		}
        //; 否则要强制实时执行
		else
		{
			boost::unique_lock<boost::mutex> lock(trackMapSyncMutex); // 跟踪和建图同步锁
			unmappedTrackedFrames.push_back(fh);
			if (needKF)
				needNewKFAfter = fh->shell->trackingRef->id;
			trackedFrameSignal.notify_all();

			while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1)
			{
				mappedFrameSignal.wait(lock); // 当没有跟踪的图像, 就一直阻塞trackMapSyncMutex, 直到notify
			}

			lock.unlock();
		}
	}


	//@ 建图线程
	void FullSystem::mappingLoop()
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

		while (runMapping)
		{
			while (unmappedTrackedFrames.size() == 0)
			{
				trackedFrameSignal.wait(lock); // 没有图像等待trackedFrameSignal唤醒
				if (!runMapping)
					return;
			}

			FrameHessian *fh = unmappedTrackedFrames.front();
			unmappedTrackedFrames.pop_front();

			// guaranteed to make a KF for the very first two tracked frames.
			if (allKeyFramesHistory.size() <= 2)
			{
				lock.unlock(); // 运行makeKeyFrame是不会影响unmappedTrackedFrames的, 所以解锁
				makeKeyFrame(fh);
				lock.lock();
				mappedFrameSignal.notify_all(); // 结束前唤醒
				continue;
			}

			if (unmappedTrackedFrames.size() > 3)
				needToKetchupMapping = true;

			if (unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
			{
				lock.unlock();
				makeNonKeyFrame(fh);
				lock.lock();

				if (needToKetchupMapping && unmappedTrackedFrames.size() > 0) // 太多了给处理掉
				{
					FrameHessian *fh = unmappedTrackedFrames.front();
					unmappedTrackedFrames.pop_front();
					{
						boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
						assert(fh->shell->trackingRef != 0);
						fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
						fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
					}
					delete fh;
				}
			}
			else
			{
				if (setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id) // 后面需要关键帧
				{
					lock.unlock();
					makeKeyFrame(fh);
					needToKetchupMapping = false;
					lock.lock();
				}
				else
				{
					lock.unlock();
					makeNonKeyFrame(fh);
					lock.lock();
				}
			}
			mappedFrameSignal.notify_all();
		}
		printf("MAPPING FINISHED!\n");
	}


	//@ 结束建图线程
	void FullSystem::blockUntilMappingIsFinished()
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		runMapping = false;
		trackedFrameSignal.notify_all();
		lock.unlock();

		mappingThread.join();
	}


	//@ 设置成非关键帧
    /**
     * @brief  当前帧被认为是非关键帧，那么该帧就用来对活动窗口中所有的关键帧中还未成熟的点进行逆深度更新。
     *    基本原理是沿着极线进行搜索ImmaturePoint::traceOn。
     * 
     * @param[in] fh  传入的当前帧
     */
	void FullSystem::makeNonKeyFrame(FrameHessian *fh)
	{
		// needs to be set by mapping thread. no lock required since we are in mapping thread.
		{
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex); // 生命周期结束后自动解锁
			assert(fh->shell->trackingRef != 0);
			// mapping时将它当前位姿取出来得到camToWorld
			fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
			// 把此时估计的位姿取出来
			fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
		}

		traceNewCoarse(fh); // 更新未成熟点(深度未收敛的点)
		delete fh;
	}


	/**
	 * @brief 传入一个FrameHessian，由它生成关键帧，也是整个后端的核心入口函数，内部的操作非常多
     *   //@ 生成关键帧, 优化, 激活点, 提取点, 边缘化关键帧
	 * @param[in] fh 
	 */
	void FullSystem::makeKeyFrame(FrameHessian *fh)
	{
		// Step 0 设置当前估计的fh的位姿, 光度参数
		// needs to be set by mapping thread
		{ // 同样取出位姿, 当前的作为最终值
			//? 为啥要从shell来设置 ???   答: 因为shell不删除, 而且参考帧还会被优化, shell是桥梁
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			assert(fh->shell->trackingRef != 0);
			fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
			//! 重点：之前一直在找最新的关键帧的线性化点是如何设置的，就是在这里！
            fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l); // 待优化值
		}

		// Step 1 和非关键帧一样，利用当前帧对前面关键帧中的未成熟点进行逆深度更新
		traceNewCoarse(fh); // 更新未成熟点(深度未收敛的点)

		boost::unique_lock<boost::mutex> lock(mapMutex); // 建图锁


		// Step 2 标记后面需要边缘化（从活动窗口踢出）的帧
		// =========================== Flag Frames to be Marginalized. =========================
		flagFramesForMarginalization(fh); // TODO 这里没用最新帧，可以改进下


		// Step 3 将当前帧加入到滑动窗口中，并计算一下该窗口中其他帧与当前帧之间的一些参数比如相对光度、距离等
		// =========================== add New Frame to Hessian Struct. =========================
		fh->idx = frameHessians.size();
		frameHessians.push_back(fh);
		fh->frameID = allKeyFramesHistory.size();
		allKeyFramesHistory.push_back(fh->shell);
		ef->insertFrame(fh, &Hcalib);   //; 这里注意，还要把当前关键帧插入到能量函数中

        //; 两个操作：
        //; 1.相对状态量计算：计算帧帧之间的FEJ线性化点的相对量，以及帧帧之间当前最新状态的相对量，
        //;   因为后面求正规方程要用相对量
        //; 2.状态增量计算：
        //;   (1)相对状态增量：当前帧帧之间相对状态 与 FEJ线性化点帧帧之间相对状态的增量；
        //;   (2)绝对状态增量：a.当前相机内参 与 相机内参先验(也可以认为是FEJ线性化点)的增量
        //;                   b.当前每帧绝对状态 与 FEJ线性化点绝对状态的增量
        //;                   c.当前每帧绝对状态 与 先验状态(实际都给了0)的增量
		setPrecalcValues(); // 每添加一个关键帧都会运行这个来设置位姿, 设置位姿线性化点


		// Step 4 遍历窗口中之前所有帧的成熟点pointHessians，构建它们和新的关键帧的点帧误差PointFrameResidual，加入到ef中；
		// =========================== add new residuals for old points =========================
		// int numFwdResAdde = 0;   // CC: 无用参数
		for (FrameHessian *fh1 : frameHessians) // go through all active frames
		{
			// 当前帧和当前帧肯定不能构成残差
            if (fh1 == fh)
				continue;
            //; 遍历之前帧所持有的所有成熟点，和最新的关键帧构成残差
			for (PointHessian *ph : fh1->pointHessians) // 全都构造之后再删除
			{
                //; 之前的帧上面的点都可以和当前帧建立残差(实际不可能，但是这里就先全部加上，后面再排除)
				PointFrameResidual *r = new PointFrameResidual(ph, fh1, fh); // 新建当前帧fh和之前帧之间的残差
				//TODO 这里是多余的，在构造函数里调用resetOOB，就把状态设置成IN了
                r->setState(ResState::IN);  //; 加上的残差一开始都认为是内点，后面再排除
				
                //! 重点：点的残差就是在这里被加入的。这样就可以保证，随着关键帧逐渐插入，之前插入的关键帧之间
                //! 已经构成了残差，所以本次插入新的关键帧之后，只需要增加前面所有帧和当前帧的残差即可
                ph->residuals.push_back(r);

                //; 将构建的残差插入到能量方程中, 会把前端的这个残差给到后端能量残差上
                ef->insertResidual(r);   

                //; 备份最新的残差，在其他地方好像也用到了，不知道干嘛的
                //TODO 还要细看这里备份的最新的残差信息有什么用
				ph->lastResiduals[1] = ph->lastResiduals[0];	// 设置上上个残差
                // 当前的设置为上一个
				ph->lastResiduals[0] = std::pair<PointFrameResidual *, ResState>(r, ResState::IN); 
				// numFwdResAdde += 1;  // CC: 无用参数
			}
		}

		// Step 6  激活所有关键帧上的部分未成熟点(构造新的残差)
        //; 因为在前面几帧都对滑窗中关键帧的未成熟点进行了观测（逆深度滤波），因此肯定有一部分未成熟点的逆深度收敛了，
        //; 可以成为地图点了，因此这里就把这些点激活，把他们从未成熟点构造成地图点
		// =========================== Activate Points (& flag for marginalization). =========================
		// 1.该函数内部把一部分未成熟点激活成地图点之后，会接着把这些点构成的残差加入到后端能量方程中，所以这里不用再显示的
        //   调用 ef->insertResidual(r) 加入能量方程了。但是注意它没有重新makeIDX，所以下面还是要调用makeIDX来重新梳理ID
        activatePointsMT();

        //; 由于上面刚激活了部分未成熟点，所以这里要重新排列ID
        //! 疑问：感觉小bug，上面insertFrame里面也调用了makeIDX，实际上有点多余，因为上面又构建了新的残差，这里还需要makeIDX
		ef->makeIDX(); // 为啥要重新设置ID呢, 是因为加新的帧了么


		// Step 7  对滑窗内的关键帧进行优化(说的轻松, 里面好多问题)
		// =========================== OPTIMIZE ALL =========================
		fh->frameEnergyTH = frameHessians.back()->frameEnergyTH; // 这两个不是一个值么???
		float rmse = optimize(setting_maxOptIterations);  //; 传入最大的优化迭代次数

		// =========================== Figure Out if INITIALIZATION FAILED =========================
		//* 所有的关键帧数小于4，认为还是初始化，此时残差太大认为初始化失败
		if (allKeyFramesHistory.size() <= 4)
		{
			if (allKeyFramesHistory.size() == 2 && rmse > 20 * benchmark_initializerSlackFactor)
			{
				printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
				initFailed = true;
			}
			if (allKeyFramesHistory.size() == 3 && rmse > 13 * benchmark_initializerSlackFactor)
			{
				printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
				initFailed = true;
			}
			if (allKeyFramesHistory.size() == 4 && rmse > 9 * benchmark_initializerSlackFactor)
			{
				printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
				initFailed = true;
			}
		}
		if (isLost)
			return; // 优化后的能量函数太大, 认为是跟丢了


		// Step 8  去除PointHessian中的外点, 把最新帧设置为参考帧
		// =========================== REMOVE OUTLIER =========================
		// Step 8.1 去除外点，内部判断使用前端的点的残差个数为0，后端删除的时候又是用后端能量的残差个数为0，有点搞不懂
        removeOutliers();
		{
            // Step 8.2 把最新帧设置为参考关键帧，为下一帧的前端跟踪使用
            //; 这里见深蓝学院PPT P17上面(这页PPT放的位置不对，当时看的就很懵逼)
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			coarseTracker_forNewKF->makeK(&Hcalib); // 更新了内参, 因此重新make
			coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians);

			coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
			coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
		}
		debugPlot("post Optimize");


		// Step 9  标记删除和边缘化的点, 并删除&边缘化
		// =========================== (Activate-)Marginalize Points =========================
		// Step 9.1 遍历边缘化帧上的点，根据其情况选择边缘化掉还是丢掉，边缘化掉的点要重新计算其在边缘化状态的残差
        flagPointsForRemoval();
        //; 上面的函数中，会边缘化或丢掉一些点，但是函数里只是设置了后端能量点的标志，并没有实际删除，
        //;  所以这里还要手动调用一下后端函数删除这些能量点
        //! 7.29增：从功能上来说，我觉得实际把这个函数放到flagPointsForRemoval函数最后更好，
        //! 并且此函数命名也不准确，命名为MarginOrDropPoints更好，即边缘化或者丢掉点。同理把下面的marginalizePointsF
        //! 函数也一并放进里面调用，实际就完成了整个点的边缘化。这样拆开反而没有逻辑。
		ef->dropPointsF();   // 扔掉drop的点
        // Step 9.2 更新零空间
        // 每次设置线性化点都会更新零空间
		getNullspaces(_
			ef->lastNullspacespose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
        // Step 9.3 真正执行对点的边缘化操作，其中就是对这个点构成的残差计算雅克比，然后再算H和b
        //   Step   然后把这个点舒尔补边缘化掉，得到的剩余的H和b累加到HM和bM上
		// 边缘化掉点, 加在HM, bM上，注意此时还没有边缘化帧，所以此时的HM和bM还是68x68维度的
		ef->marginalizePointsF();  //; 边缘化不需要的点


		// Step 10 在最新帧上提取新的未成熟点，用于下次跟踪(否则点越来越少了)
		// =========================== add new Immature points & new residuals =========================
		makeNewTraces(fh, 0);

		for (IOWrap::Output3DWrapper *ow : outputWrapper)
		{
			ow->publishGraph(ef->connectivityMap);  //; 历史帧的共视连接图
			ow->publishKeyframes(frameHessians, false, &Hcalib);
		}


        // Step 11  边缘化掉关键帧
		// =========================== Marginalize Frames =========================
		//* 边缘化一帧要删除or边缘化上面所有点
		for (unsigned int i = 0; i < frameHessians.size(); i++)
        {
			if (frameHessians[i]->flaggedForMarginalization)
			{
				marginalizeFrame(frameHessians[i]);  //; 边缘化掉关键帧
				i = 0;  // i = 0? 操什么骚操作啊？
                //! 疑问：要是frameHessians[0/1]都是margin的帧那这里就成死循环了啊？
			}
        }
		printLogLine();
		//printEigenValLine();
	}


	//@ 从初始化中提取出信息, 用于跟踪.
	/**
	 * @brief  初始化成功，设置整个系统的一些初始变量，为后面整个系统的跟踪做准备
	 *    参考博客： https://blog.csdn.net/tanyong_98/article/details/106199045?spm=1001.2014.3001.5502
	 * 
	 *  initializeFromInitializer函数是对trackFrame的结果进行一些处理，然后发送给其他的模块，类似中转站，主要作用如下：
		首先调用insertFrame()，插入第一帧到能量帧容器，并且计算相对位姿和绝对位姿的关系；
		调用setPrecalcValues()，set()计算关键帧之间的状态作为预计算值；setDeltaF()计算状态改变量 δ x \delta x δx；
		计算图像平均尺度，用iR计算，但iR的实际意义存疑；
		删除多余点，限制计算量，为后端优化做准备；
		初始化shell为下一次初始化做准备。  //CC: 感觉这个地方说的不对吧？
		————————————————
		原文链接：https://blog.csdn.net/weixin_43424002/article/details/114629354
	 * @param[in] newFrame  初始化成功的那一帧（注意整个系统的第1帧已经被保存下来了）
	 */
	void FullSystem::initializeFromInitializer(FrameHessian *newFrame)
	{
		boost::unique_lock<boost::mutex> lock(mapMutex);

		// Step 1  把第一帧设置成关键帧, 加入队列, 加入EnergyFunctional（能量函数）
		// add firstframe.
		// Step 1.1. 一些变量的初始化
		FrameHessian *firstFrame = coarseInitializer->firstFrame; // 第一帧增加进地图
		//; frameHessians是一个FrameHessian的Vector，加入Vector中
		firstFrame->idx = frameHessians.size();	  //; 在滑窗中的所有关键帧的索引
		frameHessians.push_back(firstFrame);	  // 地图内关键帧容器
		firstFrame->frameID = allKeyFramesHistory.size();	// 所有历史关键帧id
		allKeyFramesHistory.push_back(firstFrame->shell);   // 所有历史关键帧
		
		// Step 1.2. 把第1帧加入优化，即加入能量方程，其中会遍历滑窗中的所有能量帧，计算伴随矩阵
		ef->insertFrame(firstFrame, &Hcalib);   //; 当前帧和相机内参 加入能量方程
		
		// Step 1.3. 设置预计算值，包括线性化点、最新帧状态，以及最新帧状态相对线性化点的状态增量
		// 建立所有帧的目标帧，并且进行主导帧和目标帧之间相对状态的预计算
		setPrecalcValues(); // 设置相对位姿预计算值

		//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
		//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

		//; 修改第一帧的点的空间为图像点的20%，这里应该是限制计算量
		firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);			 // 20%的点数目
		firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f); // 被边缘化
		firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);			 // 丢掉的点

		// Step 2  求出平均尺度因子
		float sumID = 1e-5, numID = 1e-5;
		//; 遍历第一层的所有的点
		for (int i = 0; i < coarseInitializer->numPoints[0]; i++)
		{
			//? iR的值到底是啥
			//; 累加第一层上的点的逆深度的期望值
			sumID += coarseInitializer->points[0][i].iR; // 第0层点的中位值, 相当于
			numID++;
		}
		//; 这里sumID/numID得到图像第1层的点的逆深度的平均值，然后再求倒数(物理意义上类似深度，但是并不是深度的平均值)
		float rescaleFactor = 1 / (sumID / numID); // 求出尺度因子，注意这个是逆深度的平均值的倒数

		// randomly sub-select the points I need.
		// 目标点数 / 实际提取点数
		float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

		if (!setting_debugout_runquiet)
        {
			printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100 * keepPercentage,
				   (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0]);
        }

		// Step 3  创建PointHessian, 点加入关键帧, 加入EnergyFunctional
		//; 将第一帧的未成熟点生成PointHessian，并且设置PointHessian的相关参数，
		//;  存储到第一帧的容器pointHessians中，然后利用insertPoint()加入到后端优化中
		for (int i = 0; i < coarseInitializer->numPoints[0]; i++)
		{
			// RAND_MAX 就是 int_max，即int整数的最大值（24...多少）
			//; 哈哈，有点意思啊，keppPercentage是上面刚算出来的，如果实际点数多，那么这个比例<1
			//; 然后这里用随机数和这个比例来比较，如果>这个比例？？？代表啥呢？
			if (rand() / (float)RAND_MAX > keepPercentage)
				continue; // 如果提取的点比较少, 不执行; 提取的多, 则随机干掉

			// 取出来这个点
			Pnt *point = coarseInitializer->points[0] + i;
			// 用这个点构造未成熟的点
			ImmaturePoint *pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type, &Hcalib);

			if (!std::isfinite(pt->energyTH))
			{
				delete pt;
				continue;
			} // 点值无穷大，有问题，直接删除这个点

			// 创建ImmaturePoint就为了创建PointHessian? 是为了接口统一吧
            //; CC：是的，看下面把未成熟点丢掉了，因为PointHessian只能由ImmaturePoint构造
			pt->idepth_max = pt->idepth_min = 1;  //; 未成熟的点的逆深度最大最小值都设置成1

			//; 构造未成熟的点的时候，其构造函数内部会计算点的权重
			PointHessian *ph = new PointHessian(pt, &Hcalib);  //; 利用未成熟的点构造PointHessain，即地图点
			delete pt;  //; 把未成熟点丢掉了，其实可以看出来，未成熟点就是为了构造成熟点用的，因为PointHessian只能由ImmaturePoint构造
			
            if (!std::isfinite(ph->energyTH))
			{
				delete ph;
				continue;
			}

			// 此时的ph就是Pointhessian，就是地图点了
			//; 根据上面的尺度，对点的逆深度进行缩放（注意这个尺度没有意义，实际上缩放应该就是为了数值稳定性）
			ph->setIdepthScaled(point->iR * rescaleFactor); //? 为啥设置的是scaled之后的

			//; 一篇博客中的注释：用到FEJ中，暂时超纲
            //; 7.29: 这里就是设置逆深度的线性化点，实际上后面可以看到逆深度的线性化点没有固定，每次都是用最新的状态
			ph->setIdepthZero(ph->idepth);	 

            //; 设置这个点有深度先验，因为整个系统第一帧的位姿、逆深度都是有先验的，这个先验就是初始化计算出来的
			ph->hasDepthPrior = true;  
			ph->setPointStatus(PointHessian::ACTIVE); // 激活点

			//; 把PointHessian加入FrameHessain的点数组中
			firstFrame->pointHessians.push_back(ph);

			// 利用insertPoint()加入到后端优化中
			ef->insertPoint(ph);
		}

		// Step 4  设置第一帧和最新帧的待优化量, 参考帧
		//; 通过前面所有帧对第一帧的track以及optimization得到第一帧到第八帧的位姿：firstToNew，
        //;  并对平移部分利用尺度因子进行处理
		SE3 firstToNew = coarseInitializer->thisToNext;  //; 当前帧相对第一帧的位姿
		firstToNew.translation() /= rescaleFactor;  //; 平移部分也要缩放这个尺度

		// really no lock required, as we are initializing.
		{
			// 1.重设待优化量，这里也看出dso初始化过程是针对当前帧和第一帧进行光度残差优化。
			//   在选取位移比较大的第一帧以后，再往后运行5帧。CC：这个地方说的明显不对
			// 2.设置第一帧和第八帧的相关参数。 CC：这个感觉说的才是对的
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

            //; 整个系统第一帧肯定就是世界坐标系，这个没什么问题
			firstFrame->shell->camToWorld = SE3(); // 空的初值?
			firstFrame->shell->aff_g2l = AffLight(0, 0);
			firstFrame->shell->trackingRef = 0;
			firstFrame->shell->camToTrackingRef = SE3();
            firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(), firstFrame->shell->aff_g2l);

			newFrame->shell->camToWorld = firstToNew.inverse();
			newFrame->shell->aff_g2l = AffLight(0, 0);
			newFrame->shell->trackingRef = firstFrame->shell;
			newFrame->shell->camToTrackingRef = firstToNew.inverse();
            //! 重要：设置最新的这一帧的位姿的线性化点
            newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(), newFrame->shell->aff_g2l);
		}
		
		//; 至此初始化已经成功了，接下的操作是将第八帧作为关键帧进行处理。
		initialized = true;
		printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
	}


	//@ 提取新的像素点用来跟踪
    /**
     * @brief 在最新关键帧上提取新的特征点作为未成熟点，用于下次的跟踪优化
     * 
     * @param[in] newFrame 
     * @param[in] gtDepth 
     */
	void FullSystem::makeNewTraces(FrameHessian *newFrame, float *gtDepth)
	{
		pixelSelector->allowFast = true; //bug 没卵用
		//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
		int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap, setting_desiredImmatureDensity);

		newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
		//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
		newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
		newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

		for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
		{
			for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++)
			{
				int i = x + y * wG[0];
				if (selectionMap[i] == 0)
					continue;

				ImmaturePoint *impt = new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib);
				if (!std::isfinite(impt->energyTH))
					delete impt; // 投影得到的不是有穷数
				else
					newFrame->immaturePoints.push_back(impt);
			}
		}
		//printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());
	}


	//* 计算frameHessian的预计算值, 和状态的delta值
	//@ 设置关键帧之间的关系
    /**
     * @brief 这个预计算值应该就是线性化点的值？
     * 解答：是的，这里就是在计算由于FEJ需要保留的固定的线性化点的值
     // !7.26增：1.备份优化前的状态值作为线性化点
     *           2.计算当前状态相对线性化点状态的增量、相对大的先验状态的增量
     // !7.27增：两个操作：
        //; 1.相对状态量计算：计算帧帧之间的FEJ线性化点的相对量，以及帧帧之间当前最新状态的相对量，
        //;   因为后面求正规方程要用相对量
        //; 2.状态增量计算：
        //;   (1)相对状态增量：当前帧帧之间相对状态 与 FEJ线性化点帧帧之间相对状态的增量；
        //;   (2)绝对状态增量：a.当前相机内参 与 相机内参先验(也可以认为是FEJ线性化点)的增量
        //;                   b.当前每帧绝对状态 与 FEJ线性化点绝对状态的增量
        //;                   c.当前每帧绝对状态 与 先验状态(实际都给了0)的增量
     */
	void FullSystem::setPrecalcValues()
	{
        // Step 1 备份优化前的状态值作为线性化点，计算优化后的状态的增量
		for (FrameHessian *fh : frameHessians)
		{
			fh->targetPrecalc.resize(frameHessians.size());	  // 每个目标帧预运算容器, 大小是关键帧数

			for (unsigned int i = 0; i < frameHessians.size(); i++)		 //? 还有自己和自己的???
			{
				// set()计算窗口内，所有关键帧之间的位姿变换和光度变换参数，好像还包括自己对自己
				fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib); // 计算Host 与 target之间的变换关系
			}
		}

		// 因为在trackFrame中优化的状态是六自由度姿态​和光度仿射变换参数的相对关系，这里利用伴随矩阵计算绝对位姿和光度变换参数的变化
		// 建立相关量的微小扰动，包括：adHTdeltaF[idx]，f->delta，f->delta_prior。
        // Step 2 计算当前状态相对线性化点状态的增量、相机内参等大的先验的增量
		ef->setDeltaF(&Hcalib);  //; 后端大boss计算最新状态相对线性化点、先验点的增量
	}


	void FullSystem::printLogLine()
	{
		if (frameHessians.size() == 0)
			return;

		if (!setting_debugout_runquiet)
			printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
				   allKeyFramesHistory.back()->id,
				   statistics_lastFineTrackRMSE,
				   ef->resInA,
				   ef->resInL,
				   ef->resInM,
				   (int)statistics_numForceDroppedResFwd,
				   (int)statistics_numForceDroppedResBwd,
				   allKeyFramesHistory.back()->aff_g2l.a,
				   allKeyFramesHistory.back()->aff_g2l.b,
				   frameHessians.back()->shell->id - frameHessians.front()->shell->id,
				   (int)frameHessians.size());

		if (!setting_logStuff)
			return;

		if (numsLog != 0)
		{
			(*numsLog) << allKeyFramesHistory.back()->id << " " << statistics_lastFineTrackRMSE << " " << (int)statistics_numCreatedPoints << " " << (int)statistics_numActivatedPoints << " " << (int)statistics_numDroppedPoints << " " << (int)statistics_lastNumOptIts << " " << ef->resInA << " " << ef->resInL << " " << ef->resInM << " " << statistics_numMargResFwd << " " << statistics_numMargResBwd << " " << statistics_numForceDroppedResFwd << " " << statistics_numForceDroppedResBwd << " " << frameHessians.back()->aff_g2l().a << " " << frameHessians.back()->aff_g2l().b << " " << frameHessians.back()->shell->id - frameHessians.front()->shell->id << " " << (int)frameHessians.size() << " "
					   << "\n";
			numsLog->flush();
		}
	}


	void FullSystem::printEigenValLine()
	{
		if (!setting_logStuff)
			return;
		if (ef->lastHS.rows() < 12)
			return;

		MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
		MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
		int n = Hp.cols() / 8;
		assert(Hp.cols() % 8 == 0);

		// sub-select
		for (int i = 0; i < n; i++)
		{
			MatXX tmp6 = Hp.block(i * 8, 0, 6, n * 8);
			Hp.block(i * 6, 0, 6, n * 8) = tmp6;

			MatXX tmp2 = Ha.block(i * 8 + 6, 0, 2, n * 8);
			Ha.block(i * 2, 0, 2, n * 8) = tmp2;
		}
		for (int i = 0; i < n; i++)
		{
			MatXX tmp6 = Hp.block(0, i * 8, n * 8, 6);
			Hp.block(0, i * 6, n * 8, 6) = tmp6;

			MatXX tmp2 = Ha.block(0, i * 8 + 6, n * 8, 2);
			Ha.block(0, i * 2, n * 8, 2) = tmp2;
		}

		VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
		VecX eigenP = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
		VecX eigenA = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
		VecX diagonal = ef->lastHS.diagonal();

		std::sort(eigenvaluesAll.data(), eigenvaluesAll.data() + eigenvaluesAll.size());
		std::sort(eigenP.data(), eigenP.data() + eigenP.size());
		std::sort(eigenA.data(), eigenA.data() + eigenA.size());

		int nz = std::max(100, setting_maxFrames * 10);

		if (eigenAllLog != 0)
		{
			VecX ea = VecX::Zero(nz);
			ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
			(*eigenAllLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
			eigenAllLog->flush();
		}
		if (eigenALog != 0)
		{
			VecX ea = VecX::Zero(nz);
			ea.head(eigenA.size()) = eigenA;
			(*eigenALog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
			eigenALog->flush();
		}
		if (eigenPLog != 0)
		{
			VecX ea = VecX::Zero(nz);
			ea.head(eigenP.size()) = eigenP;
			(*eigenPLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
			eigenPLog->flush();
		}

		if (DiagonalLog != 0)
		{
			VecX ea = VecX::Zero(nz);
			ea.head(diagonal.size()) = diagonal;
			(*DiagonalLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
			DiagonalLog->flush();
		}

		if (variancesLog != 0)
		{
			VecX ea = VecX::Zero(nz);
			ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
			(*variancesLog) << allKeyFramesHistory.back()->id << " " << ea.transpose() << "\n";
			variancesLog->flush();
		}

		std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
		(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
		for (unsigned int i = 0; i < nsp.size(); i++)
			(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " ";
		(*nullspacesLog) << "\n";
		nullspacesLog->flush();
	}

	void FullSystem::printFrameLifetimes()
	{
		if (!setting_logStuff)
			return;

		boost::unique_lock<boost::mutex> lock(trackMutex);

		std::ofstream *lg = new std::ofstream();
		lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
		lg->precision(15);

		for (FrameShell *s : allFrameHistory)
		{
			(*lg) << s->id
				  << " " << s->marginalizedAt
				  << " " << s->statistics_goodResOnThis
				  << " " << s->statistics_outlierResOnThis
				  << " " << s->movedByOpt;

			(*lg) << "\n";
		}

		lg->close();
		delete lg;
	}

	void FullSystem::printEvalLine()
	{
		return;
	}
}
