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
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/CoarseTracker.h"

namespace dso
{
    /**
     * @brief 对于关键帧的边缘化策略 1.活跃点只剩下5%的;  2.和最新关键帧曝光变化大于0.7;  3.距离最远的关键帧
     * 
     * @param[in] newFH 传入的当前最新的帧，也是即将成为关键帧的那一帧
     */
	void FullSystem::flagFramesForMarginalization(FrameHessian *newFH)
	{
		//? 怎么会有这种情况呢?
        //; 在setting中设置的默认值带入应该是1 > 7，所以这个是不会成立的
		if (setting_minFrameAge > setting_maxFrames)
		{
			for (int i = setting_maxFrames; i < (int)frameHessians.size(); i++)
			{
				FrameHessian *fh = frameHessians[i - setting_maxFrames]; // setting_maxFrames个之前的都边缘化掉
				fh->flaggedForMarginalization = true;
			}
			return;
		}

		int flagged = 0; // 标记为边缘化的个数
		// marginalize all frames that have not enough points.
        //; 遍历所有的关键帧，判断是否要边缘化掉该帧
		for (int i = 0; i < (int)frameHessians.size(); i++)
		{
			FrameHessian *fh = frameHessians[i];
            //; 地图点 + 未成熟的点
			int in = fh->pointHessians.size() + fh->immaturePoints.size();				  // 还在的点
			int out = fh->pointHessiansMarginalized.size() + fh->pointHessiansOut.size(); // 边缘化和丢掉的点

            //; 当前帧到最新的关键帧的广度变换
			Vec2 refToFh = AffLight::fromToVecExposure(frameHessians.back()->ab_exposure, fh->ab_exposure,
													   frameHessians.back()->aff_g2l(), fh->aff_g2l());

			//? 这一帧里留下来的点(地图点+未成熟点，也就是出去边缘化点+删除的点)少或者曝光时间差的大, 
            //   并且边缘化掉这帧之后滑窗中还能保持最小5帧的大小，那么就可以把这帧边缘化掉
			if ((in < setting_minPointsRemaining * (in + out) || fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow) 
                && ((int)frameHessians.size()) - flagged > setting_minFrames)
			{
				//printf("MARGINALIZE frame %d, as only %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
				//		fh->frameID, in, in+out,
				//		(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
				//		(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
				//		visInLast, outInLast,
				//		fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
				fh->flaggedForMarginalization = true;
				flagged++;
			}
			else
			{
				//printf("May Keep frame %d, as %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
				//		fh->frameID, in, in+out,
				//		(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
				//		(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
				//		visInLast, outInLast,
				//		fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
			}
		}

		// marginalize one.
        //; 如果统计完上边边缘化掉的帧之后，滑窗中还有>=7帧的关键帧，那么再利用空间结构再边缘化掉几帧
		if ((int)frameHessians.size() - flagged >= setting_maxFrames)
		{
			double smallestScore = 1;
			FrameHessian *toMarginalize = 0;
			FrameHessian *latest = frameHessians.back();

			for (FrameHessian *fh : frameHessians)
			{
				//* 至少是setting_minFrameAge个之前的帧 (保留了当前帧)
				if (fh->frameID > latest->frameID - setting_minFrameAge || fh->frameID == 0)
					continue;
				//if(fh==frameHessians.front() == 0) continue;

				double distScore = 0;
				for (FrameFramePrecalc &ffh : fh->targetPrecalc)
				{
					if (ffh.target->frameID > latest->frameID - setting_minFrameAge + 1 || ffh.target == ffh.host)
						continue;
					distScore += 1 / (1e-5 + ffh.distanceLL); // 帧间距离
				}
				//* 有负号, 与最新帧距离占所有目标帧最大的被边缘化掉, 离得最远的,
				// 论文有提到, 启发式的良好的3D空间分布, 关键帧更接近
				distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);

				if (distScore < smallestScore)
				{
					smallestScore = distScore;
					toMarginalize = fh;
				}
			}

			// printf("MARGINALIZE frame %d, as it is the closest (score %.2f)!\n",
			//		toMarginalize->frameID, smallestScore);
			toMarginalize->flaggedForMarginalization = true;
			flagged++;
		}
		//	printf("FRAMES LEFT: ");
		//	for(FrameHessian* fh : frameHessians)
		//		printf("%d ", fh->frameID);
		//	printf("\n");
	}


	//@ 边缘化一个关键帧, 删除该帧上的残差
    /**
     * @brief 边缘化掉一个关键帧
     * 
     * @param[in] frame 
     */
	void FullSystem::marginalizeFrame(FrameHessian *frame)
	{
		// marginalize or remove all this frames points.
		assert((int)frame->pointHessians.size() == 0);

        // Step 1 本质还是调用后端的边缘化函数，来对H进行舒尔补
		ef->marginalizeFrame(frame->efFrame);

		// drop all observations of existing points in that frame.
		// Step 2 删除其它帧在被边缘化帧上的残差
		for (FrameHessian *fh : frameHessians)
		{
            //; 删除其他帧在边缘化帧的值，所以这里要是边缘化帧就要跳过
			if (fh == frame)
				continue;

            //; 遍历其他帧上的所有点，点上的所有残差
			for (PointHessian *ph : fh->pointHessians)
			{
				for (unsigned int i = 0; i < ph->residuals.size(); i++)
				{
					PointFrameResidual *r = ph->residuals[i];
					//; 其他帧有以边缘化帧为target帧的残差的化，就要把残差删掉
                    if (r->target == frame)
					{
						if (ph->lastResiduals[0].first == r)
							ph->lastResiduals[0].first = 0;
						else if (ph->lastResiduals[1].first == r)
							ph->lastResiduals[1].first = 0;

						if (r->host->frameID < r->target->frameID)
							statistics_numForceDroppedResFwd++;
						else
							statistics_numForceDroppedResBwd++;
                        //; 前后端都删除这个点的残差
						ef->dropResidual(r->efResidual);
						deleteOut<PointFrameResidual>(ph->residuals, i);
                        //; 这个break加不加都行，加了更快，因为某一帧上某个点的所有残差，只有可能有一个是和边缘化帧构成的
						break;  
					}
				}
			}
		}

		{
			std::vector<FrameHessian *> v;
			v.push_back(frame);
			for (IOWrap::Output3DWrapper *ow : outputWrapper)
            {
				ow->publishKeyframes(v, true, &Hcalib);
            }
		}

        //; 告诉前端shell, 边缘化的信息。这是啥信息？
		frame->shell->marginalizedAt = frameHessians.back()->shell->id;
		frame->shell->movedByOpt = frame->w2c_leftEps().norm();

        //; 从关键帧数组中弹出这个关键帧，但是注意这里并没有delete释放内存
		deleteOutOrder<FrameHessian>(frameHessians, frame);

        //; 给关键帧数组中的关键帧重新赋值索引
		for (unsigned int i = 0; i < frameHessians.size(); i++)
        {
			frameHessians[i]->idx = i;
        }
        
        // Step 3 重新设置预计算值和伴随值，因为删掉了一个关键帧
		setPrecalcValues();
		ef->setAdjointsF(&Hcalib);
	}

}
