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

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <cmath>

#include <algorithm>

namespace dso
{

	//@ 对残差进行线性化
	//@ 参数: [true是applyRes, 并去掉不好的残差] [false不进行固定线性化]
    /**
     * @brief 对残差进行线性化，实际上就是求雅克比
     * 
     * @param[in] fixLinearization ？？
     * @param[in] toRemove 判定为外点的那些残差，会去掉
     * @param[in] min  从哪个残差开始计算雅克比
     * @param[in] max  截止计算到哪个残差
     * @param[in] stats 
     * @param[in] tid 
     */  
	void FullSystem::linearizeAll_Reductor(bool fixLinearization, 
        std::vector<PointFrameResidual *> *toRemove, int min, int max, Vec10 *stats, int tid)
	{
        // 单线程就是从0开始到最后，遍历所有的残差计算雅克比
		for (int k = min; k < max; k++)
		{
			PointFrameResidual *r = activeResiduals[k];  // 取出当前的这个残差

            // Step 1 调用这个函数，内部真正计算雅克比，返回值就是这个残差构成的能量值
			(*stats)[0] += r->linearize(&Hcalib); // 线性化得到能量

            //; 固定线性化, 优化后执行？没太看懂
			if (fixLinearization) // 固定线性化（优化后执行）
			{
                // Step 2 调用这个函数，把上面linearize计算的雅克比传给后端EFResidual
				r->applyRes(true); // 把值给efResidual

				if (r->efResidual->isActive()) // 残差是in的
				{
                    //TODO 好像只有在构造函数里面赋值了isNew=true，其他地方都没有赋值，这里可以assert看一下
					if (r->isNew)
					{
						PointHessian *p = r->point;
						Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll * Vec3f(p->u, p->v, 1); // projected point assuming infinite depth.
						Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll * p->idepth_scaled; // projected point with real depth.
						float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) - (ptp.head<2>() / ptp[2])).norm(); // 0.01 = one pixel.

						if (relBS > p->maxRelBaseline)
                        {
							p->maxRelBaseline = relBS; // 正比于点的基线长度
                        }
						p->numGoodResiduals++;
					}
				}
				else
				{
					//* tid线程的id
					// 删除OOB, Outlier
					toRemove[tid].push_back(activeResiduals[k]); // 残差太大则移除
				}
			}
		}
	}


	//@ 把线性化结果传给能量函数efResidual, copyJacobians [true: 更新jacobian] [false: 不更新]
    /**
     * @brief 此函数就是计算H矩阵的一些中间量
     * 
     * @param[in] copyJacobians 
     * @param[in] min 
     * @param[in] max 
     * @param[in] stats 
     * @param[in] tid 
     */
	void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10 *stats, int tid)
	{
		for (int k = min; k < max; k++)
        {   
            activeResiduals[k]->applyRes(true);
        }
	}


	//@ 计算当前最新帧的能量阈值, 太玄学了
	void FullSystem::setNewFrameEnergyTH()
	{
		// collect all residuals and make decision on TH.
		allResVec.clear();
		allResVec.reserve(activeResiduals.size() * 2);
		FrameHessian *newFrame = frameHessians.back();

		for (PointFrameResidual *r : activeResiduals)
			if (r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame) // 新的帧上残差
			{
				allResVec.push_back(r->state_NewEnergyWithOutlier);
			}

		if (allResVec.size() == 0)
		{
			newFrame->frameEnergyTH = 12 * 12 * patternNum;
			return; // should never happen, but lets make sure.
		}

		int nthIdx = setting_frameEnergyTHN * allResVec.size(); // 以 setting_frameEnergyTHN 的能量为阈值

		assert(nthIdx < (int)allResVec.size());
		assert(setting_frameEnergyTHN < 1);

		std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx, allResVec.end()); // 排序
		float nthElement = sqrtf(allResVec[nthIdx]);    // 70% 的值都小于这个值

		//? 这阈值为啥这么设置
		//* 先扩大, 在乘上一个鲁棒函数? , 再算平方得到阈值
		newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
		newFrame->frameEnergyTH = 26.0f * setting_frameEnergyTHConstWeight + newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
		newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
		newFrame->frameEnergyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
	}


	//@ 对残差进行线性化, 并去掉不在图像内, 并且残差大的
    /**
     * @brief 对残差进行线性化，实际上就是求雅克比
     * 
     * @param[in] fixLinearization  是否固定线性化？？？
     * @return Vec3 
     */
	Vec3 FullSystem::linearizeAll(bool fixLinearization)
	{
		double lastEnergyP = 0;
        //; 下面这俩变量都没用到，哎，不知道为啥这么搞
		double lastEnergyR = 0;  
		double num = 0;

		std::vector<PointFrameResidual *> toRemove[NUM_THREADS];
		for (int i = 0; i < NUM_THREADS; i++)
        {
			toRemove[i].clear();
        }

		if (multiThreading)
		{
			// TODO 看多线程这个IndexThreadReduce
			treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4), 0, activeResiduals.size(), 0);
			lastEnergyP = treadReduce.stats[0];
		}
		else
		{
			Vec10 stats;
            // Step 1 遍历所有残差，对残差进行线性化，实际上就是求这些残差的雅克比
			linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(), &stats, 0);
			lastEnergyP = stats[0];
		}

        //; 计算当前最新帧的能量阈值？ 玄学，啥玩意
        //TODO：看看这个函数里面的操作对后面是否有影响
		setNewFrameEnergyTH();

		if (fixLinearization)
		{
			//* 前面线性化, apply之后更新了state_state, 如果有相同的, 就更新状态
			for (PointFrameResidual *r : activeResiduals)
			{
				PointHessian *ph = r->point;
				if (ph->lastResiduals[0].first == r)
					ph->lastResiduals[0].second = r->state_state;
				else if (ph->lastResiduals[1].first == r)
					ph->lastResiduals[1].second = r->state_state;
			}

			// residual创建时候都创建, 再去掉不好的
			int nResRemoved = 0;
			for (int i = 0; i < NUM_THREADS; i++) // 线程数
			{
				for (PointFrameResidual *r : toRemove[i])
				{
					PointHessian *ph = r->point;
					// 删除不好的lastResiduals
					if (ph->lastResiduals[0].first == r)
						ph->lastResiduals[0].first = 0;
					else if (ph->lastResiduals[1].first == r)
						ph->lastResiduals[1].first = 0;

					for (unsigned int k = 0; k < ph->residuals.size(); k++)
                    {
						if (ph->residuals[k] == r)
						{
							ef->dropResidual(r->efResidual);
							deleteOut<PointFrameResidual>(ph->residuals, k); // residuals删除第k个
							nResRemoved++;
							break;
						}
                    }
				}
			}
			//printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved, (int)activeResiduals.size());
		}

		return Vec3(lastEnergyP, lastEnergyR, num); // 后面两个变量都没用
	}


	// applies step to linearization point.
	//@ 更新各个状态, 并且判断是否可以停止优化
    /**
     * @brief 更新状态变量，并通过步长判断是否已经接近最优解，如果是则可以停止本次优化了
     * 
     * @param[in] stepfacC  这些参数都是各个变量更新的步长，实际上调用的时候传入的都是1
     * @param[in] stepfacT 
     * @param[in] stepfacR 
     * @param[in] stepfacA 
     * @param[in] stepfacD 
     * @return true 
     * @return false 
     */
	bool FullSystem::doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD)
	{
		//* 相当于步长了
		Vec10 pstepfac;
		pstepfac.segment<3>(0).setConstant(stepfacT);
		pstepfac.segment<3>(3).setConstant(stepfacR);
		pstepfac.segment<4>(6).setConstant(stepfacA);

		float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

		float sumNID = 0;

        //; 动量更新方式，实际没用
		if (setting_solverMode & SOLVER_MOMENTUM)
		{
			Hcalib.setValue(Hcalib.value_backup + Hcalib.step); // 内参的值进行update
			for (FrameHessian *fh : frameHessians)
			{
				Vec10 step = fh->step;
				step.head<6>() += 0.5f * (fh->step_backup.head<6>()); //? 为什么加一半  答：这种解法很奇怪。。不管了

				fh->setState(fh->state_backup + step); // 位姿 光度 update
				sumA += step[6] * step[6];			   // 光度增量平方
				sumB += step[7] * step[7];
				sumT += step.segment<3>(0).squaredNorm(); // 平移增量
				sumR += step.segment<3>(3).squaredNorm(); // 旋转增量

				for (PointHessian *ph : fh->pointHessians)
				{
					float step = ph->step + 0.5f * (ph->step_backup); //? 为啥加一半
					ph->setIdepth(ph->idepth_backup + step);
					sumID += step * step;				// 逆深度增量平方
					sumNID += fabsf(ph->idepth_backup); // 逆深度求和
					numID++;

					//* 逆深度没有使用FEJ
					ph->setIdepthZero(ph->idepth_backup + step);
				}
			}
		}
		//; 普通更新状态
        else
		{ 
            // Step 1 相机内参更新：在原状态基础上 + 本次更新步长
            //* 相机内参更新状态
			Hcalib.setValue(Hcalib.value_backup + stepfacC * Hcalib.step);

            // Step 2 滑窗中相机状态(位姿、光度)、点的逆深度更新
            //* 相机位姿, 光度参数更新
			for (FrameHessian *fh : frameHessians)
			{
                //; 1.相机状态更新，传入的值就是本次更新后，相对于线性化点的状态增量
				fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));

				sumA += fh->step[6] * fh->step[6];
				sumB += fh->step[7] * fh->step[7];
				sumT += fh->step.segment<3>(0).squaredNorm();
				sumR += fh->step.segment<3>(3).squaredNorm();

                //; 2.遍历这个帧上的所有点，对逆深度进行更新
				//* 点的逆深度更新, 注意点逆深度没使用FEJ
				for (PointHessian *ph : fh->pointHessians)
				{
                    //; 设置这次更新后的逆深度
					ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
					sumID += ph->step * ph->step;
					sumNID += fabsf(ph->idepth_backup);
					numID++;

                    //; 设置点的线性化点，同样也是最新的状态，也就是说点的逆深度部分并没有使用FEJ
					ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);
				}
			}
		}

		sumA /= frameHessians.size();
		sumB /= frameHessians.size();
		sumR /= frameHessians.size();
		sumT /= frameHessians.size();
		sumID /= numID;
		sumNID /= numID;

		if (!setting_debugout_runquiet)
        {
			printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
				   sqrtf(sumA) / (0.0005 * setting_thOptIterations),
				   sqrtf(sumB) / (0.00005 * setting_thOptIterations),
				   sqrtf(sumR) / (0.00005 * setting_thOptIterations),
				   sqrtf(sumT) * sumNID / (0.00005 * setting_thOptIterations));
        }

        // Step 3 置位相对状态标志位：因为此时更新了最新状态，但是还没有重新计算最新状态和线性化点状态的差值
		EFDeltaValid = false;

        // Step 4 紧接着上面，最新状态更新了，那么重新计算最新状态和线性化点状态的差值
		setPrecalcValues(); // 更新相对位姿, 光度

		// 步长小于阈值则可以停止了
		return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
			   sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
			   sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
			   sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
	}


	// sets linearization point.
	//@ 对帧, 点, 内参的step和state进行备份
	void FullSystem::backupState(bool backupLastStep)
	{
		if (setting_solverMode & SOLVER_MOMENTUM) //TODO 是否备份 step 有啥区别
		{
			if (backupLastStep) // 不是第0步
			{
				Hcalib.step_backup = Hcalib.step;
				Hcalib.value_backup = Hcalib.value;
				for (FrameHessian *fh : frameHessians)
				{
					fh->step_backup = fh->step;
					fh->state_backup = fh->get_state();
					for (PointHessian *ph : fh->pointHessians)
					{
						ph->idepth_backup = ph->idepth;
						ph->step_backup = ph->step;
					}
				}
			}
			else // 迭代前初始化
			{
				Hcalib.step_backup.setZero();
				Hcalib.value_backup = Hcalib.value;
				for (FrameHessian *fh : frameHessians)
				{
					fh->step_backup.setZero();
					fh->state_backup = fh->get_state();
					for (PointHessian *ph : fh->pointHessians)
					{
						ph->idepth_backup = ph->idepth;
						ph->step_backup = 0;
					}
				}
			}
		}
		else
		{
			Hcalib.value_backup = Hcalib.value;
			for (FrameHessian *fh : frameHessians)
			{
				fh->state_backup = fh->get_state();
				for (PointHessian *ph : fh->pointHessians)
					ph->idepth_backup = ph->idepth;
			}
		}
	}

	//@ 恢复为原来的值
	// sets linearization point.
	void FullSystem::loadSateBackup()
	{
		Hcalib.setValue(Hcalib.value_backup);
		for (FrameHessian *fh : frameHessians)
		{
			fh->setState(fh->state_backup);
			for (PointHessian *ph : fh->pointHessians)
			{
				ph->setIdepth(ph->idepth_backup);

				ph->setIdepthZero(ph->idepth_backup); // 没用FEJ
			}
		}

		EFDeltaValid = false;
		setPrecalcValues(); // 更新当前的状态
	}

	//@ 计算能量, 计算的是绝对的能量
	double FullSystem::calcMEnergy()
	{
		if (setting_forceAceptStep)
			return 0;
		return ef->calcMEnergyF();
	}

	void FullSystem::printOptRes(const Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b)
	{
		printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
			   res[0],
			   sqrtf((float)(res[0] / (patternNum * ef->resInA))),
			   ef->resInA,
			   ef->resInM,
			   a,
			   b);
	}

	//@ 对当前的关键帧进行GN优化
    /**
     * @brief 重要：整个后端优化的入口函数
     *   参考博客：https://blog.csdn.net/wubaobao1993/article/details/104343866
     * 
     * @param[in] mnumOptIts 
     * @return float 
     */
	float FullSystem::optimize(int mnumOptIts)
	{
		if (frameHessians.size() < 2)
			return 0;
        //; 这里如果滑窗中关键帧个数过少的话，那么不关外部设置优化多少次，这里都多优化几次
		if (frameHessians.size() < 3)
			mnumOptIts = 20; // 迭代次数
		if (frameHessians.size() < 4)
			mnumOptIts = 15;

		// get statistics and active residuals.
		// Step 1 找出未线性化(边缘化)的残差, 加入activeResiduals
        //; 可以认为activeResiduals是新的残差边，是没有经过边缘化的，所以没有线性化点
		activeResiduals.clear();
		int numPoints = 0;   // 没用
		int numLRes = 0;  
        // 遍历所有关键帧
		for (FrameHessian *fh : frameHessians)
        {
            // 遍历这个关键帧上所host的所有成熟点
			for (PointHessian *ph : fh->pointHessians)
			{
                //; 这个地图点可以和其他很多的target帧上的点构成残差
				for (PointFrameResidual *r : ph->residuals)
				{
                    //; 只要没有线性化过，那么就把它加入到新的残差中
                    //! 重要：这里的if一直满足
                    // 公益群讲的这里是一直满足的。因为线性化isLinearized这个操作是每次滑窗优化之后进行边缘化的时候进行的，
                    // 而被margin掉的点就是直接抛弃掉了，它的残差肯定也没了，所以剩下的这些残差一定都不是isLinearized的
                    //! 7.27增：
                    //; 1.首先滑窗中老帧和最新的关键帧构造的残差，在构造函数里就设置了不是线性化的，所以这里一定满足
                    //TODO：2.看边缘化掉的点是否直接被丢掉了？也就是上面公益群说的
					if (!r->efResidual->isLinearized) // 没有求线性误差
					{
						activeResiduals.push_back(r); // 新加入的残差
						r->resetOOB();	 // residual状态重置
					}
					else
                    {
                        //? CC: 可以在这里添加打印语句，看是否会进入这个分支，判断上面的isLinearized是否一直满足
                        numLRes++; //已经线性化过得计数
                    }
				}
				// numPoints++;  // CC: 没用注释掉
			}
        }

		if (!setting_debugout_runquiet)
        {
            printf("OPTIMIZE %d pts, %d active res, %d lin res!\n", ef->nPoints, (int)activeResiduals.size(), numLRes);
        }
		
		// Step 2 线性化activeResiduals的残差, 计算边缘化的能量值 (然而这里都设成0了)
        //; 这里所说的线性化实际上就是求残差对状态变量的雅克比，而并不是边缘化等操作
        //! 重要：看下面的注释，从某篇博客中摘出来的
        //    当前新关键帧的线性化点认为是track时候的位姿，就是第1步的位姿;
        //    其他关键帧的位姿就是FEJ的线性化点的位姿，都为worldToCam_evalPT
		//* 线性化, 参数: [true是进行固定线性化, 并去掉不好的残差] [false不进行固定线性化]
        //! 疑问：什么叫固定线性化、不固定线性化？
        //; 目前来看这里的操作就是计算了每个残差的雅克比，存到了前端的残差类成员变量中，但是并没有传给后端优化的类
		Vec3 lastEnergy = linearizeAll(false);

		// Step 3 计算 大的先验的能量 和 上次边缘化先验的能量（其实对求解没有那么重要）
        //; 存疑：这个到底算的是先验的能量还是线性化点的能量？
        //; 解答：先验的能量，别看后面注释瞎扯淡
		double lastEnergyL = calcLEnergy(); // islinearized的量的能量
        //; 上次状态边缘化在这次的状态下产生的能量
		double lastEnergyM = calcMEnergy(); // HM部分的能量

		// 把线性化的结果给efresidual
		if (multiThreading)
        {
            treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
        }
		else
        {
            // 所谓输出的中间量就是例如：残差对位姿的偏导的转置* 残差对逆深度的偏导等这类的东西。
            // 最后会在solveSystem函数里用到这些中间变量，其实所谓中间变量就是雅克比矩阵的一些组成元素。
            //; 注意上面linearizeAll函数内部计算了残差的雅克比，但是由于传入的实参是false，因此内部并没有
            //; 调用applyRes把计算的雅克比传给后端优化，所以这里就调用applyRes_Reductor(内部调用applyRes)
            //; 把这些残差都传递给后端优化的对象。注意这里就是把残差传给后端，并没有重新线性化一次浪费时间
            //! 疑问：上面传入false的原因是什么？
            applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);
        }   
			
		if (!setting_debugout_runquiet)
		{
			printf("Initial Error       \t");
			printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
		}
		debugPlotTracking();

		// Step 3 正式进入优化，迭代求解
		double lambda = 1e-1;
		float stepsize = 1;
		VecX previousX = VecX::Constant(CPARS + 8 * frameHessians.size(), NAN);
        //; 遍历迭代次数，不断迭代
		for (int iteration = 0; iteration < mnumOptIts; iteration++)
		{
			// Step 3.1 备份当前的各个状态值，因为有可能这次优化效果不好，后面还要回退到上次的状态
			// solve!
			backupState(iteration != 0);

			// Step 3.2 求解系统
            //! 重点：内部求解系统
            //; 此函数执行完毕后，就得到滑窗内所有帧的位姿+光度、相机内参、所有点的逆深度的增量
			solveSystem(iteration, lambda);

			double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / 
                (1e-20 + previousX.norm() * ef->lastX.norm()); // 两次下降方向的点积（dot/模长）
			previousX = ef->lastX;

			// TUM自己的解法???
            //; 牛皮，还有动量法，只不过配置文件中最后没用这里
			if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM))
			{
				float newStepsize = exp(incDirChange * 1.4);
				if (incDirChange < 0 && stepsize > 1)
					stepsize = 1;

				stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
				if (stepsize > 2)
					stepsize = 2;
				if (stepsize < 0.25)
					stepsize = 0.25;
			}

			// Step 3.3 更新状态，更新之后里面会接着判断最新状态和线性化点之间的状态差值
			//* 更新变量, 判断是否停止
            //; stepsize最后传入都是1，不用管
			bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

            // Step 3.4 使用更新后的状态重新计算能量
			// eval new energy!
			//* 更新后重新计算
            //; 更新状态后重新计算所有的能量，注意这里面还是会计算最新状态的雅克比，尽管此时不确定是否会接受这次状态
            //TODO 感觉可以改进的地方：这里不确定是否接受这次状态更新，但是也算了雅克比，万一不接受，就有点浪费了
			Vec3 newEnergy = linearizeAll(false);  // 里面计算能量的部分其实只和当前状态有关
			double newEnergyL = calcLEnergy();  // 计算先验能量，只和当前状态 - 先验状态有关
			double newEnergyM = calcMEnergy();  // 计算上次边缘化先验 在当前状态的附加能量，也是只和当前状态有关

			if (!setting_debugout_runquiet)
			{
				printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
					   (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
						lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM)
						   ? "ACCEPT"
						   : "REJECT",
					   iteration,
					   log10(lambda),
					   incDirChange,
					   stepsize);
				printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
			}

			// Step 3.5 判断是否接受这次计算
			if (setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
										   lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
			{
				// 接受更新后的量
				if (multiThreading)
                {
                    treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 
                        0, activeResiduals.size(), 50);
                }
				else
                {
                    //; 因为上面的linearizeAll已经计算了最新状态的雅克比，所以如果接受这次状态更新，那么直接调用
                    //; applyres这个函数把计算的最新的雅克比传给后端优化即可
                    applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);
                }
					
				lastEnergy = newEnergy;
				lastEnergyL = newEnergyL;
				lastEnergyM = newEnergyM;

				lambda *= 0.25; // 固定lambda
			}
			else
			{ // 不接受, roll back
				loadSateBackup(); //; 不接受本次更新，那么把所有状态量都恢复到之前的状态
				lastEnergy = linearizeAll(false);  //; 再次重新计算雅克比，感觉确实不太好，万一不接受白浪费时间算雅克比
				lastEnergyL = calcLEnergy();
				lastEnergyM = calcMEnergy();
				lambda *= 1e2;
			}

            // Step 3.6 不管上面是否接受本次更新，只要更新量足够小或者到了最大次数，那么一定要跳出循环了
			if (canbreak && iteration >= setting_minOptIterations)
				break;
		} //! ------------------------ 迭代优化完成 --------------------------------

		// Step 4 把最新帧的位姿设为线性化点，也就是设置最新帧的FEJ点
		//* 最新一帧的位姿设为线性化点, 0-5是位姿增量因此是0, 6-7是值, 直接赋值
		Vec10 newStateZero = Vec10::Zero();
		newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);
		frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam,
										newStateZero); // 最新帧设置为线性化点, 待估计量
		EFDeltaValid = false;
		EFAdjointsValid = false;
		ef->setAdjointsF(&Hcalib); // 重新计算adj
        //; 重新计算滑窗中当前状态相对线性化点的状态增量，注意在这里调用的时候，最新帧的增量就是0，因为上面
        //; 刚刚设置线性化点。而其他帧的增量一般就不是0了，因为上面我们对这些老帧进行了优化
		setPrecalcValues();	  // 更新增量

		// 更新之后的能量
        // Step 5 再次计算能量，注意此时传入true，会多出一些操作：
        //; 1.首先还是基本的计算雅克比，同时计算能量残差，也就是返回值lastEnergy
        //! 疑问：又计算雅克比，有必要吗？看看后面会不会用到
        //; 2.传入true带来两个新的操作：
        //;   (1)会把这次计算的雅克比传递给后端优化
        //;   (2)会判断要剔除的外点
		lastEnergy = linearizeAll(true);

		//* 能量函数太大, 投影的不好, 跟丢
		if (!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2]))
		{
			printf("KF Tracking failed: LOST!\n");
			isLost = true;
		}

		statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));

		if (calibLog != 0)
		{
			(*calibLog) << Hcalib.value_scaled.transpose() << " " << frameHessians.back()->get_state_scaled().transpose() << " " << sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))) << " " << ef->resInM << "\n";
			calibLog->flush();
		}

		// Step 5 把优化的结果, 给每个帧的shell, 注意这里其他帧的线性点是不更新的
		//* 把优化结果给shell
		{
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			for (FrameHessian *fh : frameHessians)
			{
                //; PRE_camToWorld就是根据 当前状态相对线性化点的状态增量 * 线性化点状态 得到的最新的状态值
				fh->shell->camToWorld = fh->PRE_camToWorld;  
				fh->shell->aff_g2l = fh->aff_g2l();
			}
		}

		debugPlotTracking();

		//* 返回平均误差rmse
		return sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));
	}


	//@ 求解系统
    /**
     * @brief 求解后端优化系统，这部分只是单纯的求解Hx=b，也是整个后端的核心数学部分
     * 
     * @param[in] iteration 
     * @param[in] lambda 
     */
	void FullSystem::solveSystem(int iteration, double lambda)
	{
        // 1.求解之前先计算零空间，目的是为了把后面计算出来的增量x向零空间投影，从而把零空间上的分量减去
		ef->lastNullspaces_forLogging = getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);

        // 2.求解系统
		ef->solveSystemF(iteration, lambda, &Hcalib);
	}

	//@ 计算能量E (chi2), 是相对的
	double FullSystem::calcLEnergy()
	{
		if (setting_forceAceptStep)
			return 0;
        //; 求先验的能量，注意这里是调用了后端来求的能量
		double Ef = ef->calcLEnergyF_MT();
		return Ef;
	}


	//@ 去除外点(残差数目变为0的)
    /**
     * @brief 前端筛选PointHessian没有构成残差的那些点，自然就是外点了。
     *  把他们的标志设置为丢弃，然后调用后端的能量函数，在后端能量函数求解中根据前端设置
     *  的标志位真正丢弃。
     * 
     */
	void FullSystem::removeOutliers()
	{
		int numPointsDropped = 0;
		for (FrameHessian *fh : frameHessians)
		{
			for (unsigned int i = 0; i < fh->pointHessians.size(); i++)
			{
				PointHessian *ph = fh->pointHessians[i];
				if (ph == 0)
                {
					continue;
                }
                //! 疑问：这里判断外点的条件是前端的点构成的残差为0，而下面dropPointsF在后端删除点的时候
                //!     其删除的也是这个能量点构成的所有残差，这样不是重复了吗？
				if (ph->residuals.size() == 0) // 如果该点的残差数为0, 则丢掉
				{
					fh->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
					fh->pointHessians[i] = fh->pointHessians.back();
					fh->pointHessians.pop_back();
					i--;
					numPointsDropped++;
				}
			}
		}
		ef->dropPointsF();
	}

	//@ 得到各个状态的零空间
	std::vector<VecX> FullSystem::getNullspaces(
		std::vector<VecX> &nullspaces_pose,
		std::vector<VecX> &nullspaces_scale,
		std::vector<VecX> &nullspaces_affA,
		std::vector<VecX> &nullspaces_affB)
	{
		nullspaces_pose.clear();  // size: 6; vec: 4+8*n
		nullspaces_scale.clear(); // size: 1;
		nullspaces_affA.clear();  // size: 1
		nullspaces_affB.clear();  // size: 1

		int n = CPARS + frameHessians.size() * 8;
		std::vector<VecX> nullspaces_x0_pre; // 所有的零空间

		//* 位姿的零空间
		for (int i = 0; i < 6; i++) // 第i个变量的零空间
		{
			VecX nullspace_x0(n);
			nullspace_x0.setZero();
			for (FrameHessian *fh : frameHessians)
			{
				nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(i);
				nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE; // 去掉scale
				nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
			}
			nullspaces_x0_pre.push_back(nullspace_x0);
			nullspaces_pose.push_back(nullspace_x0);
		}
		//* 光度参数a b的零空间
		for (int i = 0; i < 2; i++)
		{
			VecX nullspace_x0(n);
			nullspace_x0.setZero();
			for (FrameHessian *fh : frameHessians)
			{
				nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) = fh->nullspaces_affine.col(i).head<2>(); //? 这个head<2>是为什么
				nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE;
				nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
			}
			nullspaces_x0_pre.push_back(nullspace_x0);
			if (i == 0)
				nullspaces_affA.push_back(nullspace_x0);
			if (i == 1)
				nullspaces_affB.push_back(nullspace_x0);
		}

		//* 尺度零空间
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for (FrameHessian *fh : frameHessians)
		{
			nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;
			nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
			nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		nullspaces_scale.push_back(nullspace_x0);

		return nullspaces_x0_pre;
	}

}
