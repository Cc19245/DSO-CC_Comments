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

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{
	CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0, 0), thisToNext(SE3())
	{
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			points[lvl] = 0;
			numPoints[lvl] = 0;
		}

		JbBuffer = new Vec10f[ww * hh];
		JbBuffer_new = new Vec10f[ww * hh];

		frameID = -1;
		fixAffine = true;
		printDebug = false;

		//! 这是
		wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
		wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
		wM.diagonal()[6] = SCALE_A;
		wM.diagonal()[7] = SCALE_B;
	}
	CoarseInitializer::~CoarseInitializer()
	{
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			if (points[lvl] != 0)
				delete[] points[lvl];
		}

		delete[] JbBuffer;
		delete[] JbBuffer_new;
	}

	/**
	 * @brief 目的是优化两帧（ref frame 和 new frame）之间的相对状态和ref frame中所有点的逆深度
	 *        注意下面所有的操作都是为了上面的这个目的！
	 *     原文链接：https://blog.csdn.net/gbz3300255/article/details/109379330  
	 * @param[in] newFrameHessian  新传入的图像帧
	 * @param[in] wraps 显示对象
	 * @return true 
	 * @return false 
	 */
	/*
	1.对于除第一帧外的后帧，trackFrame(fh, outputWrapper)，直接法（two frame direct image alignment）
	  只利用第一帧与当前帧的数据，用高斯牛顿方法基于最小化光测误差，求解或优化参数，优化之前，变换矩阵初始化为
	  单位阵、点的逆深度初始化为1，在这个过程中，优化的初值都是没有实际意义的，优化的结果也是很不准确的。
    2.这个函数只有初始化用。
	  初始化过程中最小化光度误差目的是确定第一帧每一个点的逆深度idepth（Jacobian 对应代码中的变量 dd）、
	  第一帧和第二帧的相对位姿。两帧之间的参数用LM方法不断优化解高斯牛顿方程获得，参数包括：第一帧上特征点深度值，
	  第一帧到当前帧的位姿变换，仿射参数。
	  变换一共N(特征点个数) + 8(位姿+ 仿射系数)个解
	*/
	bool CoarseInitializer::trackFrame(FrameHessian *newFrameHessian, std::vector<IOWrap::Output3DWrapper *> &wraps)
	{
		newFrame = newFrameHessian;  //; 赋值给类成员变量中的新帧

		// Step 1 先显示新来的帧
		// 新的一帧, 在跟踪之前显示的
		for (IOWrap::Output3DWrapper *ow : wraps)
			ow->pushLiveFrame(newFrameHessian);

		int maxIterations[] = {5, 5, 10, 30, 50};

		//? 调参
		// 这个是位移的阈值，如果平移的总偏移量超过2.5 / 150 就认为此帧是snapped为true的帧了.阈值怎么来的很蹊跷
		alphaK = 2.5 * 2.5; //*freeDebugParam1*freeDebugParam1;
		alphaW = 150 * 150; //*freeDebugParam2*freeDebugParam2;
		// 近邻点对当前点逆深度的影响权重
		regWeight = 0.8;	//*freeDebugParam4;
		couplingWeight = 1; //*freeDebugParam5;

		// Step 2 初始化每个点逆深度为1, 初始化光度参数, 位姿SE3
		// 对points点中的几个数据初始化过程, 只要出现过足够大的位移后 就不再对其初始化，直接拿着里面的值去连续优化5次
		if (!snapped) //! snapped应该指的是位移足够大了，不够大就重新优化
		{
			//; 将两帧相对位姿平移部分置0
			thisToNext.translation().setZero();
			//; 遍历图像金字塔，为第1帧选择的特征点的相关信息设置初值
			for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
			{
				int npts = numPoints[lvl];
				Pnt *ptsl = points[lvl];
				for (int i = 0; i < npts; i++)
				{
					ptsl[i].iR = 1;
					ptsl[i].idepth_new = 1;
					ptsl[i].lastHessian = 0;
				}
			}
		}

		// 设置两帧之间的相对位姿变换以及由曝光时间设置两帧的光度放射变换
		SE3 refToNew_current = thisToNext;
		AffLight refToNew_aff_current = thisToNext_aff;

		//firstFrame 是第一帧图像数据信息
		//如果无光度标定那么曝光时间ab_exposure就是1.那么下面这个就是 a= 0 b = 0
		if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
			refToNew_aff_current = AffLight(logf(newFrame->ab_exposure / firstFrame->ab_exposure), 0); // coarse approximation.

		Vec3f latestRes = Vec3f::Zero();
		
		// Step 3 金字塔跟踪模型（重点）：对金字塔每层进行跟踪，由最高层开始，构建残差进行优化
		// 从顶层开始估计
		for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--)
		{
			//[ ***step 3*** ] 使用计算过的上一层来初始化下一层
			//; 1.如果不是金字塔的最高层，那么就用上层来初始化下一层, 
			//;   利用高斯分布归一化积，利用 parent （上一层）的逆深度更新当前层的点的逆深度
			if (lvl < pyrLevelsUsed - 1)
				propagateDown(lvl + 1);

			Mat88f H, Hsc;
			Vec8f b, bsc;
			//; 2.如果是最高层，那么对最高层进行操作.
			// 这个函数只调一次，用邻居点抢救下坏点。只抢救最高层的, 其余层只用这个函数的赋值操作
			resetPoints(lvl); // 这里对顶层进行初始化!  CC：错误！没发现对顶层的操作

			// Step 4  迭代之前计算能量, 正规方程的Hessian矩阵等，注意里面一边计算雅克比，一边在计算舒尔补的结果
			//! 重要：计算当前层图像的正规方程！
			Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
			
			//; 把这一次计算出来的点的能量、逆深度等赋值给旧的对应变量，用于下一次计算
			applyStep(lvl); // 新的能量付给旧的

			float lambda = 0.1;
			float eps = 1e-4;
			int fails = 0;

			if (printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					   lvl, 0, lambda,
					   "INITIA",
					   sqrtf((float)(resOld[0] / resOld[2])), // 卡方(res*res)平均值
					   sqrtf((float)(resOld[1] / resOld[2])), // 逆深度能量平均值
					   sqrtf((float)(resOld[0] / resOld[2])),
					   sqrtf((float)(resOld[1] / resOld[2])),
					   (resOld[0] + resOld[1]) / resOld[2],
					   (resOld[0] + resOld[1]) / resOld[2],
					   0.0f);
				std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() << "\n";
			}

			// Step 5 LM迭代求解求解本次正规方程对状态变量的更新
			int iteration = 0;
			while (true)
			{
				// Step 5.1  计算边缘化后的Hessian矩阵, 以及一些骚操作
				Mat88f Hl = H;
				// 对角线元素乘了个值, 就是LM算法
				for (int i = 0; i < 8; i++)
					Hl(i, i) *= (1 + lambda); // 这不是LM么,论文说没用, 嘴硬
				// 舒尔补, 边缘化掉逆深度状态, 对应高斯牛顿方程消元后左侧δx21的系数
				Hl -= Hsc * (1 / (1 + lambda)); // 因为dd必定是对角线上的, 所以也乘倒数
				// 对应方程右的值 二者合起来就是Hl*x21 = bl  x21为未知量
				Vec8f bl = b - bsc * (1 / (1 + lambda));
				//? wM为什么这么乘, 它对应着状态的SCALE
				//? (0.01f/(w[lvl]*h[lvl]))是为了减小数值, 更稳定?
				// wM是个对角矩阵 对角线原始为 1  1  1  0.5  0.5  0.5  10  1000
				//; wM这些位置对应的就是平移3、旋转3、广度2。感觉这里和VINS里面的操作是一样的，就是让不同部分的数量级
				//; 相差不要太大，从而提高求矩阵的解的时候的数值稳定性
				Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
				bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));

				// Step 5.2  求解增量
				Vec8f inc;
				if (fixAffine) // 固定光度参数
				{
					inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() * (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
					inc.tail<2>().setZero();  //; 后面光度参数更新置0
				}
				else
				{
					//; 注意这里和VINS中的也是一样的，前面对变量都*wM，这里求解的最终结果还是要*wM，而不应该是/wM,
					//; 因为前面是在方程组的系数上*wM，相当于把方程组的解缩小了wM，所以这里恢复正常大小还要*wM
					inc = -(wM * (Hl.ldlt().solve(bl))); // =-H^-1 * b. 正规方程求解公式
				}

				// Step 5.3  更新状态, doStep中更新逆深度
				SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
				AffLight refToNew_aff_new = refToNew_aff_current;
				refToNew_aff_new.a += inc[6];
				refToNew_aff_new.b += inc[7];
				doStep(lvl, lambda, inc);  //; 求解逆深度增量，因为是对角阵，所以就变成求解标量了，运算量小

				// Step 5.4  计算更新后的能量并且与旧的对比判断是否accept
				Mat88f H_new, Hsc_new;
				Vec8f b_new, bsc_new;
				//; 注意这里又调用求正规方程的函数了，但是主要目的是为了求能量值
				Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
				Vec3f regEnergy = calcEC(lvl);   //; 计算上次求解的逆深度和这次求解的逆深度相对期望值的方差
				
				//; res部分是光度能量+附加的alpha能量（就是平移部分），reg部分是深度的方差能量
				float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
				float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

				bool accept = eTotalOld > eTotalNew;  // 新求解的能量要 < 上一次的能量，才接受这一次迭代

				if (printDebug)
				{
					printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						   lvl, iteration, lambda,
						   (accept ? "ACCEPT" : "REJECT"),
						   sqrtf((float)(resOld[0] / resOld[2])),
						   sqrtf((float)(regEnergy[0] / regEnergy[2])),
						   sqrtf((float)(resOld[1] / resOld[2])),
						   sqrtf((float)(resNew[0] / resNew[2])),
						   sqrtf((float)(regEnergy[1] / regEnergy[2])),
						   sqrtf((float)(resNew[1] / resNew[2])),
						   eTotalOld / resNew[2],
						   eTotalNew / resNew[2],
						   inc.norm());
					std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() << "\n";
				}

				// Step 5.5  接受的话, 更新状态,; 不接受则增大lambda
				if (accept)
				{
					//? 这是啥   答：应该是位移足够大，才开始优化IR
					//; 这个部分在计算正规方程的函数里有，如果alpha能量足够大的话，这个时候就是平移足够大了，
					//; 此时返回的resNew[1]结果就是 alphaK * numPoints[lvl]
					if (resNew[1] == alphaK * numPoints[lvl]) // 当 alphaEnergy > alphaK*npts
						snapped = true;   //; 标记位移已经足够大了
					//; 注意这里直接更新了正规方程，因为上面求新的能量的时候就求出了新的正规方程
					H = H_new;
					b = b_new;
					Hsc = Hsc_new;
					bsc = bsc_new;
					resOld = resNew;
					refToNew_aff_current = refToNew_aff_new;
					refToNew_current = refToNew_new;
					//; 把这一次计算出来的点的能量、逆深度等赋值给旧的对应变量，用于下一次计算
					applyStep(lvl);
					//; 更新所有点的逆深度期望值iR
					optReg(lvl); // 更新iR
					lambda *= 0.5;
					fails = 0;
					if (lambda < 0.0001)
						lambda = 0.0001;
				}
				else
				{
					fails++;
					lambda *= 4;   //; 就是LM算法，这个对lambda的跟新算法和LM是一样的
					if (lambda > 10000)
						lambda = 10000;
				}

				bool quitOpt = false;
				// 迭代停止条件, 收敛/大于最大次数/失败2次以上
				if (!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
				{
					//; 定义这个局部变量干啥？也没用到
					Mat88f H, Hsc;   
					Vec8f b, bsc;

					quitOpt = true;
				}

				if (quitOpt)
					break;
				iteration++;
			}

			//; 更新这一层得到的总的能量值
			latestRes = resOld;
		}

		// Step 6  优化后赋值位姿, 利用高斯分布归一化积 从底层计算上层点的深度
		thisToNext = refToNew_current;
		thisToNext_aff = refToNew_aff_current;

		// 整个都结束了 用低层的逆深度点对高层的逆深度做了次更新
		for (int i = 0; i < pyrLevelsUsed - 1; i++)
			propagateUp(i);

		frameID++;
		// 只要有一次snapped为true了 以后就一直true了
		if (!snapped)
			snappedAt = 0;

		if (snapped && snappedAt == 0)
			snappedAt = frameID; // 第一次出现位移够大的时刻的帧号 

		debugPlot(0, wraps);

		// 位移足够大, 再优化5帧才行
		// 然后在第一次出现位移够大后连续再优化它5帧 最终结果作为输出
		return snapped && frameID > snappedAt + 5;
	}

	void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper *> &wraps)
	{
		bool needCall = false;
		for (IOWrap::Output3DWrapper *ow : wraps)
			needCall = needCall || ow->needPushDepthImage();
		if (!needCall)
			return;

		int wl = w[lvl], hl = h[lvl];
		Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];

		MinimalImageB3 iRImg(wl, hl);

		for (int i = 0; i < wl * hl; i++)
			iRImg.at(i) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);

		int npts = numPoints[lvl];

		float nid = 0, sid = 0;
		for (int i = 0; i < npts; i++)
		{
			Pnt *point = points[lvl] + i;
			if (point->isGood)
			{
				nid++;
				sid += point->iR;
			}
		}
		float fac = nid / sid;

		for (int i = 0; i < npts; i++)
		{
			Pnt *point = points[lvl] + i;

			if (!point->isGood)
				iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0));

			else
				iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR * fac));
		}

		//IOWrap::displayImage("idepth-R", &iRImg, false);
		for (IOWrap::Output3DWrapper *ow : wraps)
			ow->pushDepthImage(&iRImg);
	}


	// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
	/**
	 * @brief 重点：计算能量函数和Hessian矩阵, 以及舒尔补, sc代表Schur
	 *      注意只是算出了增量方程schur消元系数 还没解增量方程呢.

	 * @param[in] lvl   金字塔层数
	 * @param[in] H_out   H  b  Hsc  bsc  对应增量方程schur消元后的几个矩阵有了他们就能
	 * @param[in] b_out    直接计算出增量方程的解了, 解就是位姿增量、仿射系数增量和逆深度增量
	 * @param[in] H_out_sc 
	 * @param[in] b_out_sc 
	 * @param[in] refToNew     新帧到参考帧的位姿变换
	 * @param[in] refToNew_aff 新帧到参考帧的affine光度仿射变换
	 * @param[in] plot   没用，估计本来想在里面写画图函数的，结果发现好像也没法画图
	 * @return Vec3f 
	 */
	Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff,
		bool plot)
	{
		int wl = w[lvl], hl = h[lvl];   // 当前金字塔层图像的宽高

		Eigen::Vector3f *colorRef = firstFrame->dIp[lvl];   // 第1帧图像的梯度
		Eigen::Vector3f *colorNew = newFrame->dIp[lvl];     // 当前帧图像的梯度

		// 旋转矩阵R * 内参矩阵K_inv，方便后面把点从像素平面投到归一化相机平面
		Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
		Vec3f t = refToNew.translation().cast<float>();	  // 平移
		Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b); // 光度参数

		// 该层的相机投影内参
		float fxl = fx[lvl];
		float fyl = fy[lvl];
		float cxl = cx[lvl];
		float cyl = cy[lvl];

		Accumulator11 E;   // 1*1 的累加器，用来存储最后总的能量值
		acc9.initialize(); // 9维向量，相乘得到9x9的hessian矩阵，用来计算H矩阵的位姿和光度部分
		E.initialize();    // 1*1累加器的初始化

		// 遍历该层金字塔图像的所有特征点	
		int npts = numPoints[lvl];
		Pnt *ptsl = points[lvl];
		// Step 1. 遍历所有点，计算两个重要内容：
		//;   1.H矩阵中的U和b_A(深蓝PPT P29)，并且利用SSE加速pattern中的8个点的计算，结果存储在acc9的9x9矩阵中
        //;   2.H矩阵中的W，b_B，V对应的每个点的雅克比部分（注意不是最后的矩阵），为后面计算acc9SC中的Hsc和bsc做准备
		for (int i = 0; i < npts; i++)
		{
			Pnt *point = ptsl + i;  // 取出当前特征点

			point->maxstep = 1e10;   //; 这个是逆深度更新时的最大增量值
			//; 如果这个点不好，那么就用它上次的能量值累加进去，但是并不计算这个点的雅克比
			if (!point->isGood) // 点不好
			{
				E.updateSingle((float)(point->energy[0])); // 累加
				point->energy_new = point->energy;
				point->isGood_new = false;
				continue;   //; 不考虑这个点，直接遍历下一个点
			}
			//; 8*1向量, 每个点附近的残差个数为8个, 因为是取一个pattern范围内的8个点
			VecNRf dp0; 
			VecNRf dp1;
			VecNRf dp2;
			VecNRf dp3;
			VecNRf dp4;
			VecNRf dp5;
			VecNRf dp6;
			VecNRf dp7;
			VecNRf dd;
			VecNRf r;
			//; 10*1 向量，存储每个点对应于H矩阵的舒尔消元部分的雅克比
			JbBuffer_new[i].setZero(); 

			// sum over all residuals.
			bool isGood = true;
			float energy = 0;  // 这个点的pattern内计算的总能量，作为这个点的能量
			// 遍历当前这个点周围的pattern，一共是8个点
			for (int idx = 0; idx < patternNum; idx++)
			{
				// pattern的坐标偏移
				int dx = patternP[idx][0];
				int dy = patternP[idx][1];

				// Pj' = R*(X/Z, Y/Z, 1) + t/Z, 变换到新的点, 深度仍然使用Host帧的!
				//; 这里得到的就是j帧相机坐标系下的虚拟点（不是j帧归一化相机平面的点，而是除以了i帧下的深度，而不是j帧下的深度）
				Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;
				// 归一化坐标 Pj，也就是在j帧相机坐标系的归一化平面上的点
				float u = pt[0] / pt[2];
				float v = pt[1] / pt[2];
				// 像素坐标pj，j帧相机像素坐标系上的坐标
				float Ku = fxl * u + cxl;
				float Kv = fyl * v + cyl;
				// dpi/pz'，对应这个点变换到j帧相机坐标系之后的逆深度
				float new_idepth = point->idepth_new / pt[2]; // 新一帧上的逆深度

				// 落在边缘附近，深度小于0, 说明这个点不好，直接退出此次对这个点的pattern的遍历
				if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0))
				{
					isGood = false;  // 这个点不好，直接退出此次对这个点的pattern的遍历
					break;
				}
				// 插值得到新图像中的 patch 像素值，(输入3维，输出3维像素值 + x方向梯度 + y方向梯度)
				Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
				//Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

				// 参考帧上的 patch 上的像素值, 输出一维像素值
				//float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
				float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

				// 像素值有穷, good
				//! 疑问：像素值怎么可能是无穷的呢？
				if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
				{
					isGood = false;   // 这个点不好，直接退出此次对这个点的pattern的遍历
					break;
				}

				// 残差, 对应公式 I_j - e^a_ji * I_i - b_ji
				float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
				// Huber权重: e < 阈值k，则权重为1； >阈值k则权重为k/|e|。配置文件中设置为k=9
				// 论文公式4   huber范数用来降低高梯度点的权重？？感觉不是这样吧？huber和高梯度权重是俩东西
			    // https://blog.csdn.net/xxxlinttp/article/details/89379785 博客写的很清晰 对应公式9 只不过乘
			    // 了个2倍  无所谓 都乘约掉了
				//! 疑问：这个鲁棒核函数还不太懂，到底是不是根据像素梯度设置的权重那部分？
				float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
				// huberweight * (2-huberweight) = Objective Function
				// robust 权重和函数之间的关系
				energy += hw * residual * residual * (2 - hw);

				//; Pj 对 逆深度 di 求导的中间变量(注意这里只是一个中间变量，并不是最终的逆深度求导结果)
				// 1/Pz * (tx - u*tz), u = px'/pz'
				float dxdd = (t[0] - t[2] * u) / pt[2];
				// 1/Pz * (ty - v*tz), v = py'/pz'
				float dydd = (t[1] - t[2] * v) / pt[2];

				//; huber核函数的权重
				if (hw < 1)
					hw = sqrtf(hw); //?? 为啥开根号, 答: 鲁棒核函数等价于加权最小二乘
				// dxfx, dyfy
				float dxInterp = hw * hitColor[1] * fxl;
				float dyInterp = hw * hitColor[2] * fyl;
				//* 残差对 j(新状态) 位姿求导, 前三个是平移，后三个是旋转
				dp0[idx] = new_idepth * dxInterp;						// dpi/pz' * dxfx
				dp1[idx] = new_idepth * dyInterp;						// dpi/pz' * dyfy
				dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp); // -dpi/pz' * (px'/pz'*dxfx + py'/pz'*dyfy)
				dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;	// - px'py'/pz'^2*dxfy - (1+py'^2/pz'^2)*dyfy
				dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;	// (1+px'^2/pz'^2)*dxfx + px'py'/pz'^2*dxfy
				dp5[idx] = -v * dxInterp + u * dyInterp;				// -py'/pz'*dxfx + px'/pz'*dyfy
				//* 残差对光度参数求导
				dp6[idx] = -hw * r2new_aff[0] * rlR; // exp(aj-ai)*I(pi)
				dp7[idx] = -hw * 1;					 // 对 b 导
				//* 残差对 i(旧状态) 逆深度求导
				dd[idx] = dxInterp * dxdd + dyInterp * dydd; // dxfx * 1/Pz * (tx - u*tz) +　dyfy * 1/Pz * (tx - u*tz)
				r[idx] = hw * residual;						 // 残差 res

				//* 像素误差对逆深度的导数，取模倒数
				//! 疑问：这是啥玩意?
				//! 解答：point->maxstep是逆深度更新的最大增量值，但是这里为啥这么设置就不太懂了
				//; 新增一个博客中的解释：maxstep可以理解为移动单位像素，深度的变化
				float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm(); //? 为什么这么设置
				if (maxstep < point->maxstep)
					point->maxstep = maxstep;

				// immediately compute dp*dd' and dd*dd' in JbBuffer1.
				//; 注意这里就是在算H矩阵的舒尔消元部分的雅克比，并不是H矩阵。这里的相加是因为现在在遍历第i个点的pattern，
				//; 一个点的pattern范围内一共是8个点，所以相加。这里的索引i就是当前金字塔层上的所有点。
				// 1.深蓝学院PPT P31，H矩阵中的W部分，其中的J0——J7，注意仍然是雅克比，不是最后的H矩阵结果
				JbBuffer_new[i][0] += dp0[idx] * dd[idx];
				JbBuffer_new[i][1] += dp1[idx] * dd[idx];
				JbBuffer_new[i][2] += dp2[idx] * dd[idx];
				JbBuffer_new[i][3] += dp3[idx] * dd[idx];
				JbBuffer_new[i][4] += dp4[idx] * dd[idx];
				JbBuffer_new[i][5] += dp5[idx] * dd[idx];
				JbBuffer_new[i][6] += dp6[idx] * dd[idx];
				JbBuffer_new[i][7] += dp7[idx] * dd[idx];
				// 2.深蓝学院PPT P31，H矩阵中的b_B部分，其中的J8，注意仍然是雅克比，不是最后的H矩阵结果
				JbBuffer_new[i][8] += r[idx] * dd[idx];
				// 3.深蓝学院PPT P31，H矩阵中的V部分，其中的w，注意仍然是雅克比，不是最后的H矩阵结果
				JbBuffer_new[i][9] += dd[idx] * dd[idx];
			}  // 结束八个点的累计了

			// energy是 8个点的残差值和异常点或者越界或者能量异常
			// 如果点的pattern(其中一个像素)超出图像,像素值无穷, 或者残差大于阈值
			if (!isGood || energy > point->outlierTH * 20)
			{
				//对应论文Outlier and Occlusion Detection 这节描述内容剔除异常点
				E.updateSingle((float)(point->energy[0])); // 上一帧的加进来
				point->isGood_new = false;   // 异常点会将isGood_new的值修改一次设置为false
				point->energy_new = point->energy; //上一次的给当前次的
				continue;
			}

			// 内点则加进能量函数
			// add into energy.
			// SSEData[0] 累加式赋值
			E.updateSingle(energy);   //; E 是一个1x1的累加器，这里存储总的能量
			point->isGood_new = true;
			point->energy_new[0] = energy;

			//! 因为使用128位相当于每次加4个数（一个float 32位，4个float正好128位）, 因此i+=4, 妙啊!
			// update Hessian matrix.  更新hessian矩阵
			//; 注意patternNum=8，i = 0/4正好索引两次，把pattern中8个点每次算4个点，算2次
			for (int i = 0; i + 3 < patternNum; i += 4)
			{
				//; 这里简单理解SSE加速：dp0-7和r都是8维向量，每个维度都对应pattern中的一个点，这些变量中的每个维度组合起来
				//; 得到一个9x1的向量（因为这里有9个变量），然后因为这些向量都是8维的（对应pattern中8个点），所以最后相当
				//; 于我有8个9x1的向量，要计算9x1乘以1x9向量得到的9x9矩阵，得到的8个9x9矩阵再相加（为什么相加？因为当前点
				//; 要算他周围pattern的8个点的和作为当前点的结果）。这里SSE加速就是每次我都可以算4个9x9矩阵的结果，也就是
				//; 每次算pattern中的4个点，这样pattern中的8个点我只需要计算2次即可，而不需要计算8次。
				acc9.updateSSE(
					_mm_load_ps(((float *)(&dp0)) + i),
					_mm_load_ps(((float *)(&dp1)) + i),
					_mm_load_ps(((float *)(&dp2)) + i),
					_mm_load_ps(((float *)(&dp3)) + i),
					_mm_load_ps(((float *)(&dp4)) + i),
					_mm_load_ps(((float *)(&dp5)) + i),
					_mm_load_ps(((float *)(&dp6)) + i),
					_mm_load_ps(((float *)(&dp7)) + i),
					_mm_load_ps(((float *)(&r)) + i));
			}
			// 加0, 4, 8后面多余的值, 因为SSE2是以128为单位相加, 多余的单独加
			// 不是四倍数会最后再来一次 就是要加速计算
			//; 对于DSO正常操作来说，这里并不会执行了，因为DSO用的pattern上面2次就能算完了
			for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
			{
				acc9.updateSingle(
					(float)dp0[i], (float)dp1[i], (float)dp2[i], (float)dp3[i],
					(float)dp4[i], (float)dp5[i], (float)dp6[i], (float)dp7[i],
					(float)r[i]);
			}
		} // 点循环结束.

		// E中存总残差值算的是E中A值
		E.finish();  //; 这里调用finish，应该就是把存在SSE中的结果处理一下，存到E.A中，也就是最后输出的结果
		acc9.finish();  //; 这里简单理解就是把上面SSE计算的存储在特定数组中的中间结果，更新到最后我们要的9x9的H矩阵中

	
		// Step 2 计算附加的alpha能量，不太懂具体原理，但是基本和PPT能对应上
		// calculate alpha energy, and decide if we cap it.	
		// 计算α能量，并决定是否限制它。
		Accumulator11 EAlpha;
		EAlpha.initialize();
		// 一顿计算E的点是为了干啥呢，这段代码完全无用啊笔误还是什么呢
		//; CC: 这段代码应该没啥问题吧，就是在计算逆深度需要是1的正则化项
		for (int i = 0; i < npts; i++)
		{
			Pnt *point = ptsl + i;
			//; 如果是坏点，那么正则化项就用之前算出来的正则化项
			if (!point->isGood_new) // 点不好用之前的
			{
				E.updateSingle((float)(point->energy[1])); //! 又是故意这样写的，没用的代码？？ CC：这个有用啊，没啥毛病吧
			}
			//; 如果是好点，那么就添加逆深度均值为1的正则化项
			else
			{
				// 最开始初始化都是成1
				point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1); //? 什么原理?
				E.updateSingle((float)(point->energy_new[1]));
			}
		}
		// 最终结果是算一个EAlpha的A值。但是EAlpha只定义了下初始化了一下
		// 没有运算过程，所以A只恒为0
		EAlpha.finish();	
		//squaredNorm 二范数的平方		
		//! 疑问：这里对逆深度均值为1的限制到底用没用？没用的话上面对E又计算了。所以最后结果就是它用在了E中，但是
		//!  没有乘以前面了alphaW，也没有参与后面的hessain矩阵计算（即没有对变量变化提供帮助？）
		float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts); // 平移越大, 越容易初始化成功?

		//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);

		//! 疑问：下面这个操作实在没看懂，不知道它怎么操作的
		// compute alpha opt.计算alpha选择。
		float alphaOpt;  //; 这是对后面逆深度的梯度的放大系数
		// 150 * 150 * 平移分量的二范数平方×点总数 > 2.5 * 2.5 ×点总数
		// 等价于150 * 平移分量的二范数> 2.5 ( 因为EAlpha.A恒为0)
		if (alphaEnergy > alphaK * npts) // 平移大于一定值
		{
			alphaOpt = 0;   //; 这里对逆深度的矩阵为1的限制就没有了
			alphaEnergy = alphaK * npts;  //; 限制alpha能量为2.5*总点数
		}
		else
		{
			alphaOpt = alphaW;  // 150*150
		}

		// Step 3 计算acc9SC，即H矩阵中舒尔消元的部分
		acc9SC.initialize();
		for (int i = 0; i < npts; i++)
		{
			Pnt *point = ptsl + i;
			if (!point->isGood_new)
				continue;

			point->lastHessian_new = JbBuffer_new[i][9]; // 对逆深度 dd*dd

			// Step 3.1. 添加附加能量函数的雅克比部分
			//? 这又是啥??? 对逆深度的值进行加权? 深度值归一化?
			// 前面Energe加上了（d-1)*(d-1), 所以dd = 1， r += (d-1)
			// 看了个博客，说这个叫正则项为了加速优化过程，使其快速收敛
			// 博客地址: https://blog.csdn.net/wubaobao1993/article/details/103871697 
			// 说是下面这个是应对小位移的正则项 
			//! 注意：这里为什么只有8和9的位置加了对逆深度均值为1的能量的雅克比？
			//; 因为8对应的是残差部分，对于一个能量函数来说（注意不是残差函数），其对应到正规方程里面，能量函数的一阶导
			//; 就是残差部分，而能量函数的二阶导就是完整的hessian矩阵（注意不是近似的hessian，但是由于我们大部分都是
			//; 用近似的hessian，所以这里附加部分用2完整的hessian也无所谓）
			// Step 3.1.1. 如果是位移较小的附加能量函数
			JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1); // r*dd
			JbBuffer_new[i][9] += alphaOpt;							  // 对逆深度导数为1 // dd*dd
			
			// Step 3.1.2. 如果是位移较大的附加能量函数
			//; 如果满足这个，说明平移足够大了，那么附加的能量函数就切换成第2个，也就是对点的深度期望值的能量函数
			if (alphaOpt == 0)
			{
				// 说是下面这个是应对大位移的正则项       
		   		// 此时的idepth_new是点的上一次优化更新的结果
				JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->iR);
				JbBuffer_new[i][9] += couplingWeight;
			}
			
			//; 注意：上面算的一直都是舒尔矩阵中的V部分，知道这里才对它求逆，变成V^-1，也就是最后的权重
			// 博客中的公式的分母多了个1  防止JbBuffer_new[i][9]过小造成系统不稳定吧
			JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]); // 取逆是协方差，做权重
		
			//; 注意这里就是简单的计算，并没有SSE加速的内容。
			//;  这里就计算完成了每一个点对H矩阵中舒尔消元部分的内容，每次计算都会累加每个点的计算结果，因为N个点
			//;  组成的N个残差，最后矩阵相乘之后和1个点组成的一个矩阵，然后一共N个矩阵相加结果是一样的（其实就是
			//;  矩阵乘法按照左边一列乘以右边一行得到一个矩阵，然后所有矩阵相加得到最后的矩阵乘法结果）
			acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0], (float)JbBuffer_new[i][1], (float)JbBuffer_new[i][2], (float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4], (float)JbBuffer_new[i][5], (float)JbBuffer_new[i][6], (float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8], (float)JbBuffer_new[i][9]);
		}
		acc9SC.finish();  //; finish里面就是根据矩阵的对称性，把对角线之下的元素补充上

		// Step 4 计算完毕，把舒尔消元的结果从SSE计算的内存中取出来
		//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
		// 对应舒尔消元矩阵的Hx21x21, Hx21x21  
		//! 下面的变量对比深蓝学院PPT的P29 
		//; H_out = U（位姿和广度的H矩阵部分，8x8）
		H_out = acc9.H.topLeftCorner<8, 8>();		// / acc9.num;  dp^T*dp
		// 对应舒尔消元矩阵的Jx21转置* r21,  (Jx21).t*r21
		//; b_out = b_A(位姿和广度的b部分，8x1)
		b_out = acc9.H.topRightCorner<8, 1>();		// / acc9.num;  dp^T*r
		// 对应舒尔消元矩阵的(Hρx21).t *  (Hρρ).-1 * (Hρx21)
		//; H_out_sc = W * V^-1 * W^T, 即把逆深度舒尔消元后得到的关于位姿和广度的H矩阵部分，8x8
		H_out_sc = acc9SC.H.topLeftCorner<8, 8>();	// / acc9.num; 	(dp*dd)^T*(dd*dd)^-1*(dd*dp)
		// 对应舒尔消元矩阵的(Hρx21).t *  (Hρρ).-1 * (Jρ).t * r21
		//; b_out_sc = W * V^-1 * b_B，即把逆深度舒尔消元后得到的关于位姿和广度的b部分，8x1
		b_out_sc = acc9SC.H.topRightCorner<8, 1>(); // / acc9.num;	(dp*dd)^T*(dd*dd)^-1*(dp^T*r)

		// Step 5 最后对平移的附加能量函数部分，对位姿和广度部分还要加上H和b的部分
		//! 真的是严谨啊！这个作者属实有点东西！
		//; 首先明确，在计算能量函数的时候，最后附加了两种能量函数：位移较小和位移较大（见深蓝PPT P28），其中两种附加能量函数都和
		//; 逆深度有关，而位移较小的能量函数还和位姿的平移部分有关。但是代码中添加这部分能量函数并且求雅克比的地方，是在计算完了
		//; 位姿和广度的H矩阵acc9之后，然后对附加能量函数的雅克比部分只在处理逆深度的舒尔补的时候添加了。所以如果是位移较小的附加
		//; 能量函数，显然和位移有关，对位姿部分的H矩阵也是有H和B的贡献的，但是上面的代码中我们并没有添加，因此这里要再添加上
		// t*t*ntps
		// 给 t 对应的Hessian, 对角线加上一个数, b也加上
		// 将位移较小情况的正则项加入
		//; H 矩阵直接是二阶导数，t被求导求掉了，变成单位矩阵
		H_out(0, 0) += alphaOpt * npts;
		H_out(1, 1) += alphaOpt * npts;
		H_out(2, 2) += alphaOpt * npts;

		//; 而B部分是一阶导数，其中含有t
		Vec3f tlog = refToNew.log().head<3>().cast<float>(); // 李代数, 平移部分 (上一次的位姿值)
		b_out[0] += tlog[0] * alphaOpt * npts;
		b_out[1] += tlog[1] * alphaOpt * npts;
		b_out[2] += tlog[2] * alphaOpt * npts;

		// 能量值, ? , 使用的点的个数
		// 总残差值，平移二范数，总点数
		return Vec3f(E.A, alphaEnergy, E.num);
	}

	float CoarseInitializer::rescale()
	{
		float factor = 20 * thisToNext.translation().norm();
		//	float factori = 1.0f/factor;
		//	float factori2 = factori*factori;
		//
		//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		//	{
		//		int npts = numPoints[lvl];
		//		Pnt* ptsl = points[lvl];
		//		for(int i=0;i<npts;i++)
		//		{
		//			ptsl[i].iR *= factor;
		//			ptsl[i].idepth_new *= factor;
		//			ptsl[i].lastHessian *= factori2;
		//		}
		//	}
		//	thisToNext.translation() *= factori;

		return factor;
	}

	//* 计算旧的和新的逆深度与iR的差值, 返回旧的差, 新的差, 数目
	/**
	 * @brief 计算上一次求解的逆深度与逆深度期望iR的方差、这一次求解的逆深度与逆深度期望值iR的方差、计算方差用的好的点数目
	 *   //? iR到底是啥呢     答：IR是逆深度的均值，尺度收敛到IR
	 * @param[in] lvl 
	 * @return Vec3f 
	 */
	Vec3f CoarseInitializer::calcEC(int lvl)
	{
		if (!snapped)
			return Vec3f(0, 0, numPoints[lvl]);
		AccumulatorX<2> E;  // 2维的累加器，计算二维向量每个维度的和
		E.initialize();
		int npts = numPoints[lvl];  // 这一层点的数量
		for (int i = 0; i < npts; i++)
		{
			Pnt *point = points[lvl] + i;  // 取出这个点
			if (!point->isGood_new)
				continue;
			//; 下面两个变量就计算上一次的深度与深度期望的差  以及  这一次的深度与深度期望的差
			float rOld = (point->idepth - point->iR);
			float rNew = (point->idepth_new - point->iR);
			E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew)); // 求和

			//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
		}
		E.finish();  //; 把累加的结果更新到累加器成员变量 A1m中

		//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
		//; 返回上一次深度的期望方差、这一次的深度期望方差、总的点数
		return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
	}

	//* 使用最近点来更新每个点的iR, smooth的感觉
	/**
	 * @brief 用当前点的逆深度 和 当前点的最近邻点的逆深度期望值iR 做一个平均，更新当前点的iR
	 * 
	 * @param[in] lvl 
	 */
	void CoarseInitializer::optReg(int lvl)
	{
		int npts = numPoints[lvl];
		Pnt *ptsl = points[lvl];

		//* 位移不足够则设置iR是1
		// Step 1 如果snapped为false，说明此时初始化的位移还不足够，那么强制逆深度的期望iR值为1
		if (!snapped)
		{
			for (int i = 0; i < npts; i++)
				ptsl[i].iR = 1;
			return;
		}

		// Step 2 运行到这里的时候snapped为true，说明初始化位移已经足够大了，那么用邻居点决定其值
		for (int i = 0; i < npts; i++)
		{
			Pnt *point = ptsl + i;
			if (!point->isGood)
				continue;

			float idnn[10];
			int nnn = 0;
			// 获得当前点周围最近10个点, 质量好的点的iR
			for (int j = 0; j < 10; j++)
			{
				if (point->neighbours[j] == -1)
					continue;
				Pnt *other = ptsl + point->neighbours[j];  //; 去除这个点的邻居点
				if (!other->isGood)
					continue;
				idnn[nnn] = other->iR;  //; 存储邻居点的逆深度期望值
				nnn++;
			}

			// 与最近点中位数进行加权获得新的iR
			if (nnn > 2)
			{
				// nth_element是求区间第k小的（划重点） 例如里面就是求nnn/2这个值呢
				//; 这个函数应该就是传入一个数组，然后排序，然后算里面第k小的值
				std::nth_element(idnn, idnn + nnn / 2, idnn + nnn); // 获得中位数
				// IR赋值位置 用了个权重来做平衡
				//; 这里类似低通滤波，用当前点的逆深度值和他邻居点的逆深度的期望值做一个平均
				//; regWeight = 0.8(玄学参数)
				point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
			}
		}
	}

	//* 使用归一化积来更新高层逆深度值
	/**
	 * @brief 利用底层，更新上一层的逆深度和iR，利用的原理是高斯分布归一化积
	 * 
	 * @param[in] srcLvl 
	 */
	void CoarseInitializer::propagateUp(int srcLvl)
	{
		assert(srcLvl + 1 < pyrLevelsUsed);
		// set idepth of target

		int nptss = numPoints[srcLvl];
		int nptst = numPoints[srcLvl + 1];
		Pnt *ptss = points[srcLvl];
		Pnt *ptst = points[srcLvl + 1];

		// set to zero.
		// Step 1  遍历高层的点，先对他们的iR清0
		for (int i = 0; i < nptst; i++)
		{
			Pnt *parent = ptst + i;
			parent->iR = 0;
			parent->iRSumNum = 0;
		}

		//* 更新在上一层的parent
		// Step 2  遍历这一层的点，寻找这个点的父点，计算高斯分布归一化积的中间变量
		for (int i = 0; i < nptss; i++)
		{
			Pnt *point = ptss + i;
			if (!point->isGood)
				continue;
			//; 根据这一层的点的父亲点的索引，在高层上找到对应的父亲点
			Pnt *parent = ptst + point->parent;
			parent->iR += point->iR * point->lastHessian; //! 均值*信息矩阵 ∑ (sigma*u)
			parent->iRSumNum += point->lastHessian;		  //! 新的信息矩阵 ∑ sigma
		}

		// Step 3  再遍历上一层的点，计算最后高斯分布归一化积的结果
		for (int i = 0; i < nptst; i++)
		{
			Pnt *parent = ptst + i;
			if (parent->iRSumNum > 0)
			{
				//; 逆深度和逆深度的均值，都是用高斯分布归一化积来更新
				parent->idepth = parent->iR = (parent->iR / parent->iRSumNum); //! 高斯归一化积后的均值
				parent->isGood = true;
			}
		}
		// Step 4  最后再对高层的点，利用其临近点进行一波逆深度和iR的更新
		optReg(srcLvl + 1); // 使用附近的点来更新IR和逆深度
	}


	//@ 使用上层信息来初始化下层
	//@ param: 当前的金字塔层+1
	//@ note: 没法初始化顶层值
	/**
	 * @brief 利用上一层，更新当前层的逆深度和iR，利用的原理是高斯分布归一化积
	 * 
	 * @param[in] srcLvl 
	 */
	void CoarseInitializer::propagateDown(int srcLvl)
	{
		assert(srcLvl > 0);
		// set idepth of target

		int nptst = numPoints[srcLvl - 1]; // 当前层的点数目
		Pnt *ptss = points[srcLvl];		   // 当前层+1, 上一层的点集
		Pnt *ptst = points[srcLvl - 1];	   // 当前层点集

		// Step 1 遍历当前层的所有点，用父点更新当前层点的逆深度和iR
		for (int i = 0; i < nptst; i++)
		{
			Pnt *point = ptst + i;				// 遍历当前层的点
			Pnt *parent = ptss + point->parent; // 找到当前点的parrent

			if (!parent->isGood || parent->lastHessian < 0.1)
				continue;
			if (!point->isGood)
			{
				// 当前点不好, 则把父点的值直接给它, 并且置位good
				point->iR = point->idepth = point->idepth_new = parent->iR;
				point->isGood = true;
				point->lastHessian = 0;
			}
			//; 如果这个点是一个好点，那么就利用父点的逆深度和当前点的逆深度，使用高斯分布归一化积，来更新当前点的逆深度
			else
			{
				// 通过hessian给point和parent加权求得新的iR
				// iR可以看做是深度的值, 使用的高斯归一化积, Hessian是信息矩阵
				float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian) / (point->lastHessian * 2 + parent->lastHessian);
				point->iR = point->idepth = point->idepth_new = newiR;
			}
		}
		// Step 2 对当前层的点的iR，考虑其邻居点，进行一个平滑
		//? 为什么在这里又更新了iR, 没有更新 idepth
		// 感觉更多的是考虑附近点的平滑效果
		optReg(srcLvl - 1); // 当前层
	}

	//* 低层计算高层, 像素值和梯度
	void CoarseInitializer::makeGradients(Eigen::Vector3f **data)
	{
		for (int lvl = 1; lvl < pyrLevelsUsed; lvl++)
		{
			int lvlm1 = lvl - 1;
			int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

			Eigen::Vector3f *dINew_l = data[lvl];
			Eigen::Vector3f *dINew_lm = data[lvlm1];
			// 使用上一层得到当前层的值
			for (int y = 0; y < hl; y++)
				for (int x = 0; x < wl; x++)
					dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x + 2 * y * wlm1][0] +
													  dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
													  dINew_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
													  dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
			// 根据像素计算梯度
			for (int idx = wl; idx < wl * (hl - 1); idx++)
			{
				dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
				dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
			}
		}
	}

	/**
	 * @brief 设置第一帧
	 * 
	 * @param[in] HCalib   相机内参的类（包括相机内参和一些优化使用的参数）？？为啥都加Hessian的后缀？
	 * @param[in] newFrameHessian  输入的图像帧
	 */
	void CoarseInitializer::setFirst(CalibHessian *HCalib, FrameHessian *newFrameHessian)
	{
		// Step 1 计算图像每层的内参
		makeK(HCalib);  //计算每层图像金字塔的内参，用于后续的金字塔跟踪模型
		firstFrame = newFrameHessian;

		// 进行像素点选取的相关类的初始实例化以及变量的初始定义
		PixelSelector sel(w[0], h[0]); // 像素选择
		//; 这两个变量中对应位置的值表示该位置的像素点是否被选取
		float *statusMap = new float[w[0] * h[0]];   //; 第0层的像素点对应在哪一层的梯度上被提取出来作为特征点，感觉用int更好啊？
		bool *statusMapB = new bool[w[0] * h[0]];
		//; 金字塔越往上像素点越少，所以点占的比例越多
		float densities[] = {0.03, 0.05, 0.15, 0.5, 1}; // 不同层取的点密度，越往上取的点密度越大
		// Step 2 遍历图像金字塔，在不同层选择特征点，并且为选择出来的特征点构造Pnt点类
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			// Step 2.1 针对不同层数选择大梯度像素, 第0层比较复杂1d, 2d, 4d大小block来选择3个层次阈值的像素
			sel.currentPotential = 3; // 设置网格大小，3*3大小格
			int npts;		// 选出的点的数量
			if (lvl == 0)	// 第0层（原始图像）提取特征像素
				npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);
			else // 其它层则选出goodpoints,很简单，就是在pot内梯度模长满足阈值，并且dx dy dx-dy dx+dy是最大值的那个点
				npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, 
					w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);

			// 如果点非空, 则释放空间, 创建新的
			if (points[lvl] != 0)
				delete[] points[lvl];
			points[lvl] = new Pnt[npts];   // 创建npts个点Pnt结构体的数组，存储提取出来的npts个点

			// set idepth map to initially 1 everywhere.
			int wl = w[lvl], hl = h[lvl]; // 每一层的图像大小
			Pnt *pl = points[lvl];		  // 每一层上的点
			int nl = 0;
			// 要留出pattern的空间, 2 border. patternPadding是配置文件参数，默认2
			// Step 2.2 在选出的像素中, 添加点信息
			for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
			{
				for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++)
				{
					//if(x==2) printf("y=%d!\n",y);
					// 如果是被选中的像素
					if ((lvl != 0 && statusMapB[x + y * wl]) || (lvl == 0 && statusMap[x + y * wl] != 0))
					{
						//assert(patternNum==9);
						// 选取的像素点相关值的初始化，nl为像素点的ID
						//; 注意：这里给点的坐标值是在它对应的那层金字塔上的像素坐标值
						pl[nl].u = x + 0.1; //? 加0.1干啥
						pl[nl].v = y + 0.1;
						//! 重要：把点的初始深度都设置成1
						pl[nl].idepth = 1; 
						pl[nl].iR = 1;   //; 所有点逆深度的期望值设置为1
						pl[nl].isGood = true;   //; 注意这里，每个点都被标记为good点
						pl[nl].energy.setZero();
						pl[nl].lastHessian = 0;   //; 点的逆深度的协方差的逆，在使用高斯分布归一化积融合深度的时候用到
						pl[nl].lastHessian_new = 0;
						pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

						Eigen::Vector3f *cpt = firstFrame->dIp[lvl] + x + y * w[lvl]; // 该像素梯度
						float sumGrad2 = 0;
						// 计算pattern内像素梯度和， patternNum=8，配置文件中的参数
						for (int idx = 0; idx < patternNum; idx++)
						{
							int dx = patternP[idx][0]; // pattern 的坐标偏移，分别是xy坐标的偏移
							int dy = patternP[idx][1];
							float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
							sumGrad2 += absgrad;
						}
						//; 这里被作者注释掉了，也就是算了半天pattern内的像素梯度和，结果最后没使用？
						// float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
						// pl[nl].outlierTH = patternNum*gth*gth;
						//! 外点的阈值与pattern的大小有关, 一个像素是12*12
						//? 这个阈值怎么确定的...
						//; 这里外点的阈值就是一个确定的值？8*12*12
						pl[nl].outlierTH = patternNum * setting_outlierTH;

						nl++;
						assert(nl <= npts);
					}
				}
			}
			numPoints[lvl] = nl; // 点的数目,  去掉了一些边界上的点(因为上面需要计算一共8个点的pattern,所以边上要留出来2个点的空闲)
		}
		delete[] statusMap;
		delete[] statusMapB;

		// Step 3 为金字塔每一层构造的特征点，计算其同层的邻居点 和 上一层的父亲点
		makeNN();

		// 设置一些变量的值，thisToNext表示当前帧到下一帧的位姿变换，
		// snapped frameID snappedAt这三个变量在后续判断是否跟踪了足够多的图像帧能够初始化时用到。
		//; 初始化的旋转是单位阵 初始化的 平移是0 0 0 
		thisToNext = SE3();   // 当前帧到下一阵的位姿变换
		snapped = false;   //; 初始化的时候位移是否足够大了
		frameID = snappedAt = 0;  //; 注意这里更新了frameID=0，这样setFrist函数只会进入一次

		for (int i = 0; i < pyrLevelsUsed; i++)
			dGrads[i].setZero();  //; 这个变量好像没用到？
	}

	/**
	 * @brief 重置点的energy, idepth_new参数
	 * 
	 * @param[in] lvl  传入的金字塔层数
	 */
	void CoarseInitializer::resetPoints(int lvl)
	{
		Pnt *pts = points[lvl];
		int npts = numPoints[lvl];
		for (int i = 0; i < npts; i++)
		{
			// 重置
			pts[i].energy.setZero();  //; 实际在第1帧初始化的时候这里能量就设置成0了
			pts[i].idepth_new = pts[i].idepth;

			// 如果是最顶层, 则使用周围点平均值来重置
			//! 错误：这里第1帧的点都被设置成好点了，所以第2帧图像来的时候并不会执行这个
			//; 如果是金字塔最高层并且点为非good点 用邻居对其抢救一次
			if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood)
			{
				float snd = 0, sn = 0;
				for (int n = 0; n < 10; n++)
				{
					if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood)
						continue;
					snd += pts[pts[i].neighbours[n]].iR;
					sn += 1;
				}

				if (sn > 0)
				{
					pts[i].isGood = true;
					pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;
				}
			}
		}
	}

	/**
	 * @brief 求出状态增量后, 计算被边缘化掉的逆深度的更新量, 更新逆深度
	 * 
	 * @param[in] lvl  金字塔层数
	 * @param[in] lambda  LM求解的时候对角线的系数变化
	 * @param[in] inc  8位状态变量的增量结果, 用于求解逆深度部分正规方程的右侧b部分
	 */
	void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
	{
		const float maxPixelStep = 0.25;
		const float idMaxStep = 1e10;
		Pnt *pts = points[lvl];
		int npts = numPoints[lvl];
		for (int i = 0; i < npts; i++)
		{
			if (!pts[i].isGood)
				continue;
			// Step 1 计算这个点的逆深度增量值, 对角阵求逆，直接变成了n个标量方程的计算
			// JbBuffer[i][8] 是  (Jρ).t * r21
			// JbBuffer[i].head<8>().dot(inc)对应Hρx21*δx21
			//! dd*r + (dp*dd)^T*delta_p
			float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
			//! dd * delta_d = dd*r - (dp*dd)^T*delta_p = b
			//! delta_d = b * dd^-1
			// 这个是逆深度的增量值	  
			//; 注意这个时候jbBuffer[i][9]已经就是V^-1了，所以这里直接*即可，不用/
			float step = -b * JbBuffer[i][9] / (1 + lambda);
			
			//; 逆深度更新的最大增量值
			float maxstep = maxPixelStep * pts[i].maxstep; // 逆深度最大只能增加这些
			if (maxstep > idMaxStep)
				maxstep = idMaxStep;

			if (step > maxstep)
				step = maxstep;
			if (step < -maxstep)
				step = -maxstep;

			// 更新得到新的逆深度
			// 这个是此点最终的新逆深度值
			float newIdepth = pts[i].idepth + step;
			if (newIdepth < 1e-3)
				newIdepth = 1e-3;
			if (newIdepth > 50)
				newIdepth = 50;
			// 点的最新逆深度值
			pts[i].idepth_new = newIdepth;
		}
	}

	//* 新的值赋值给旧的 (能量, 点状态, 逆深度, hessian)
	void CoarseInitializer::applyStep(int lvl)
	{
		Pnt *pts = points[lvl];
		int npts = numPoints[lvl];
		for (int i = 0; i < npts; i++)
		{
			// 如果当前点是坏点，把逆深度和最新的逆深度都复制成对逆深度的期望值 
			if (!pts[i].isGood)
			{
				pts[i].idepth = pts[i].idepth_new = pts[i].iR;
				continue;
			}
			// 否则是好点，然后就用这一层计算出来的新的能量、逆深度、协方差，来更新上一次的对应变量，为下一次做准备
			pts[i].energy = pts[i].energy_new;
			pts[i].isGood = pts[i].isGood_new;
			pts[i].idepth = pts[i].idepth_new;
			pts[i].lastHessian = pts[i].lastHessian_new;
		}
		std::swap<Vec10f *>(JbBuffer, JbBuffer_new);
	}

	//@ 计算每个金字塔层的相机参数
	void CoarseInitializer::makeK(CalibHessian *HCalib)
	{
		w[0] = wG[0];
		h[0] = hG[0];

		fx[0] = HCalib->fxl();
		fy[0] = HCalib->fyl();
		cx[0] = HCalib->cxl();
		cy[0] = HCalib->cyl();
		// 求各层的K参数
		for (int level = 1; level < pyrLevelsUsed; ++level)
		{

			w[level] = w[0] >> level;
			h[level] = h[0] >> level;
			fx[level] = fx[level - 1] * 0.5;
			fy[level] = fy[level - 1] * 0.5;
			//* 0.5 offset 看README是设定0.5到1.5之间积分表示1的像素值？
			cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
			cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
		}
		// 求K_inverse参数
		for (int level = 0; level < pyrLevelsUsed; ++level)
		{
			K[level] << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
			Ki[level] = K[level].inverse();
			fxi[level] = Ki[level](0, 0);
			fyi[level] = Ki[level](1, 1);
			cxi[level] = Ki[level](0, 2);
			cyi[level] = Ki[level](1, 2);
		}
	}

	//@ 生成每一层点的KDTree, 并用其找到邻近点集和父点
	/**
	 * @brief 这个操作简单明了：为上一步在金字塔不同层提取到的特征点（坐标是在该点所在的金字塔层上的坐标）构建kdtree，
	 *        然后为每一层上的所有特征点，首先寻找同一层它的最近10个点标记为邻居点，然后把它坐标变换到上一层寻找上一层
	 *        最近的点标记为父亲点。
	 */
	void CoarseInitializer::makeNN()
	{
		/* 每一金字塔层选取的像素点构成一个KD树，为每层的点找到最近邻的10个点。pts[i].neighbours[myidx]=ret_index[k];
		  并且会在上一层找到该层像素点的parent，（最高层除外）pts[i].parent = ret_index[0];。用于后续提供初值，加速优化。
		 原文链接：https://blog.csdn.net/tanyong_98/article/details/106177134
		*/
		const float NNDistFactor = 0.05;
		// 第一个参数为distance, 第二个是datasetadaptor, 第三个是维数
		typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud>,
			FLANNPointcloud, 2> KDTree;

		// build indices
		FLANNPointcloud pcs[PYR_LEVELS]; // 每层建立一个点云
		KDTree *indexes[PYR_LEVELS];	 // 点云建立KDtree
		//* 每层建立一个KDTree索引二维点云
		for (int i = 0; i < pyrLevelsUsed; i++)
		{
			pcs[i] = FLANNPointcloud(numPoints[i], points[i]); // 二维点点云
			// 参数: 维度, 点数据, 叶节点中最大的点数(越大build快, query慢)
			indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
			indexes[i]->buildIndex();
		}

		const int nn = 10;

		// find NN & parents
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			Pnt *pts = points[lvl];
			int npts = numPoints[lvl];

			int ret_index[nn];	// 搜索到的临近点的索引
			float ret_dist[nn]; // 搜索到点的距离
			// 搜索结果, 最近的nn个和1个
			nanoflann::KNNResultSet<float, int, int> resultSet(nn);
			nanoflann::KNNResultSet<float, int, int> resultSet1(1);
			//; 遍历这一层的所有点
			for (int i = 0; i < npts; i++)
			{
				//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
				resultSet.init(ret_index, ret_dist);  //; 搜索的最近点的结果
				Vec2f pt = Vec2f(pts[i].u, pts[i].v); // 当前点
				// 使用建立的KDtree, 来查询最近邻
				indexes[lvl]->findNeighbors(resultSet, (float *)&pt, nanoflann::SearchParams());
				int myidx = 0;
				float sumDF = 0;
				//* 给每个点的neighbours赋值
				for (int k = 0; k < nn; k++)
				{
					pts[i].neighbours[myidx] = ret_index[k];	  // 最近的索引
					//! 疑问：距离使用指数形式是啥意思？而且还乘了一个常量0.05
					float df = expf(-ret_dist[k] * NNDistFactor); // 距离使用指数形式
					sumDF += df;								  // 距离和
					pts[i].neighboursDist[myidx] = df;  //; 存储指数距离
					assert(ret_index[k] >= 0 && ret_index[k] < npts);
					myidx++;
				}
				// 对距离进行归10化,,,,,
				//! 归10化又是啥操作？
				for (int k = 0; k < nn; k++)
					pts[i].neighboursDist[k] *= 10 / sumDF;

				//* 高一层的图像中找到该点的父节点
				if (lvl < pyrLevelsUsed - 1)
				{
					resultSet1.init(ret_index, ret_dist);  //; 在当前层的高一层搜索的父节点的结果
					pt = pt * 0.5f - Vec2f(0.25f, 0.25f); // 换算到高一层
					//; 利用高一层构造的kdtree查找父节点，注意使用的是resultSet1, 定义变量的时候传给构造函数的
					//; 参数是1，所以这里就只查找最近的一个点
					indexes[lvl + 1]->findNeighbors(resultSet1, (float *)&pt, nanoflann::SearchParams());

					pts[i].parent = ret_index[0];						   // 父节点
					pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor); // 到父节点的距离(在高层中的距离)

					assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
				}
				else // 最高层没有父节点
				{
					pts[i].parent = -1;
					pts[i].parentDist = -1;
				}
			}
		}

		// done.

		for (int i = 0; i < pyrLevelsUsed; i++)
			delete indexes[i];
	}
}
