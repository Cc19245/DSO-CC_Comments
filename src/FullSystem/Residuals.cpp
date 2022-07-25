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
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso
{
	int PointFrameResidual::instanceCounter = 0;

	long runningResID = 0;

	PointFrameResidual::PointFrameResidual()
	{
		assert(false);
		instanceCounter++;
	}

	PointFrameResidual::~PointFrameResidual()
	{
		assert(efResidual == 0);
		instanceCounter--;
		delete J;
	}

	PointFrameResidual::PointFrameResidual(PointHessian *point_, FrameHessian *host_, FrameHessian *target_) : 
        point(point_), host(host_), target(target_)
	{
		efResidual = 0;
		instanceCounter++;
		resetOOB();
		J = new RawResidualJacobian(); // 各种雅克比
		assert(((long)J) % 16 == 0);   // 16位对齐

		isNew = true;
	}

	//@ 求对各个参数的导数, 和能量值
    /**
     * @brief 后端滑窗优化的时候，计算点的残差对各个优化变量的雅克比
     *   1.在计算 RawResidualJacobian 时，是计算 PointFrameResidual 的 J，尔后会将这个 J 转移到
     *     EFResidual 的 J，并且计算 EFResidual::JpJdF
     *   2.参考涂金戈的博客，写的非常好：https://www.cnblogs.com/JingeTU/p/8395046.html
     * @param[in] HCalib 
     * @return double 
     */
	double PointFrameResidual::linearize(CalibHessian *HCalib)
	{
		state_NewEnergyWithOutlier = -1;

		if (state_state == ResState::OOB)
		{
			state_NewState = ResState::OOB;
			return state_energy;
		}

		FrameFramePrecalc *precalc = &(host->targetPrecalc[target->idx]); // 得到这个目标帧在主帧上的一些预计算参数
		float energyLeft = 0;											 
		const Eigen::Vector3f *dIl = target->dI;
		const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
		const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
		const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
		const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
		const float *const color = point->color; // host帧上颜色
		const float *const weights = point->weights;

		Vec2f affLL = precalc->PRE_aff_mode; // 待优化的a和b, 就是host和target合的
		float b0 = precalc->PRE_b0_mode;	 // 主帧的单独 b

		// x=0时候求几何的导数, 使用FEJ!! ,逆深度没有使用FEJ
		Vec6f d_xi_x, d_xi_y;
		Vec4f d_C_x, d_C_y;
		float d_d_x, d_d_y;

		{
            //; uv是target帧上的归一化平面坐标，dresclae是/rou_2 * /rou_1^-1, new_idepth就是/rou_2
			float drescale, u, v, new_idepth;  
			float Ku, Kv;  //; Ku,Kv是target帧上的像素坐标
			Vec3f KliP;  //; KliP是host帧的归一化平面上的点

            // Step 1 把当前点利用 线性化点处的状态 投影到target帧上
			if (!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0, HCalib,
							  PRE_RTll_0, PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{
				state_NewState = ResState::OOB;
				return state_energy;
			} // 投影不在图像里, 则返回OOB

			centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

			//TODO 这些在初始化都写过了又写一遍 !!! 放到一起好不好, ai

            // Step 2 求解像素点对各个状态的雅克比
			//* 像素点对host上逆深度求导(由于乘了SCALE_IDEPTH倍, 因此乘上)
            // Step 2.1 像素点Pj'对逆深度求导，Vec2f Jpdd，2x1
			// diff d_idepth
			d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u) * SCALE_IDEPTH * HCalib->fxl();
			d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v) * SCALE_IDEPTH * HCalib->fyl();

            // Step 2.2 像素点Pj'对相机内参求导，VecCf Jpdc[2]，2x4
			//* 像素点对相机内参fx fy cx cy的导数第一部分
            //; 2.1.像素点对内参求导的反投影部分，包括对坐标u和v两部分的求导
            //! 疑问：怎么感觉下面的代码和公式推导有点对不上啊？哪里有点问题好像？
			// diff calib
			// [0]: 1/Pz'*Px*(R20*Px'/Pz' - R00)
			// [1]: 1/Pz'*Py*fx/fy*(R21*Px'/Pz' - R01)
			// [2]: 1/Pz'*(R20*Px'/Pz' - R00)
			// [3]: 1/Pz'*fx/fy*(R21*Px'/Pz' - R01)
            //; 这个地方涂金戈博客好像多个1?
            //; 解答：不是的，涂金戈博客算的是最后的结果，而这里算的是中间结果，所以都缺一点，但是在下面2.2的地方就都补上了
			d_C_x[2] = drescale * (PRE_RTll_0(2, 0) * u - PRE_RTll_0(0, 0));   //; 涂金戈博客 多1
			d_C_x[3] = HCalib->fxl() * drescale * (PRE_RTll_0(2, 1) * u - PRE_RTll_0(0, 1)) * HCalib->fyli();  //; 涂金戈博客 相同
			d_C_x[0] = KliP[0] * d_C_x[2];  //; 涂金戈博客 多u2'
			d_C_x[1] = KliP[1] * d_C_x[3];  //; 涂金戈博客 相同

			// [0]: 1/Pz'*Px*fy/fy*(R20*Py'/Pz' - R10)
			// [1]: 1/Pz'*Py*(R21*Py'/Pz' - R11)
			// [2]: 1/Pz'*fy/fy*(R20*Py'/Pz' - R10)
			// [3]: 1/Pz'*(R21*Py'/Pz' - R11)
			d_C_y[2] = HCalib->fyl() * drescale * (PRE_RTll_0(2, 0) * v - PRE_RTll_0(1, 0)) * HCalib->fxli();  //; 涂金戈博客 相同
			d_C_y[3] = drescale * (PRE_RTll_0(2, 1) * v - PRE_RTll_0(1, 1));   //; 涂金戈博客 多1
			d_C_y[0] = KliP[0] * d_C_y[2];   //; 涂金戈博客 多v2'
			d_C_y[1] = KliP[1] * d_C_y[3];   //; 涂金戈博客 相同

            //; 2.2.像素点对内参求导的投影部分
			//* 第二部分 同样project时候一样使用了scaled的内参
			// [Px'/Pz'  0  1  0;
			//  0  Py'/Pz'  0  1]
			d_C_x[0] = (d_C_x[0] + u) * SCALE_F;  //; 对比涂金戈博客，增加u2'
			d_C_x[1] *= SCALE_F;            
			d_C_x[2] = (d_C_x[2] + 1) * SCALE_C;  //; 对比涂金戈博客，增加1
			d_C_x[3] *= SCALE_C;

			d_C_y[0] *= SCALE_F;
			d_C_y[1] = (d_C_y[1] + v) * SCALE_F;  //; 对比涂金戈博客，增加v2'
			d_C_y[2] *= SCALE_C; 
			d_C_y[3] = (d_C_y[3] + 1) * SCALE_C;  //; 对比图净额博客，增加1

            // Step 2.3 像素点Pj'对位姿求导，Vec6f Jpdxi[2]， 2x6
			//* 像素点对位姿的导数, 位移在前!
			// 公式见初始化那儿
			d_xi_x[0] = new_idepth * HCalib->fxl();
			d_xi_x[1] = 0;
			d_xi_x[2] = -new_idepth * u * HCalib->fxl();
			d_xi_x[3] = -u * v * HCalib->fxl();
			d_xi_x[4] = (1 + u * u) * HCalib->fxl();
			d_xi_x[5] = -v * HCalib->fxl();

			d_xi_y[0] = 0;
			d_xi_y[1] = new_idepth * HCalib->fyl();
			d_xi_y[2] = -new_idepth * v * HCalib->fyl();
			d_xi_y[3] = -(1 + v * v) * HCalib->fyl();
			d_xi_y[4] = u * v * HCalib->fyl();
			d_xi_y[5] = u * HCalib->fyl();
		}

        //; 把上面求导的中间变量结果放到类成员变量中
		{
            // 新帧上像素坐标对位姿导数, 2x6
			J->Jpdxi[0] = d_xi_x;
			J->Jpdxi[1] = d_xi_y;

            // 新帧上像素坐标对相机内参导数，2x4
			J->Jpdc[0] = d_C_x;
			J->Jpdc[1] = d_C_y;

            // 新帧上像素坐标对逆深度导数，2x1
			J->Jpdd[0] = d_d_x;
			J->Jpdd[1] = d_d_y;
		}

		float JIdxJIdx_00 = 0, JIdxJIdx_11 = 0, JIdxJIdx_10 = 0;
		float JabJIdx_00 = 0, JabJIdx_01 = 0, JabJIdx_10 = 0, JabJIdx_11 = 0;
		float JabJab_00 = 0, JabJab_01 = 0, JabJab_11 = 0;

		float wJI2_sum = 0;

        //; 遍历当前点周围的pattern点
		for (int idx = 0; idx < patternNum; idx++)
		{
			float Ku, Kv;
			//? 为啥这里使用idepth_scaled, 上面使用的是zero； 
            // 答： 其实和上面一样的....同时调用了setIdepth() setIdepthZero()
			// 答: 这里是求图像导数, 由于线性误差大, 就不使用FEJ, 所以使用当前的状态
            //; 注意这里使用的是当前的状态
			if (!projectPoint(point->u + patternP[idx][0], point->v + patternP[idx][1], 
                point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{
				state_NewState = ResState::OOB;
				return state_energy;
			}

			// 像素坐标
			projectedTo[idx][0] = Ku;
			projectedTo[idx][1] = Kv;

            //; 插值得到辐照值
			Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
			float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]); // 残差
            
            //; 对光度参数求导
			//* 残差对光度仿射a求导
			// 光度参数使用固定线性化点了
			float drdA = (color[idx] - b0);

			if (!std::isfinite((float)hitColor[0]))
			{
				state_NewState = ResState::OOB;
				return state_energy;
			}

            //; 这里很重要，和梯度大小成反比的权重
			//* 和梯度大小成比例的权重
			float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
			//* 和patch位置相关的权重
			w = 0.5f * (w + weights[idx]);

			//* huber函数, 能量值(chi2)
			float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
			energyLeft += w * w * hw * residual * residual * (2 - hw);

			{
				if (hw < 1)
					hw = sqrtf(hw);
				hw = hw * w;

				hitColor[1] *= hw;
				hitColor[2] *= hw;

				// 残差 res*w*sqrt(hw)
				J->resF[idx] = residual * hw;

                // Step 2.4 Pattern的残差对像素点Pj'求导，VecNRf JIdx[2]， 2x8
				// 图像导数 dx dy
				J->JIdx[0][idx] = hitColor[1];
				J->JIdx[1][idx] = hitColor[2];

                // Step 2.5 Pattern的残差对广度求导，VecNRf JabF[2]， 2x8、
				// 对光度合成后a b的导数 [Ii-b0  1]
				// Ij - a*Ii - b  (a = tj*e^aj / ti*e^ai,   b = bj - a*bi)
				//bug 正负号有影响 ???
                //! 注意：这部分和涂金戈博客对不上，它博客中也说了。看上面公益群也说了好像有个bug?
                // 对光度参数求导，其中a是固定了线性化点？
				J->JabF[0][idx] = drdA * hw;
				J->JabF[1][idx] = hw;

                
                // Step 2.6 这里就是在计算涂金戈博客中写的789部分，比如2x8 * 8x2 = 2x2，所以这里用的是+=来累加
				// dIdx&dIdx hessian block
				JIdxJIdx_00 += hitColor[1] * hitColor[1];
				JIdxJIdx_11 += hitColor[2] * hitColor[2];
				JIdxJIdx_10 += hitColor[1] * hitColor[2];
				// dIdx&dIdab hessian block
				JabJIdx_00 += drdA * hw * hitColor[1];
				JabJIdx_01 += drdA * hw * hitColor[2];
				JabJIdx_10 += hw * hitColor[1];
				JabJIdx_11 += hw * hitColor[2];
				// dIdab&dIdab hessian block
				JabJab_00 += drdA * drdA * hw * hw;
				JabJab_01 += drdA * hw * hw;
				JabJab_11 += hw * hw;

				wJI2_sum += hw * hw * (hitColor[1] * hitColor[1] + hitColor[2] * hitColor[2]); // 梯度平方

				if (setting_affineOptModeA < 0)
					J->JabF[0][idx] = 0;
				if (setting_affineOptModeB < 0)
					J->JabF[1][idx] = 0;
			}
		} // 遍历pattern结束

        //; 把上面pattern累加的2x8 * 8x2 = 2*2 等结果存入到类成员变量中
		// 都是对host到target之间的变化量导数
		J->JIdx2(0, 0) = JIdxJIdx_00;
		J->JIdx2(0, 1) = JIdxJIdx_10;
		J->JIdx2(1, 0) = JIdxJIdx_10;
		J->JIdx2(1, 1) = JIdxJIdx_11;
		J->JabJIdx(0, 0) = JabJIdx_00;
		J->JabJIdx(0, 1) = JabJIdx_01;
		J->JabJIdx(1, 0) = JabJIdx_10;
		J->JabJIdx(1, 1) = JabJIdx_11;
		J->Jab2(0, 0) = JabJab_00;
		J->Jab2(0, 1) = JabJab_01;
		J->Jab2(1, 0) = JabJab_01;
		J->Jab2(1, 1) = JabJab_11;

		state_NewEnergyWithOutlier = energyLeft;

		//* 大于阈值则视为有外点
		if (energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2)
		{
			energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
			state_NewState = ResState::OUTLIER;
		}
		else
		{
			state_NewState = ResState::IN;
		}

		state_NewEnergy = energyLeft;
		return energyLeft;
	}


	void PointFrameResidual::debugPlot()
	{
		if (state_state == ResState::OOB)
			return;
		Vec3b cT = Vec3b(0, 0, 0);

		if (freeDebugParam5 == 0)
		{
			float rT = 20 * sqrt(state_energy / 9);
			if (rT < 0)
				rT = 0;
			if (rT > 255)
				rT = 255;
			cT = Vec3b(0, 255 - rT, rT);
		}
		else
		{
			if (state_state == ResState::IN)
				cT = Vec3b(255, 0, 0);
			else if (state_state == ResState::OOB)
				cT = Vec3b(255, 255, 0);
			else if (state_state == ResState::OUTLIER)
				cT = Vec3b(0, 0, 255);
			else
				cT = Vec3b(255, 255, 255);
		}

		for (int i = 0; i < patternNum; i++)
		{
			if ((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 && projectedTo[i][0] < wG[0] - 3 && projectedTo[i][1] < hG[0] - 3))
				target->debugImage->setPixel1((float)projectedTo[i][0], (float)projectedTo[i][1], cT);
		}
	}

	//@ 把计算的残差,雅克比值给EFResidual, 更新残差的状态(好坏)
	void PointFrameResidual::applyRes(bool copyJacobians)
	{
        //; 调用的时候传入的都是true
		if (copyJacobians)
		{
			if (state_state == ResState::OOB)
			{
				assert(!efResidual->isActiveAndIsGoodNEW);
				return; // can never go back from OOB
			}
			if (state_NewState == ResState::IN) // && )
			{
				efResidual->isActiveAndIsGoodNEW = true;
				//; 终点：takeDataF，把PointFrameResidual中计算的雅克比中间量传给EF后端
				efResidual->takeDataF(); // 从当前取jacobian数据
			}
			else
			{
				efResidual->isActiveAndIsGoodNEW = false;
			}
		}

		setState(state_NewState);
		state_energy = state_NewEnergy;
	}
}
