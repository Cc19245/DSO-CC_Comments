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

#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso
{

	//@ 从ImmaturePoint构造函数, 不成熟点变地图点
	PointHessian::PointHessian(const ImmaturePoint *const rawPoint, CalibHessian *Hcalib)
	{
		instanceCounter++;
		host = rawPoint->host; // 主帧
		hasDepthPrior = false;

		idepth_hessian = 0;
		maxRelBaseline = 0;
		numGoodResiduals = 0;

		// set static values & initialization.
		u = rawPoint->u;
		v = rawPoint->v;
		assert(std::isfinite(rawPoint->idepth_max));
		//idepth_init = rawPoint->idepth_GT;

		my_type = rawPoint->my_type; //似乎是显示用的

		setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min) * 0.5); //深度均值
		setPointStatus(PointHessian::INACTIVE);

		int n = patternNum;
		memcpy(color, rawPoint->color, sizeof(float) * n); // 一个点对应8个像素
		memcpy(weights, rawPoint->weights, sizeof(float) * n);
		energyTH = rawPoint->energyTH;

		efPoint = 0; // 指针=0
	}

	//@ 释放residual
	void PointHessian::release()
	{
		for (unsigned int i = 0; i < residuals.size(); i++)
			delete residuals[i];
		residuals.clear();
	}

	//@ 设置固定线性化点位置的状态
	void FrameHessian::setStateZero(const Vec10 &state_zero)
	{
		//! 前六维位姿必须是0，因为前6维给的是位姿增量
		assert(state_zero.head<6>().squaredNorm() < 1e-20);

        // Step 1 设置相对线性化点的位姿增量、光度绝对量
		this->state_zero = state_zero;

        // Step 2 后面就是求零空间
		// 感觉这个nullspaces_pose就是 Adj_T
		// Exp(Adj_T*zeta)=T*Exp(zeta)*T^{-1}
		// 全局转为局部的，左乘边右乘
		// T_c_w * delta_T_g * T_c_w_inv = delta_T_l
		for (int i = 0; i < 6; i++)
		{
			Vec6 eps;
			eps.setZero();
			eps[i] = 1e-3;
			SE3 EepsP = Sophus::SE3::exp(eps);
			SE3 EepsM = Sophus::SE3::exp(-eps);
			SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
			SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
			nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
		}
		//nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
		//nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

		// scale change
		SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
		w2c_leftEps_P_x0.translation() *= 1.00001;
		w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
		SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
		w2c_leftEps_M_x0.translation() /= 1.00001;
		w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
		nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);

		nullspaces_affine.setZero();
		nullspaces_affine.topLeftCorner<2, 1>() = Vec2(1, 0);
		assert(ab_exposure > 0);
		nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
	};


	void FrameHessian::release()
	{
		// DELETE POINT
		// DELETE RESIDUAL
		for (unsigned int i = 0; i < pointHessians.size(); i++)
			delete pointHessians[i];
		for (unsigned int i = 0; i < pointHessiansMarginalized.size(); i++)
			delete pointHessiansMarginalized[i];
		for (unsigned int i = 0; i < pointHessiansOut.size(); i++)
			delete pointHessiansOut[i];
		for (unsigned int i = 0; i < immaturePoints.size(); i++)
			delete immaturePoints[i];

		pointHessians.clear();
		pointHessiansMarginalized.clear();
		pointHessiansOut.clear();
		immaturePoints.clear();
	}

	/**
	 * @brief  计算当前帧图像的各层金字塔图像的像素值和梯度
	 * 
	 * @param[in] color   传入的经过光度校正后的图像辐照值
	 * @param[in] HCalib  相机内参hessian
	 */
	void FrameHessian::makeImages(float *color, CalibHessian *HCalib)
	{
		// 每一层创建图像值, 和图像梯度的存储空间
		for (int i = 0; i < pyrLevelsUsed; i++)   // pyrLevelsUsed图像金字塔层数，设置为6
		{
			//; 3维向量的三个分量分别是：辐照值（或者简单认为没有去光度畸变的灰度值）、x方向梯度、y方向梯度
			dIp[i] = new Eigen::Vector3f[wG[i] * hG[i]];   //; wG[i]/hG[i]应该是每一层图像的宽高
			//; x和y方向梯度的平方和
			absSquaredGrad[i] = new float[wG[i] * hG[i]];  
		}
		dI = dIp[0]; // 原来他们指向同一个地方
		
		/*
		d=dIp[0];  获取金字塔第0层，若要获取其他层，修改中括号里面即可；
		d[idx][0]  表示图像金字塔第0层，idx位置处的像素的像素灰度值;(这是因为DSO中
					 存储图像像素值都是采用一维数组来表示，类似于opencv里面的data数组。)
		d[idx][1]  表示图像金字塔第0层，idx位置处的像素的x方向的梯度
		d[idx][2]  表示图像金字塔第0层，idx位置处的像素的y方向的梯度
		abs=absSquaredGrad[1]; ///获取金字塔第1层，若要获取其他层，修改中括号里面即可；
		abs[idx]   表示图像金字塔第1层，，idx位置处的像素x,y方向的梯度平方和
		*/

		// make d0
		int w = wG[0]; // 零层宽
		int h = hG[0]; // 零层高
		for (int i = 0; i < w * h; i++)
			dI[i][0] = color[i];  //; 最后一个索引0是Vector3d的第0个位置，即辐照

		// 遍历所有层的金字塔
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			int wl = wG[lvl], hl = hG[lvl]; // 当前层图像大小
			Eigen::Vector3f *dI_l = dIp[lvl];  // 当前层图像的Vector3d值
			float *dabs_l = absSquaredGrad[lvl];  // 当前层图像的梯度平方和

			// Step 1 : 计算第0层之上的各层的像素值，要用下层图像梯度值4合1求平均来计算
			if (lvl > 0)
			{
				int lvlm1 = lvl - 1;  //; 当前层图像的下一层索引
				int wlm1 = wG[lvlm1]; //; 下一层图像的宽度
				Eigen::Vector3f *dI_lm = dIp[lvlm1];  //; 下一层图像的Vector3d值

				// 下层图像的像素4合1求平均值得到上层图像的像素值, 生成金字塔
				for (int y = 0; y < hl; y++)
				{
					for (int x = 0; x < wl; x++)
					{
						// 上层金字塔图像的像素值是由下层图像的4个像素值均匀采样得到的。
						// 下述代码中dI_l表示上层金字塔图像首地址，dI_lm表示下层图像金字塔的首地址。
						dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x + 2 * y * wlm1][0] +
													   dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
													   dI_lm[2 * x + 2 * y * wlm1 + wlm1][0] +
													   dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
					}
				}
			}

			// Step 2 : 计算各层的梯度值，前面像素值都已经求完了，所以这里每层都可以算梯度
			for (int idx = wl; idx < wl * (hl - 1); idx++) // 注意从第二行的像素开始算
			{
				// 梯度的求取：利用前后两个像素的差值作为x方向的梯度，
				//   利用上下两个像素的差值作为y方向的梯度，注意会跳过边缘像素点的梯度计算。
				float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
				float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

				if (!std::isfinite(dx))
					dx = 0;
				if (!std::isfinite(dy))
					dy = 0;

				dI_l[idx][1] = dx; // 梯度
				dI_l[idx][2] = dy;

				dabs_l[idx] = dx * dx + dy * dy; // 梯度平方

				// HCalib != 0说明有相机校正类，这个一般都满足。
				// setting_gammaWeightsPixelSelect配置文件中设置的是1，注释说如果是1就是用原始像素，否则使用辐照
				if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0)
				{
					//! 乘上响应函数, 变换回正常的颜色, 因为光度矫正时 I = G^-1(I) / V(x)
					float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
					//; 这里把梯度恢复成像素梯度，也就是在去除光度响应函数之前的值
                    // convert to gradient of original color space (before removing response).
                    //; 这里有点意思，看这个变量的命名可以知道他说的是像素选择的时候的配置，这里也只是修改了像素梯度的平方
                    //; 并没有对像素梯度从辐照值变成像素值。
                    //TODO 看看像素梯度的平方在哪里使用了？为什么要这样设置？
					dabs_l[idx] *= gw * gw; 
				}
			}
		}
	}


	//@ 计算优化前和优化后的相对位姿, 相对光度变化, 及中间变量
    /**
     * @brief 这个代码对应深蓝PPT中的P38，由于使用了FEJ，所以这里就是在保存不同的状态，包括固定的线性化点处的相对位姿，
     *    优化更新状态后的相对位姿等(每次优化更新之后都会调用(调用这个函数的)函数，从而重新计算更新后的相对位姿)
     //! 7.27增：
        //; 相对状态量计算：计算帧帧之间的FEJ线性化点的相对量，以及帧帧之间当前最新状态的相对量，
        //;   因为后面求正规方程要用相对量
     * @param[in] host 
     * @param[in] target 
     * @param[in] HCalib 
     */
	void FrameFramePrecalc::set(FrameHessian *host, FrameHessian *target, CalibHessian *HCalib)
	{
		this->host = host; // 这个是赋值, 计数会增加, 不是拷贝
		this->target = target;

		// 实在不懂leftToleft_0这个名字怎么个含义
        // Step 1 线性化点的值
		// 优化前host target间位姿变换
        //TODO 这里最新帧的线性化点get_worldToCam_evalPT是在哪设置的？找了很久也没有找到
        //; get_worldToCam_evalPT是这一帧正在估计的相机位姿？那么这个在优化之前调用，也就是优化之前的相对位姿？
		SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
		PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();  //; PRE是Precalc的前缀pre, 表示预计算
		PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();

        // Step 2 优化后的值，也就是最新的位姿
		// 优化后host到target间位姿变换
        //! 疑问：这里为什么用的是预计算的值来求优化后的位姿变换？
		SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
		PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
		PRE_tTll = (leftToLeft.translation()).cast<float>();
		distanceLL = leftToLeft.translation().norm();

        // Step 3 优化后内参的一些变化量
		// 乘上内参, 中间量?
        //; 这里没有调用makeK，那么是最新的内参吗？
        //; CC解答：应该是的
		Mat33f K = Mat33f::Zero();
		K(0, 0) = HCalib->fxl();
		K(1, 1) = HCalib->fyl();
		K(0, 2) = HCalib->cxl();
		K(1, 2) = HCalib->cyl();
		K(2, 2) = 1;
		PRE_KRKiTll = K * PRE_RTll * K.inverse();
		PRE_RKiTll = PRE_RTll * K.inverse();
		PRE_KtTll = K * PRE_tTll;

		// 光度仿射值
		PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
		PRE_b0_mode = host->aff_g2l_0().b;
	}
}
