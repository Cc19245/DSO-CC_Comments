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

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "IOWrapper/ImageRW.h"
#include <algorithm>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

	//! 生成2^b个字节对齐
	template <int b, typename T>
	T *allocAligned(int size, std::vector<T *> &rawPtrVec)
	{
		const int padT = 1 + ((1 << b) / sizeof(T)); //? 为什么加上这个值  答: 为了对齐,下面会移动b
		T *ptr = new T[size + padT];
		rawPtrVec.push_back(ptr);
		T *alignedPtr = (T *)((((uintptr_t)(ptr + padT)) >> b) << b); //! 左移右移之后就会按照2的b次幂字节对齐, 丢掉不对齐的
		return alignedPtr;
	}

	//@ 构造函数, 申请内存, 初始化
	CoarseTracker::CoarseTracker(int ww, int hh) : lastRef_aff_g2l(0, 0)
	{
		// make coarse tracking templates.
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			int wl = ww >> lvl;
			int hl = hh >> lvl;

			idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
			weightSums[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
			weightSums_bak[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);

			pc_u[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
			pc_v[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
			pc_idepth[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
			pc_color[lvl] = allocAligned<4, float>(wl * hl, ptrToDelete);
		}

		// warped buffers
		buf_warped_idepth = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_u = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_v = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_dx = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_dy = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_residual = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_weight = allocAligned<4, float>(ww * hh, ptrToDelete);
		buf_warped_refColor = allocAligned<4, float>(ww * hh, ptrToDelete);

		newFrame = 0;
		lastRef = 0;
		debugPlot = debugPrint = true;
		w[0] = h[0] = 0;
		refFrameID = -1;
	}
	CoarseTracker::~CoarseTracker()
	{
		for (float *ptr : ptrToDelete)
			delete[] ptr;
		ptrToDelete.clear();
	}

	//@ 构造内参矩阵, 以及一些中间量,
	//TODO  每个类都有这个, 直接用一个多好
	void CoarseTracker::makeK(CalibHessian *HCalib)
	{
		w[0] = wG[0];
		h[0] = hG[0];

		fx[0] = HCalib->fxl();
		fy[0] = HCalib->fyl();
		cx[0] = HCalib->cxl();
		cy[0] = HCalib->cyl();

		for (int level = 1; level < pyrLevelsUsed; ++level)
		{
			w[level] = w[0] >> level;
			h[level] = h[0] >> level;
			fx[level] = fx[level - 1] * 0.5;
			fy[level] = fy[level - 1] * 0.5;
			cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
			cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
		}

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

	//@ 使用在当前帧上投影的点的逆深度, 来生成每个金字塔层上点的逆深度值
	void CoarseTracker::makeCoarseDepthL0(std::vector<FrameHessian *> frameHessians)
	{
		// make coarse tracking templates for latstRef.
		memset(idepth[0], 0, sizeof(float) * w[0] * h[0]); // 第0层
		memset(weightSums[0], 0, sizeof(float) * w[0] * h[0]);
		//[ ***step 1*** ] 计算其它点在最新帧投影第0层上的各个像素的逆深度权重, 和加权逆深度
		for (FrameHessian *fh : frameHessians)
		{
			for (PointHessian *ph : fh->pointHessians)
			{
				// 点的上一次残差正常
				//* 优化之后上一次不好的置为0，用来指示，而点是没有删除的，残差删除了
				if (ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN)
				{
					PointFrameResidual *r = ph->lastResiduals[0].first;
					assert(r->efResidual->isActive() && r->target == lastRef); // 点的残差是好的, 上一次优化的target是这次的ref
					int u = r->centerProjectedTo[0] + 0.5f;					   // 四舍五入
					int v = r->centerProjectedTo[1] + 0.5f;
					float new_idepth = r->centerProjectedTo[2];
					float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12)); // 协方差逆做权重

					idepth[0][u + w[0] * v] += new_idepth * weight; // 加权后的
					weightSums[0][u + w[0] * v] += weight;
				}
			}
		}

		//[ ***step 2*** ] 从下层向上层生成逆深度和权重
		for (int lvl = 1; lvl < pyrLevelsUsed; lvl++)
		{
			int lvlm1 = lvl - 1;
			int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

			float *idepth_l = idepth[lvl];
			float *weightSums_l = weightSums[lvl];

			float *idepth_lm = idepth[lvlm1];
			float *weightSums_lm = weightSums[lvlm1];

			for (int y = 0; y < hl; y++)
				for (int x = 0; x < wl; x++)
				{
					int bidx = 2 * x + 2 * y * wlm1;
					//? 为什么不除以4   答: 后面除以权重的和了 nice!
					idepth_l[x + y * wl] = idepth_lm[bidx] +
										   idepth_lm[bidx + 1] +
										   idepth_lm[bidx + wlm1] +
										   idepth_lm[bidx + wlm1 + 1];

					weightSums_l[x + y * wl] = weightSums_lm[bidx] +
											   weightSums_lm[bidx + 1] +
											   weightSums_lm[bidx + wlm1] +
											   weightSums_lm[bidx + wlm1 + 1];
				}
		}

		//[ ***step 3*** ] 0和1层 对于没有深度的像素点, 使用周围斜45度的四个点来填充
		// dilate idepth by 1.
		for (int lvl = 0; lvl < 2; lvl++)
		{
			int numIts = 1;

			for (int it = 0; it < numIts; it++)
			{
				int wh = w[lvl] * h[lvl] - w[lvl]; // 空出一行
				int wl = w[lvl];
				float *weightSumsl = weightSums[lvl];
				float *weightSumsl_bak = weightSums_bak[lvl];
				memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float)); // 备份
				float *idepthl = idepth[lvl];										   // dotnt need to make a temp copy of depth, since I only
																					   // read values with weightSumsl>0, and write ones with weightSumsl<=0.
				for (int i = w[lvl]; i < wh; i++)									   // 上下各空一行
				{
					if (weightSumsl_bak[i] <= 0)
					{
						// 使用四个角上的点来填充没有深度的
						//bug: 对于竖直边缘上的点不太好把, 使用上两行的来计算
						float sum = 0, num = 0, numn = 0;
						if (weightSumsl_bak[i + 1 + wl] > 0)
						{
							sum += idepthl[i + 1 + wl];
							num += weightSumsl_bak[i + 1 + wl];
							numn++;
						}
						if (weightSumsl_bak[i - 1 - wl] > 0)
						{
							sum += idepthl[i - 1 - wl];
							num += weightSumsl_bak[i - 1 - wl];
							numn++;
						}
						if (weightSumsl_bak[i + wl - 1] > 0)
						{
							sum += idepthl[i + wl - 1];
							num += weightSumsl_bak[i + wl - 1];
							numn++;
						}
						if (weightSumsl_bak[i - wl + 1] > 0)
						{
							sum += idepthl[i - wl + 1];
							num += weightSumsl_bak[i - wl + 1];
							numn++;
						}
						if (numn > 0)
						{
							idepthl[i] = sum / numn;
							weightSumsl[i] = num / numn;
						}
					}
				}
			}
		}

		//[ ***step 4*** ] 2层向上, 对于没有深度的像素点, 使用上下左右的四个点来填充
		// dilate idepth by 1 (2 on lower levels).
		for (int lvl = 2; lvl < pyrLevelsUsed; lvl++)
		{
			int wh = w[lvl] * h[lvl] - w[lvl];
			int wl = w[lvl];
			float *weightSumsl = weightSums[lvl];
			float *weightSumsl_bak = weightSums_bak[lvl];
			memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
			float *idepthl = idepth[lvl]; // dotnt need to make a temp copy of depth, since I only
										  // read values with weightSumsl>0, and write ones with weightSumsl<=0.
			for (int i = w[lvl]; i < wh; i++)
			{
				if (weightSumsl_bak[i] <= 0)
				{
					float sum = 0, num = 0, numn = 0;
					if (weightSumsl_bak[i + 1] > 0)
					{
						sum += idepthl[i + 1];
						num += weightSumsl_bak[i + 1];
						numn++;
					}
					if (weightSumsl_bak[i - 1] > 0)
					{
						sum += idepthl[i - 1];
						num += weightSumsl_bak[i - 1];
						numn++;
					}
					if (weightSumsl_bak[i + wl] > 0)
					{
						sum += idepthl[i + wl];
						num += weightSumsl_bak[i + wl];
						numn++;
					}
					if (weightSumsl_bak[i - wl] > 0)
					{
						sum += idepthl[i - wl];
						num += weightSumsl_bak[i - wl];
						numn++;
					}
					if (numn > 0)
					{
						idepthl[i] = sum / numn;
						weightSumsl[i] = num / numn;
					}
				}
			}
		}

		//[ ***step 5*** ] 归一化点的逆深度并赋值给成员变量pc_*
		// normalize idepths and weights.
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			float *weightSumsl = weightSums[lvl];
			float *idepthl = idepth[lvl];
			Eigen::Vector3f *dIRefl = lastRef->dIp[lvl];

			int wl = w[lvl], hl = h[lvl];

			int lpc_n = 0;
			//!!!! 指针, 只是把指针传过去, 怎么总想有没有赋值, 智障

			float *lpc_u = pc_u[lvl];
			float *lpc_v = pc_v[lvl];
			float *lpc_idepth = pc_idepth[lvl];
			float *lpc_color = pc_color[lvl];

			for (int y = 2; y < hl - 2; y++)
				for (int x = 2; x < wl - 2; x++)
				{
					int i = x + y * wl;

					if (weightSumsl[i] > 0) // 有值的
					{
						idepthl[i] /= weightSumsl[i];
						lpc_u[lpc_n] = x;
						lpc_v[lpc_n] = y;
						lpc_idepth[lpc_n] = idepthl[i];
						lpc_color[lpc_n] = dIRefl[i][0];

						if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0))
						{
							idepthl[i] = -1;
							continue; // just skip if something is wrong.
						}
						lpc_n++;
					}
					else
						idepthl[i] = -1;

					weightSumsl[i] = 1; // 求完就变成1了
				}

			pc_n[lvl] = lpc_n;
		}
	}


    /**
     * @brief 对跟踪的最新帧和参考帧之间的残差, 求 Hessian 和 b
     * 
     * @param[in] lvl  当前最新帧所在的金字塔层数
     * @param[in] H_out   计算得到的H
     * @param[in] b_out   计算得到的b
     * @param[in] refToNew 当前最新帧到参考帧之间的位姿变化初值
     * @param[in] aff_g2l  当前最新帧到参考帧之间的广度初值
     */
	void CoarseTracker::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l)
	{
		acc.initialize();

		__m128 fxl = _mm_set1_ps(fx[lvl]);
		__m128 fyl = _mm_set1_ps(fy[lvl]);
        //; 这个是计算上一个参考关键帧的绝对广度系数？
        //! 这个地方看深蓝的PPT，PPT中有写的对应的公式，其中a和b0部分明确标记出来了
		__m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
		__m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]));

		__m128 one = _mm_set1_ps(1);
		__m128 minusOne = _mm_set1_ps(-1);
		__m128 zero = _mm_set1_ps(0);

        //; 注意这个变量就是判断必须是16字节对齐的，这样才可以使用SSE加速。而16字节对齐在上一步计算
        //; 残差的最后已经手动补齐了，所以这里不会出现问题。
		int n = buf_warped_n;
		assert(n % 4 == 0);  
        //; 遍历所有的构造残差的点，128位一跳，也就是每次可以计算4个点的H和b，加速运算
		for (int i = 0; i < n; i += 4)
		{
            //; 这里load_ps感觉应该是从某个变量开始，然后一次性取128位？也就是4个float
			__m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx + i), fxl); // dx*fx
			__m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy + i), fyl); // dy*fy
			__m128 u = _mm_load_ps(buf_warped_u + i);
			__m128 v = _mm_load_ps(buf_warped_v + i);
			__m128 id = _mm_load_ps(buf_warped_idepth + i);

            //! 重要：传入计算的雅克比J，利用updateSSE_eighted直接计算出来H和b
			acc.updateSSE_eighted(
                //! 疑问：这里有点没看懂，按照深蓝PPT中，分母上还有Pz'啊？难道是Pz'=1?
				_mm_mul_ps(id, dx),	    // 对位移x导数
				_mm_mul_ps(id, dy),     // 对位移y导数
                // 对位移z导数，对应PPT中公式是把-dpi提出来，然后再计算
				_mm_sub_ps(zero, _mm_mul_ps(id, _mm_add_ps(_mm_mul_ps(u, dx), _mm_mul_ps(v, dy)))), 
                // 对旋转xi_1求导
				_mm_sub_ps(zero, _mm_add_ps(_mm_mul_ps(_mm_mul_ps(u, v), dx),
									 _mm_mul_ps(dy, _mm_add_ps(one, _mm_mul_ps(v, v))))), 
				// 对旋转xi_2求导
                _mm_add_ps(
					_mm_mul_ps(_mm_mul_ps(u, v), dy),
					_mm_mul_ps(dx, _mm_add_ps(one, _mm_mul_ps(u, u)))),			
                // 对旋转xi_3求导	
				_mm_sub_ps(_mm_mul_ps(u, dy), _mm_mul_ps(v, dx)),		
                // 对目标帧a求导			 
				_mm_mul_ps(a, _mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor + i))), 
                // 对目标帧b求导
				minusOne,															 
				_mm_load_ps(buf_warped_residual + i),	// 残差
				_mm_load_ps(buf_warped_weight + i)      // huber权重
            );	
		}
		acc.finish();
        //! 疑问： 取出H和b，这里为什么要/n呢？
		H_out = acc.H.topLeftCorner<8, 8>().cast<double>() * (1.0f / n);
		b_out = acc.H.topRightCorner<8, 1>().cast<double>() * (1.0f / n);

        //; 为了保证数值稳定性，还要对H和b的不同部位进行缩放
        //! bug : 平移旋转顺序错了
		H_out.block<8, 3>(0, 0) *= SCALE_XI_ROT; 
		H_out.block<8, 3>(0, 3) *= SCALE_XI_TRANS;
		H_out.block<8, 1>(0, 6) *= SCALE_A;
		H_out.block<8, 1>(0, 7) *= SCALE_B;
		H_out.block<3, 8>(0, 0) *= SCALE_XI_ROT;
		H_out.block<3, 8>(3, 0) *= SCALE_XI_TRANS;
		H_out.block<1, 8>(6, 0) *= SCALE_A;
		H_out.block<1, 8>(7, 0) *= SCALE_B;
		b_out.segment<3>(0) *= SCALE_XI_ROT;
		b_out.segment<3>(3) *= SCALE_XI_TRANS;
		b_out.segment<1>(6) *= SCALE_A;
		b_out.segment<1>(7) *= SCALE_B;
	}


    /**
     * @brief 计算当前位姿投影得到的残差(能量值), 并进行一些统计
     *      构造尽量多的点, 有助于跟踪
     * @param[in] lvl  当前的金字塔层数
     * @param[in] refToNew 当前帧初始位姿 
     * @param[in] aff_g2l  当前帧初始光度值
     * @param[in] cutoffTH ？？？
     * @return Vec6 
     */
	Vec6 CoarseTracker::calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH)
	{
		float E = 0;
		int numTermsInE = 0;
		int numTermsInWarped = 0;
		int numSaturated = 0;

        //; 这一层图像宽高、图像梯度、相机内参
		int wl = w[lvl];
		int hl = h[lvl];
		Eigen::Vector3f *dINewl = newFrame->dIp[lvl];
		float fxl = fx[lvl];
		float fyl = fy[lvl];
		float cxl = cx[lvl];
		float cyl = cy[lvl];

        //; 旋转矩阵*内参
		Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
		Vec3f t = (refToNew.translation()).cast<float>();
		// 这个函数会把前后两帧的光度参数变成两个值
        //; 这里应该就是刚开始学的时候讲的那个，把绝对的光度参数变成两帧之间的相对光度参数(a,b)
		Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l).cast<float>();

		float sumSquaredShiftT = 0;
		float sumSquaredShiftRT = 0;
		float sumSquaredShiftNum = 0;

		// 经过huber函数后的能量阈值
		float maxEnergy = 2 * setting_huberTH * cutoffTH - setting_huberTH * setting_huberTH; // energy for r=setting_coarseCutoffTH.

		MinimalImageB3 *resImage = 0; // 自己定义的图像 nb
		if (debugPlot)
		{
			resImage = new MinimalImageB3(wl, hl);
			resImage->setConst(Vec3b(255, 255, 255));
		}

		//* 投影在ref帧上的点
		int nl = pc_n[lvl];
		float *lpc_u = pc_u[lvl];
		float *lpc_v = pc_v[lvl];
		float *lpc_idepth = pc_idepth[lvl];
		float *lpc_color = pc_color[lvl];

        // Step 1 遍历当前金字塔层上的所有点，计算残差和能量值
		for (int i = 0; i < nl; i++)
		{
            //; 这个点的逆深度、这个点的xy坐标
			float id = lpc_idepth[i];
			float x = lpc_u[i];
			float y = lpc_v[i];

			// 投影点
            //! 疑问： 把当前帧的这个点，投影到参考帧上？
			Vec3f pt = RKi * Vec3f(x, y, 1) + t * id;
			float u = pt[0] / pt[2]; // 归一化坐标
			float v = pt[1] / pt[2];
			float Ku = fxl * u + cxl; // 像素坐标
			float Kv = fyl * v + cyl;
			float new_idepth = id / pt[2]; // 当前帧上的深度

            //! 疑问：靠，这在搞什么啊？
			if (lvl == 0 && i % 32 == 0) //* 第0层 每隔32个点
			{
				//* 只正的平移 translation only (positive)
                //; 注意这里直接乘以Ki，实际上就是旋转矩阵为单位帧，所以就相当于只有平移
				Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t * id;
				float uT = ptT[0] / ptT[2];
				float vT = ptT[1] / ptT[2];
				float KuT = fxl * uT + cxl;
				float KvT = fyl * vT + cyl;

				//* 只负的平移 translation only (negative)
				Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t * id;
				float uT2 = ptT2[0] / ptT2[2];
				float vT2 = ptT2[1] / ptT2[2];
				float KuT2 = fxl * uT2 + cxl;
				float KvT2 = fyl * vT2 + cyl;

				//* 旋转+负的平移 translation and rotation (negative)
				Vec3f pt3 = RKi * Vec3f(x, y, 1) - t * id;
				float u3 = pt3[0] / pt3[2];
				float v3 = pt3[1] / pt3[2];
				float Ku3 = fxl * u3 + cxl;
				float Kv3 = fyl * v3 + cyl;

				//translation and rotation (positive)
				//already have it.

				//* 统计像素的移动大小
				sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
				sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
				sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
				sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
				sumSquaredShiftNum += 2;
			}

			//* 图像边沿, 深度为负 则跳过
			if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0))
				continue;

			// 计算残差
			float refColor = lpc_color[i];
            //; 双线性插值得到新帧上的图像梯度
			Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl); // 新帧上插值
			if (!std::isfinite((float)hitColor[0]))
				continue;
            //; 索引0的位置是辐照，所以这里计算残差就是计算辐照的差
			float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
            // 添加鲁棒核函数
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

            //; 如果残差大于设置的阈值，那么需要进行限制，直接设置成最大的能量值
			if (fabs(residual) > cutoffTH)
			{
				if (debugPlot)
					resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0, 0, 255));
				E += maxEnergy; // 能量值
				numTermsInE++;	// E 中数目
				numSaturated++; // 大于阈值数目
			}
			else
			{
				if (debugPlot)
					resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(residual + 128, residual + 128, residual + 128));

                //; 对残差添加鲁棒核函数，再加入到总能量中。这个推导就是深蓝学院PPT最开始的那部分
				E += hw * residual * residual * (2 - hw);
				numTermsInE++;  // E 中数目

                //; 记录这个点投影到参考关键帧上的逆深度、像素梯度等信息
				buf_warped_idepth[numTermsInWarped] = new_idepth;
				buf_warped_u[numTermsInWarped] = u;
				buf_warped_v[numTermsInWarped] = v;
				buf_warped_dx[numTermsInWarped] = hitColor[1];
				buf_warped_dy[numTermsInWarped] = hitColor[2];
				buf_warped_residual[numTermsInWarped] = residual;
				buf_warped_weight[numTermsInWarped] = hw;
				buf_warped_refColor[numTermsInWarped] = lpc_color[i];
				numTermsInWarped++;
			}
		}

        // Step 2 把最后的结果进行16字节对齐填充，应该是为了方便后面做128位的SSE加速
		//* 16字节对齐, 填充上
		while (numTermsInWarped % 4 != 0)
		{
			buf_warped_idepth[numTermsInWarped] = 0;
			buf_warped_u[numTermsInWarped] = 0;
			buf_warped_v[numTermsInWarped] = 0;
			buf_warped_dx[numTermsInWarped] = 0;
			buf_warped_dy[numTermsInWarped] = 0;
			buf_warped_residual[numTermsInWarped] = 0;
			buf_warped_weight[numTermsInWarped] = 0;
			buf_warped_refColor[numTermsInWarped] = 0;
			numTermsInWarped++;
		}
		buf_warped_n = numTermsInWarped;  //; 类成员变量，要16字节对齐

		if (debugPlot)
		{
			IOWrap::displayImage("RES", resImage, false);
			IOWrap::waitKey(0);
			delete resImage;
		}

        // Step 3 返回结果
		Vec6 rs;
		rs[0] = E;											   // 投影的能量值
		rs[1] = numTermsInE;								   // 投影的点的数目
		rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1); // 纯平移时 平均像素移动的大小
		rs[3] = 0;
		rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1); // 平移+旋转 平均像素移动大小
		rs[5] = numSaturated / (float)numTermsInE;				// 大于cutoff阈值的百分比
		return rs;
	}


	//@ 把优化完的最新帧设为参考帧
	void CoarseTracker::setCoarseTrackingRef(
		std::vector<FrameHessian *> frameHessians)
	{
		assert(frameHessians.size() > 0);
		lastRef = frameHessians.back();
		makeCoarseDepthL0(frameHessians); // 生成逆深度估值

		refFrameID = lastRef->shell->id;
		lastRef_aff_g2l = lastRef->aff_g2l();

		firstCoarseRMSE = -1;
	}

    /**
     * @brief 对新来的帧进行跟踪, 优化得到位姿, 光度参数
     * 
     * @param[in] newFrameHessian  最新帧的Hessian
     * @param[in] lastToNew_out  估计的当前帧到上一个参考帧的运动初值
     * @param[in] aff_g2l_out  估计的当前帧到上一个参考帧的光度变化的初值
     * @param[in] coarsestLvl  从哪个金字塔层数开始跟踪
     * @param[in] minResForAbort  最小的能量值，如果大于这个能量值的1.5倍，本次直接失败，继续下一次
     * @param[in] wrap 
     * @return true 
     * @return false 
     */
	bool CoarseTracker::trackNewestCoarse(
		FrameHessian *newFrameHessian,
		SE3 &lastToNew_out, AffLight &aff_g2l_out,
		int coarsestLvl,
		Vec5 minResForAbort,
		IOWrap::Output3DWrapper *wrap)
	{
		debugPlot = setting_render_displayCoarseTrackingFull;
		debugPrint = false;

        //; 一些判断，为啥传入的开始金字塔层数必须<5?
		assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);
        
        // Step 1 一些变量的初始化
		lastResiduals.setConstant(NAN);
		lastFlowIndicators.setConstant(1000);

		newFrame = newFrameHessian;
		int maxIterations[] = {10, 20, 50, 50, 50}; // 不同层迭代的次数
		float lambdaExtrapolationLimit = 0.001;

		SE3 refToNew_current = lastToNew_out; // 优化的初始值
		AffLight aff_g2l_current = aff_g2l_out;

		bool haveRepeated = false; // 是否重复计算了

		// Step 2 使用金字塔进行跟踪, 从顶层向下开始跟踪
		for (int lvl = coarsestLvl; lvl >= 0; lvl--)
		{
			Mat88 H;
			Vec8 b;
			float levelCutoffRepeat = 1;
			// Step 2.1. 计算残差
            //; setting_coarseCutoffTH是配置参数，默认20
			Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);

			// Step 2.2. 如果能量大于阈值的超过60%，则放大阈值再次计算，知道满足要求
			while (resOld[5] > 0.6 && levelCutoffRepeat < 50)
			{
				levelCutoffRepeat *= 2; // 超过阈值的多, 则放大阈值重新计算
				resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);

				if (!setting_debugout_runquiet)
					printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH * levelCutoffRepeat, resOld[5]);
			}

            // Step 2.3. 计算正规方程，内部使用SSE加速
			calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);

			float lambda = 0.01;

			if (debugPrint)
			{
				Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_current).cast<float>();
				printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
					   lvl, -1, lambda, 1.0f,
					   "INITIA",
					   0.0f,
					   resOld[0] / resOld[1],
					   0, (int)resOld[1],
					   0.0f);
				std::cout << refToNew_current.log().transpose() << " AFF " << aff_g2l_current.vec().transpose() << " (rel " << relAff.transpose() << ")\n";
			}

			// Step 2.4. 迭代优化，注意不同金字塔层的迭代次数不一样
			for (int iteration = 0; iteration < maxIterations[lvl]; iteration++)
			{
				// Step 2.4.1. 计算增量
				Mat88 Hl = H;
				for (int i = 0; i < 8; i++)
					Hl(i, i) *= (1 + lambda);  //; 对角线上+lambda，因此是LM算法求解
				Vec8 inc = Hl.ldlt().solve(-b);  //; 注意这里是-b，因为SSE计算的时候算的是J'e

                //; 再根据配置文件中的设置，是否固定广度系数。如果有广度校准文件的话，这里默认应该都是>0？
				if (setting_affineOptModeA < 0 && setting_affineOptModeB < 0) // fix a, b
				{
					inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
					inc.tail<2>().setZero();
				}
				if (!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0) // fix b
				{
					inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-b.head<7>());
					inc.tail<1>().setZero();
				}
				if (setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0)) // fix a
				{
					//? 怎么又换了个方法求....
					Mat88 HlStitch = Hl;
					Vec8 bStitch = b;
					HlStitch.col(6) = HlStitch.col(7);
					HlStitch.row(6) = HlStitch.row(7);
					bStitch[6] = bStitch[7];
					Vec7 incStitch = HlStitch.topLeftCorner<7, 7>().ldlt().solve(-bStitch.head<7>());
					inc.setZero();
					inc.head<6>() = incStitch.head<6>();
					inc[6] = 0;
					inc[7] = incStitch[6];
				}

				//? lambda太小的化, 就给增量一个因子, 啥原理????
				float extrapFac = 1;
				if (lambda < lambdaExtrapolationLimit)
					extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
				inc *= extrapFac;

                //; 对求解的增量再乘以缩放系数，才是最后真正的增量
				Vec8 incScaled = inc;
				incScaled.segment<3>(0) *= SCALE_XI_ROT;
				incScaled.segment<3>(3) *= SCALE_XI_TRANS;
				incScaled.segment<1>(6) *= SCALE_A;
				incScaled.segment<1>(7) *= SCALE_B;

				if (!std::isfinite(incScaled.sum()))
					incScaled.setZero();

                // Step 2.4.2. 使用增量更新后, 重新计算能量值
				SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
				AffLight aff_g2l_new = aff_g2l_current;
				aff_g2l_new.a += incScaled[6];
				aff_g2l_new.b += incScaled[7];

                //; 使用计算出来的增量更新位姿、广度之后，再次计算能量
				Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH * levelCutoffRepeat);

                //; 0是总能量值，1是构成总能量的点的个数，相除就是平均能量
				bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]); // 平均能量值小则接受

				if (debugPrint)
				{
					Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_new).cast<float>();
					printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
						   lvl, iteration, lambda,
						   extrapFac,
						   (accept ? "ACCEPT" : "REJECT"),
						   resOld[0] / resOld[1],
						   resNew[0] / resNew[1],
						   (int)resOld[1], (int)resNew[1],
						   inc.norm());
					std::cout << refToNew_new.log().transpose() << " AFF " << aff_g2l_new.vec().transpose() << " (rel " << relAff.transpose() << ")\n";
				}

				// Step 2.4.3. 接受本次迭代则求正规方程, 继续迭代, 优化到增量足够小
				if (accept)
				{
					calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
					resOld = resNew;
					aff_g2l_current = aff_g2l_new;
					refToNew_current = refToNew_new;
					lambda *= 0.5;
				}
                //; 否则本次迭代失败，根据LM算法需要对lambda*4继续计算一次
				else
				{
					lambda *= 4;
					if (lambda < lambdaExtrapolationLimit)
						lambda = lambdaExtrapolationLimit;
				}

				if (!(inc.norm() > 1e-3))
				{
					if (debugPrint)
						printf("inc too small, break!\n");
					break;
				}
			}

			// Step 2.5. 迭代优化完成，记录上一次残差, 光流指示, 如果调整过阈值则重新计算这一层
            // 优化完成之后查看一下该层最终的标准差，如果标准差大于阈值的1.5倍时，认为该次优化失败了，直接退出。
            // 这里阈值是动态调节的，假设这是对第k+1个运动假设进行优化，0～k次得到的最优运动假设误差为E（阈值就是这个），
            // 那么该次优化的误差如果超过了NE（作者使用N=1.5），那就没必要再优化了，直接用最优的结果就好了，
            // 帮助删除一些不必要的优化时间，但是如果没有超过NE，就认为该运动假设可以继续进行优化
			// set last residual for that level, as well as flow indicators.
			lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1])); // 上一次的残差
			lastFlowIndicators = resOld.segment<3>(2);					//
            //; 如果这一层优化之后的能量值>1.5*设置的最小能量值，那么直接放弃
			if (lastResiduals[lvl] > 1.5 * minResForAbort[lvl])
				return false; //! 如果算出来大于最好的直接放弃

            // 如果该层的初始误差状态并不佳，但是最终的误差确实在NE范围中，那么说明这个初值还有希望，
            // 就再优化一遍，不过这个机会是整个运动假设优化过程中唯一的一次机会，用掉了就没有了
			if (levelCutoffRepeat > 1 && !haveRepeated)
			{
				lvl++; // 这一层重新算一遍
				haveRepeated = true;
				printf("REPEAT LEVEL!\n");
			}
		}

		// set!
		lastToNew_out = refToNew_current;
		aff_g2l_out = aff_g2l_current;

		//[ ***step 4*** ] 判断优化失败情况
		if ((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2)) || (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
			return false;

		Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_out).cast<float>();

		if ((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5)) || (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
			return false;

		// 固定情况
		if (setting_affineOptModeA < 0)
			aff_g2l_out.a = 0;
		if (setting_affineOptModeB < 0)
			aff_g2l_out.b = 0;

		return true;
	}


	void CoarseTracker::debugPlotIDepthMap(float *minID_pt, float *maxID_pt, std::vector<IOWrap::Output3DWrapper *> &wraps)
	{
		if (w[1] == 0)
			return;

		int lvl = 0;

		{
			std::vector<float> allID;
			for (int i = 0; i < h[lvl] * w[lvl]; i++)
			{
				if (idepth[lvl][i] > 0)
					allID.push_back(idepth[lvl][i]);
			}
			std::sort(allID.begin(), allID.end());
			int n = allID.size() - 1;

			float minID_new = allID[(int)(n * 0.05)];
			float maxID_new = allID[(int)(n * 0.95)];

			float minID, maxID;
			minID = minID_new;
			maxID = maxID_new;
			if (minID_pt != 0 && maxID_pt != 0)
			{
				if (*minID_pt < 0 || *maxID_pt < 0)
				{
					*maxID_pt = maxID;
					*minID_pt = minID;
				}
				else
				{

					// slowly adapt: change by maximum 10% of old span.
					float maxChange = 0.3 * (*maxID_pt - *minID_pt);

					if (minID < *minID_pt - maxChange)
						minID = *minID_pt - maxChange;
					if (minID > *minID_pt + maxChange)
						minID = *minID_pt + maxChange;

					if (maxID < *maxID_pt - maxChange)
						maxID = *maxID_pt - maxChange;
					if (maxID > *maxID_pt + maxChange)
						maxID = *maxID_pt + maxChange;

					*maxID_pt = maxID;
					*minID_pt = minID;
				}
			}

			MinimalImageB3 mf(w[lvl], h[lvl]);
			mf.setBlack();
			for (int i = 0; i < h[lvl] * w[lvl]; i++)
			{
				int c = lastRef->dIp[lvl][i][0] * 0.9f;
				if (c > 255)
					c = 255;
				mf.at(i) = Vec3b(c, c, c);
			}
			int wl = w[lvl];
			for (int y = 3; y < h[lvl] - 3; y++)
				for (int x = 3; x < wl - 3; x++)
				{
					int idx = x + y * wl;
					float sid = 0, nid = 0;
					float *bp = idepth[lvl] + idx;

					if (bp[0] > 0)
					{
						sid += bp[0];
						nid++;
					}
					if (bp[1] > 0)
					{
						sid += bp[1];
						nid++;
					}
					if (bp[-1] > 0)
					{
						sid += bp[-1];
						nid++;
					}
					if (bp[wl] > 0)
					{
						sid += bp[wl];
						nid++;
					}
					if (bp[-wl] > 0)
					{
						sid += bp[-wl];
						nid++;
					}

					if (bp[0] > 0 || nid >= 3)
					{
						float id = ((sid / nid) - minID) / ((maxID - minID));
						mf.setPixelCirc(x, y, makeJet3B(id));
						//mf.at(idx) = makeJet3B(id);
					}
				}
			//IOWrap::displayImage("coarseDepth LVL0", &mf, false);

			for (IOWrap::Output3DWrapper *ow : wraps)
				ow->pushDepthImage(&mf);

			if (debugSaveImages)
			{
				char buf[1000];
				snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png", lastRef->shell->id, refFrameID);
				IOWrap::writeImage(buf, &mf);
			}
		}
	}

	void CoarseTracker::debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper *> &wraps)
	{
		if (w[1] == 0)
			return;
		int lvl = 0;
		MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
		for (IOWrap::Output3DWrapper *ow : wraps)
			ow->pushDepthImageFloat(&mim, lastRef);
	}

	CoarseDistanceMap::CoarseDistanceMap(int ww, int hh)
	{
		//* 在第一层上算的, 所以除4
		fwdWarpedIDDistFinal = new float[ww * hh / 4];

		bfsList1 = new Eigen::Vector2i[ww * hh / 4];
		bfsList2 = new Eigen::Vector2i[ww * hh / 4];

		int fac = 1 << (pyrLevelsUsed - 1);

		coarseProjectionGrid = new PointFrameResidual *[2048 * (ww * hh / (fac * fac))];
		coarseProjectionGridNum = new int[ww * hh / (fac * fac)];

		w[0] = h[0] = 0;
	}
	CoarseDistanceMap::~CoarseDistanceMap()
	{
		delete[] fwdWarpedIDDistFinal;
		delete[] bfsList1;
		delete[] bfsList2;
		delete[] coarseProjectionGrid;
		delete[] coarseProjectionGridNum;
	}

	//@ 对于目前所有的地图点投影, 生成距离场图
	void CoarseDistanceMap::makeDistanceMap(
		std::vector<FrameHessian *> frameHessians,
		FrameHessian *frame)
	{
		int w1 = w[1]; //? 为啥使用第一层的
		int h1 = h[1];
		int wh1 = w1 * h1;
		for (int i = 0; i < wh1; i++)
			fwdWarpedIDDistFinal[i] = 1000;

		// make coarse tracking templates for latstRef.
		int numItems = 0;

		for (FrameHessian *fh : frameHessians)
		{
			if (frame == fh)
				continue;

			SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
			Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]); // 0层到1层变换
			Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

			for (PointHessian *ph : fh->pointHessians)
			{
				assert(ph->status == PointHessian::ACTIVE);
				Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * ph->idepth_scaled; // 投影到frame帧
				int u = ptp[0] / ptp[2] + 0.5f;
				int v = ptp[1] / ptp[2] + 0.5f;
				if (!(u > 0 && v > 0 && u < w[1] && v < h[1]))
					continue;
				fwdWarpedIDDistFinal[u + w1 * v] = 0;
				bfsList1[numItems] = Eigen::Vector2i(u, v);
				numItems++;
			}
		}

		growDistBFS(numItems);
	}

	void CoarseDistanceMap::makeInlierVotes(std::vector<FrameHessian *> frameHessians)
	{
	}

	//@ 生成每一层的距离, 第一层为1, 第二层为2....
	void CoarseDistanceMap::growDistBFS(int bfsNum)
	{
		assert(w[0] != 0);
		int w1 = w[1], h1 = h[1];
		for (int k = 1; k < 40; k++)
		{
			int bfsNum2 = bfsNum;
			//* 每一次都是在上一次的点周围找
			std::swap<Eigen::Vector2i *>(bfsList1, bfsList2); // 每次迭代一遍就交换
			bfsNum = 0;

			if (k % 2 == 0) // 偶数
			{
				for (int i = 0; i < bfsNum2; i++)
				{
					int x = bfsList2[i][0];
					int y = bfsList2[i][1];
					if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1)
						continue;
					int idx = x + y * w1;

					//* 右边
					if (fwdWarpedIDDistFinal[idx + 1] > k) // 没有赋值的位置
					{
						fwdWarpedIDDistFinal[idx + 1] = k; // 赋值为2, 4, 6 ....
						bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
						bfsNum++;
					}
					//* 左边
					if (fwdWarpedIDDistFinal[idx - 1] > k)
					{
						fwdWarpedIDDistFinal[idx - 1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
						bfsNum++;
					}
					//* 下边
					if (fwdWarpedIDDistFinal[idx + w1] > k)
					{
						fwdWarpedIDDistFinal[idx + w1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
						bfsNum++;
					}
					//* 上边
					if (fwdWarpedIDDistFinal[idx - w1] > k)
					{
						fwdWarpedIDDistFinal[idx - w1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
						bfsNum++;
					}
				}
			}
			else
			{
				for (int i = 0; i < bfsNum2; i++)
				{
					int x = bfsList2[i][0];
					int y = bfsList2[i][1];
					if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1)
						continue;
					int idx = x + y * w1;
					//* 上下左右
					if (fwdWarpedIDDistFinal[idx + 1] > k)
					{
						fwdWarpedIDDistFinal[idx + 1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
						bfsNum++;
					}
					if (fwdWarpedIDDistFinal[idx - 1] > k)
					{
						fwdWarpedIDDistFinal[idx - 1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
						bfsNum++;
					}
					if (fwdWarpedIDDistFinal[idx + w1] > k)
					{
						fwdWarpedIDDistFinal[idx + w1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
						bfsNum++;
					}
					if (fwdWarpedIDDistFinal[idx - w1] > k)
					{
						fwdWarpedIDDistFinal[idx - w1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
						bfsNum++;
					}

					//* 四个角
					if (fwdWarpedIDDistFinal[idx + 1 + w1] > k)
					{
						fwdWarpedIDDistFinal[idx + 1 + w1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y + 1);
						bfsNum++;
					}
					if (fwdWarpedIDDistFinal[idx - 1 + w1] > k)
					{
						fwdWarpedIDDistFinal[idx - 1 + w1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y + 1);
						bfsNum++;
					}
					if (fwdWarpedIDDistFinal[idx - 1 - w1] > k)
					{
						fwdWarpedIDDistFinal[idx - 1 - w1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y - 1);
						bfsNum++;
					}
					if (fwdWarpedIDDistFinal[idx + 1 - w1] > k)
					{
						fwdWarpedIDDistFinal[idx + 1 - w1] = k;
						bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y - 1);
						bfsNum++;
					}
				}
			}
		}
	}

	//@ 在点(u, v)附近生成距离场
	void CoarseDistanceMap::addIntoDistFinal(int u, int v)
	{
		if (w[0] == 0)
			return;
		bfsList1[0] = Eigen::Vector2i(u, v);
		fwdWarpedIDDistFinal[u + w[1] * v] = 0;
		growDistBFS(1);
	}

	void CoarseDistanceMap::makeK(CalibHessian *HCalib)
	{
		w[0] = wG[0];
		h[0] = hG[0];

		fx[0] = HCalib->fxl();
		fy[0] = HCalib->fyl();
		cx[0] = HCalib->cxl();
		cy[0] = HCalib->cyl();

		for (int level = 1; level < pyrLevelsUsed; ++level)
		{
			w[level] = w[0] >> level;
			h[level] = h[0] >> level;
			fx[level] = fx[level - 1] * 0.5;
			fy[level] = fy[level - 1] * 0.5;
			cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
			cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
		}

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
}
