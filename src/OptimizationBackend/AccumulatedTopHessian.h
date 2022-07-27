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

#include "util/NumType.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "vector"
#include <math.h>
#include "util/IndexThreadReduce.h"

namespace dso
{

	class EFPoint;
	class EnergyFunctional;

	class AccumulatedTopHessianSSE
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		inline AccumulatedTopHessianSSE()
		{
			for (int tid = 0; tid < NUM_THREADS; tid++)
			{
				nres[tid] = 0;
				acc[tid] = 0;
				nframes[tid] = 0;
			}
		};
		inline ~AccumulatedTopHessianSSE()
		{
			for (int tid = 0; tid < NUM_THREADS; tid++)
			{
				if (acc[tid] != 0)
					delete[] acc[tid];
			}
		};
		//@ 初始化
		inline void setZero(int nFrames, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0)
		{
			//? 什么情况下不等
			if (nFrames != nframes[tid])
			{
				if (acc[tid] != 0)
					delete[] acc[tid];
#if USE_XI_MODEL
				acc[tid] = new Accumulator14[nFrames * nFrames];
#else
				acc[tid] = new AccumulatorApprox[nFrames * nFrames];
#endif
			}

			// 初始化, 设置初值
			for (int i = 0; i < nFrames * nFrames; i++)
			{
                //! 重要：第tid(默认0)个线程中开辟8*8个累加器，用来计算和存储13x13的hessian和b
				acc[tid][i].initialize();
			}

			nframes[tid] = nFrames;
			nres[tid] = 0;
		}
		void stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior, bool useDelta, int tid = 0);

		template <int mode>
		void addPoint(EFPoint *p, EnergyFunctional const *const ef, int tid = 0);


		//@ 获得最终的 H 和 b
        /**
         * @brief 把上一步构造的所有的8*8的小的hessian累加起来，构造成最后整个大的hessian
         *        此外还要注意在这个累加过程中会把相对状态的小hessian转化成绝对状态的大hessian
         * @param[in] red 
         * @param[in] H 
         * @param[in] b 
         * @param[in] EF 能量函数
         * @param[in] usePrior true表示把最后的大的hessian矩阵加上先验
         * @param[in] MT false
         */
		void stitchDoubleMT(IndexThreadReduce<Vec10> *red, MatXX &H, VecX &b, 
            EnergyFunctional const *const EF, bool usePrior, bool MT)
		{
			// sum up, splitting by bock in square.
			if (MT)
			{
				MatXX Hs[NUM_THREADS];
				VecX bs[NUM_THREADS];
				for (int i = 0; i < NUM_THREADS; i++)
				{
					assert(nframes[0] == nframes[i]);
					//* 所有的优化变量维度
					Hs[i] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
					bs[i] = VecX::Zero(nframes[0] * 8 + CPARS);  
				}

				red->reduce(boost::bind(&AccumulatedTopHessianSSE::stitchDoubleInternal,
										this, Hs, bs, EF, usePrior, _1, _2, _3, _4),
							0, nframes[0] * nframes[0], 0);

				// sum up results
				H = Hs[0];
				b = bs[0];
				//* 所有线程求和
				for (int i = 1; i < NUM_THREADS; i++)
				{
					H.noalias() += Hs[i];
					b.noalias() += bs[i];
					nres[0] += nres[i];
				}
			}
			else // 默认不使用多线程
			{
                //; 整个大的H和b的维度，假设是8个关键帧，每个关键帧是6个位姿+2个光度，即8个参数，整个滑窗还维护统一的4个内参
                //; 所以状态变量维度是8 * 8 + 4 = 68维
				H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
				b = VecX::Zero(nframes[0] * 8 + CPARS);  //; 注意b = J'*e = 68xn * nx1 = 68x1，n是残差个数
                // Step 1 调用函数，把小hessian拼接成大的hessian，同时把小hessian中的相对状态转成绝对状态
				stitchDoubleInternal(&H, &b, EF, usePrior, 0, nframes[0] * nframes[0], 0, -1);
			}

            // Step 2 沿着对角线复制
			// make diagonal by copying over parts.
			for (int h = 0; h < nframes[0]; h++)
			{
				int hIdx = CPARS + h * 8;  

                // Step 2.1 先拷贝内参位置的部分，也就是上面4行和左边4列是对称的，但是在上面拼接的时候
                //    Step  是给左边四列赋值，所以这里要把左边四列赋值给上边四行
                //; noalias声明没有混淆，否则可能会出现赋值错误的情况
                // [内参, 位姿] 对称部分
				H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose(); 

                // Step 2.2 然后以当前帧为host帧，它后面的帧为target帧，拷贝相机状态(位姿+光度)部分
				for (int t = h + 1; t < nframes[0]; t++)
				{
					int tIdx = CPARS + t * 8;
					// 对于位姿, 相同两帧之间的Hessian需要加起来, 即对称位置的, (J差负号, J'J平方之后负号就消掉了)
					H.block<8, 8>(hIdx, tIdx).noalias() += H.block<8, 8>(tIdx, hIdx).transpose();
                    //; 注意上面已经加完了，所以和它对称的位置直接赋值就可以了
					H.block<8, 8>(tIdx, hIdx).noalias() = H.block<8, 8>(hIdx, tIdx).transpose();
				}
			}
		}


		int nframes[NUM_THREADS]; //< 每个线程的帧数

		EIGEN_ALIGN16 AccumulatorApprox *acc[NUM_THREADS]; //< 计算hessian的累乘器

		int nres[NUM_THREADS]; //< 残差计数

		template <int mode>
		void addPointsInternal(
			std::vector<EFPoint *> *points, EnergyFunctional const *const ef,
			int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0)
		{
			for (int i = min; i < max; i++)
            {
				addPoint<mode>((*points)[i], ef, tid);
            }
		}

	private:
		void stitchDoubleInternal(
			MatXX *H, VecX *b, EnergyFunctional const *const EF, bool usePrior,
			int min, int max, Vec10 *stats, int tid);
	};
}
