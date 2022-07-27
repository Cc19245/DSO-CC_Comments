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
#include "util/IndexThreadReduce.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "vector"
#include <math.h>

namespace dso
{

	class EFPoint;
	class EnergyFunctional;

	class AccumulatedSCHessianSSE
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		inline AccumulatedSCHessianSSE()
		{
			for (int i = 0; i < NUM_THREADS; i++) // 多线程
			{
				accE[i] = 0;
				accEB[i] = 0;
				accD[i] = 0;
				nframes[i] = 0;
			}
		};
		inline ~AccumulatedSCHessianSSE()
		{
			for (int i = 0; i < NUM_THREADS; i++)
			{
				if (accE[i] != 0)
					delete[] accE[i];
				if (accEB[i] != 0)
					delete[] accEB[i];
				if (accD[i] != 0)
					delete[] accD[i];
			}
		};

		inline void setZero(int n, int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0)
		{
            //; n是传入的数组个数，舒尔消元在计算正常的H之后紧接着进行，所以这里不会满足这个条件
			if (n != nframes[tid]) // 如果帧数有变化
			{
				if (accE[tid] != 0)
					delete[] accE[tid];
				if (accEB[tid] != 0)
					delete[] accEB[tid];
				if (accD[tid] != 0)
					delete[] accD[tid];
				// 这三数组
				accE[tid] = new AccumulatorXX<8, CPARS>[n * n];
				accEB[tid] = new AccumulatorX<8>[n * n];
				accD[tid] = new AccumulatorXX<8, 8>[n * n * n];
			}
			accbc[tid].initialize();
			accHcc[tid].initialize();

			for (int i = 0; i < n * n; i++)
			{
				accE[tid][i].initialize();
				accEB[tid][i].initialize();

				for (int j = 0; j < n; j++)
					accD[tid][i * n + j].initialize();
			}
			nframes[tid] = n;
		}
		void stitchDouble(MatXX &H_sc, VecX &b_sc, EnergyFunctional const *const EF, int tid = 0);
		void addPoint(EFPoint *p, bool shiftPriorToZero, int tid = 0);


		//@ 多线程得到Schur complement
        /**
         * @brief 得到最后的舒尔补结果
         * 
         * @param[in] red 
         * @param[in] H 
         * @param[in] b 
         * @param[in] EF 
         * @param[in] MT 
         */
		void stitchDoubleMT(IndexThreadReduce<Vec10> *red, MatXX &H, VecX &b, EnergyFunctional const *const EF, bool MT)
		{
			// sum up, splitting by bock in square.
			if (MT)
			{
				MatXX Hs[NUM_THREADS];
				VecX bs[NUM_THREADS];
				// 分配空间大小
				for (int i = 0; i < NUM_THREADS; i++)
				{
					assert(nframes[0] == nframes[i]);
					Hs[i] = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
					bs[i] = VecX::Zero(nframes[0] * 8 + CPARS);
				}

				red->reduce(boost::bind(&AccumulatedSCHessianSSE::stitchDoubleInternal,
										this, Hs, bs, EF, _1, _2, _3, _4),
							0, nframes[0] * nframes[0], 0);

				// sum up results
				H = Hs[0];
				b = bs[0];

				for (int i = 1; i < NUM_THREADS; i++)
				{
					H.noalias() += Hs[i];
					b.noalias() += bs[i];
				}
			}
            //; 默认配置走这个，不是多线程
			else
			{
                //; H：68x68, b：68x1
				H = MatXX::Zero(nframes[0] * 8 + CPARS, nframes[0] * 8 + CPARS);
				b = VecX::Zero(nframes[0] * 8 + CPARS);
                // Step 1 优化信息统计，就是把在8*8的小的hessian中计算的舒尔补，拼凑成最后整个大的舒尔补的H和b,
                //   Step 同时在内部还把之前小的hessian中的相对状态转成了绝对状态
				stitchDoubleInternal(&H, &b, EF, 0, nframes[0] * nframes[0], 0, -1);
			}

            // Step 2 同正常计算的H一样，这里也要沿着对角线拷贝
			//* 对称部分
			// make diagonal by copying over parts.
			for (int h = 0; h < nframes[0]; h++)
			{
                // Step 2.1 先拷贝了相机内参所在的上面4行和左边4列
				int hIdx = CPARS + h * 8;
				H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();

                // Step 2.2 对于正常计算的H来说，这里还需要把H的相机位姿部分沿着对角线相加，然后再拷贝，让H是一个
                //  Step 对称矩阵。但是这里就没有了，是不是因为和它内部计算accD的时候有两个循环有关？
			}
		}

		AccumulatorXX<8, CPARS> *accE[NUM_THREADS]; //!< 位姿和内参关于逆深度的 Schur
		AccumulatorX<8> *accEB[NUM_THREADS];		//!< 位姿光度关于逆深度的 b*Schur
        //; 指针数组，相当于二维数组了，第1个维度是NUM_THREADS个线程的数组
        //; 第2个维度就是每个线程的数组
		AccumulatorXX<8, 8> *accD[NUM_THREADS];		//!< 两位姿光度关于逆深度的 Schur

		AccumulatorXX<CPARS, CPARS> accHcc[NUM_THREADS]; //!< 内参关于逆深度的 Schur
		AccumulatorX<CPARS> accbc[NUM_THREADS];			 //!< 内参关于逆深度的 b*Schur
		int nframes[NUM_THREADS];

		void addPointsInternal(
			std::vector<EFPoint *> *points, bool shiftPriorToZero,
			int min = 0, int max = 1, Vec10 *stats = 0, int tid = 0)
		{
			for (int i = min; i < max; i++)
				addPoint((*points)[i], shiftPriorToZero, tid);
		}

	private:
		void stitchDoubleInternal(
			MatXX *H, VecX *b, EnergyFunctional const *const EF,
			int min, int max, Vec10 *stats, int tid);
	};

}
