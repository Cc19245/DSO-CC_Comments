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

#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso
{
    /**
     * @brief 对逆深度执行舒尔补，为后面求解相机位姿+光度参数的正规方程做最后的准备
     *  参考博客：https://www.cnblogs.com/JingeTU/p/8586172.html
     * 
     * @param[in] p 
     * @param[in] shiftPriorToZero 
     * @param[in] tid 多线程？形参默认值是0
     */
	void AccumulatedSCHessianSSE::addPoint(EFPoint *p, bool shiftPriorToZero, int tid)
	{
		int ngoodres = 0;
		for (EFResidual *r : p->residualsAll)
        {
			if (r->isActive())
            {
				ngoodres++;
            }
        }

        //; 如果没有active状态的残差点，那么这里设置一下变量直接返回
        //; 不知道为什么要统计这个，感觉正常情况下，这个是一定不会满足的
		if (ngoodres == 0)
		{
            //; 这里HdiF就是在正常计算hessian的时候，最后计算的hessian中关于行列都是逆深度的部分
			p->HdiF = 0;
			p->bdSumF = 0;
			p->data->idepth_hessian = 0;
			p->data->maxRelBaseline = 0;
			return;
		}

		//* hessian + 边缘化得到hessian + 先验hessian
        //; Hdd_accAF：正常计算的本次逆深度hessian;
        //; Hdd_accLF：边缘化的逆深度hessian，但是前面调用accumulateLF_MT，会把它设置成0
        //; priorF：先验的逆深度，只有整个系统的第一帧有
		float H = p->Hdd_accAF + p->Hdd_accLF + p->priorF;
		if (H < 1e-10)
        {
			H = 1e-10;
        }
        p->data->idepth_hessian = H;
		p->HdiF = 1.0 / H;  //; 逆深度hessian的逆，所以这里取倒数

		//* 逆深度残差
        //; 这个应该就是深蓝PPT P36的舒尔补bd部分，也就是J'*e对应的这个点的残差那一行，是1x1的
		p->bdSumF = p->bd_accAF + p->bd_accLF;

        //; shiftPriorToZero调用传入的时候是true
		if (shiftPriorToZero)
        {
            //; 还加上逆深度的先验部分，很简单，就是b = J'*e
			p->bdSumF += p->priorF * p->deltaF;
        }

		//* 逆深度和内参的交叉项
        //; 这个也是深蓝PPT中的
		VecCf Hcd = p->Hcd_accAF + p->Hcd_accLF;

        // Step 2 执行舒尔补
		//* schur complement
        //; 在accHcc中加上了针对当前点的Hcc，就是下面写的公式，也是深蓝学院PPT P36中的Hcc
        // Hcd * Hdd_inv * Hcd^T
		accHcc[tid].update(Hcd, Hcd, p->HdiF);
        //; 在accbc中加上了针对当前点的bc，也是下面写的公式，一样
		// Hcd * Hdd_inv * bd
		accbc[tid].update(Hcd, p->bdSumF * p->HdiF);

		assert(std::isfinite((float)(p->HdiF)));

        //; 总的帧数，正常运行状态下应该是8*8
		int nFrames2 = nframes[tid] * nframes[tid];
        //; 遍历当前点构成的所有残差
		for (EFResidual *r1 : p->residualsAll)
		{
			if (!r1->isActive())
            {
				continue;
            }
            //; 计算这个是什么？在整个8*8中的索引？
            int r1ht = r1->hostIDX + r1->targetIDX * nframes[tid];

			for (EFResidual *r2 : p->residualsAll)
			{
				if (!r2->isActive())
                {
					continue;
                }
                //! 疑问：这个地方属实没有怎么看懂
                // Hfd_1 * Hdd_inv * Hfd_2^T,  f = [xi, a b]位姿 光度
				accD[tid][r1ht + r2->targetIDX * nFrames2].update(r1->JpJdF, r2->JpJdF, p->HdiF);
			}
            //; 在accE[r1ht]中加上了针对当前residual（target, host）的Hfd * Hdd_inv * Hcd^T部分，和深蓝PPT可以对应
			// Hfd * Hdd_inv * Hcd^T
			accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);
            //; 在accEB中加上了针对当前residual的Hfd * Hdd_inv * bd，和深蓝PPT也能对应
			// Hfd * Hdd_inv * bd
			accEB[tid][r1ht].update(r1->JpJdF, p->HdiF * p->bdSumF);
		}
	}


	//@ 从累加器里面得到 hessian矩阵Schur complement
    /**
     * @brief 从累加器里面把舒尔补部分拿出来
     * 
     * @param[in] H 
     * @param[in] b 
     * @param[in] EF 
     * @param[in] min 
     * @param[in] max 
     * @param[in] stats 
     * @param[in] tid 
     */
	void AccumulatedSCHessianSSE::stitchDoubleInternal(
		MatXX *H, VecX *b, EnergyFunctional const *const EF,
		int min, int max, Vec10 *stats, int tid)
	{
		int toAggregate = NUM_THREADS;
		if (tid == -1)
		{
			toAggregate = 1;
			tid = 0;
		} // special case: if we dont do multithreading, dont aggregate.
		if (min == max)
			return;

		int nf = nframes[0];
		int nframes2 = nf * nf;

        //; 遍历所有的8*8帧
		for (int k = min; k < max; k++)
		{
			int i = k % nf;
			int j = k / nf;

			int iIdx = CPARS + i * 8;
			int jIdx = CPARS + j * 8;
			int ijIdx = i + nf * j;

			Mat8C Hpc = Mat8C::Zero();  // 8x4
			Vec8 bp = Vec8::Zero();     // 8x1

			//* 所有线程求和
			for (int tid2 = 0; tid2 < toAggregate; tid2++)
			{
				accE[tid2][ijIdx].finish();
				accEB[tid2][ijIdx].finish();
				Hpc += accE[tid2][ijIdx].A1m.cast<double>();
				bp += accEB[tid2][ijIdx].A1m.cast<double>();
			}
			// Hfc部分Schur
			H[tid].block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * Hpc;
			H[tid].block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * Hpc;
			// 位姿,光度部分的残差Schur
			b[tid].segment<8>(iIdx) += EF->adHost[ijIdx] * bp;
			b[tid].segment<8>(jIdx) += EF->adTarget[ijIdx] * bp;

            //; 这个循环就是在计算Hff部分的舒尔补，和计算accD的时候一样，也是需要累加
			for (int k = 0; k < nf; k++)
			{
				int kIdx = CPARS + k * 8;
				int ijkIdx = ijIdx + k * nframes2;
				int ikIdx = i + nf * k;

				Mat88 accDM = Mat88::Zero();

				for (int tid2 = 0; tid2 < toAggregate; tid2++)
				{
					accD[tid2][ijkIdx].finish();
					if (accD[tid2][ijkIdx].num == 0)
						continue;
					accDM += accD[tid2][ijkIdx].A1m.cast<double>();
				}
				// Hff部分Schur
				H[tid].block<8, 8>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
				H[tid].block<8, 8>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
				H[tid].block<8, 8>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
				H[tid].block<8, 8>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
			}
		}

		if (min == 0)
		{
            //; 相机内参部分的舒尔补
			for (int tid2 = 0; tid2 < toAggregate; tid2++)
			{
				accHcc[tid2].finish();
				accbc[tid2].finish();
				// Hcc 部分Schur
				H[tid].topLeftCorner<CPARS, CPARS>() += accHcc[tid2].A1m.cast<double>();
				// 内参部分的残差Schur
				b[tid].head<CPARS>() += accbc[tid2].A1m.cast<double>();
			}
		}
	}


	//@ 对单独某一线程进行计算Schur H b
	void AccumulatedSCHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF, int tid)
	{

		int nf = nframes[0];
		int nframes2 = nf * nf;

		H = MatXX::Zero(nf * 8 + CPARS, nf * 8 + CPARS);
		b = VecX::Zero(nf * 8 + CPARS);

		for (int i = 0; i < nf; i++)
			for (int j = 0; j < nf; j++)
			{
				int iIdx = CPARS + i * 8;
				int jIdx = CPARS + j * 8;
				int ijIdx = i + nf * j;

				accE[tid][ijIdx].finish();
				accEB[tid][ijIdx].finish();

				Mat8C accEM = accE[tid][ijIdx].A1m.cast<double>();
				Vec8 accEBV = accEB[tid][ijIdx].A1m.cast<double>();

				H.block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * accEM;
				H.block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * accEM;

				b.segment<8>(iIdx) += EF->adHost[ijIdx] * accEBV;
				b.segment<8>(jIdx) += EF->adTarget[ijIdx] * accEBV;

				for (int k = 0; k < nf; k++)
				{
					int kIdx = CPARS + k * 8;
					int ijkIdx = ijIdx + k * nframes2;
					int ikIdx = i + nf * k;

					accD[tid][ijkIdx].finish();
					if (accD[tid][ijkIdx].num == 0)
						continue;
					Mat88 accDM = accD[tid][ijkIdx].A1m.cast<double>();

					H.block<8, 8>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();

					H.block<8, 8>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();

					H.block<8, 8>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();

					H.block<8, 8>(iIdx, kIdx) += EF->adHost[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
				}
			}

		accHcc[tid].finish();
		accbc[tid].finish();
		H.topLeftCorner<CPARS, CPARS>() = accHcc[tid].A1m.cast<double>();
		b.head<CPARS>() = accbc[tid].A1m.cast<double>();

		// ----- new: copy transposed parts for calibration only.
		for (int h = 0; h < nf; h++)
		{
			int hIdx = CPARS + h * 8;
			H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();
		}
	}

}
