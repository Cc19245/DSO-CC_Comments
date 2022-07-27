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
     *   注意：这里面仍然是对每个残差构成的小的13x13的hessian进行舒尔补，然后再把这些小的hessian拼接成
     *         绝对状态的他的hessian，而不是直接在大的hessian上进行舒尔补
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
        //; Hdd_accAF：正常计算的本次逆深度hessian，它是在计算每个点的每个残差的hessian的时候累加的
        //; Hdd_accLF：上次margin边缘化的逆深度hessian，但是前面调用accumulateLF_MT，在里面没有实质操作，会把它设置成0
        //; priorF：先验的逆深度，只有整个系统的第一帧有
        //! 7.27增：点的逆深度hessian是在前面调用accumulateAF_MT时，累加它构成的所有残差的hessian得到的，为什么可以累加？
        //;  解答：尽管这个点包含的所有残差和host/target帧有关，也就是accumulateAF_MT里对同一个点的不同残差得到的
        //;     13x13的hessian，是和不同的帧有关的。但是他们的逆深度部分都是关于这一个点的，也就是行、列均为逆深度
        //;     的位置，即右下角(13, 13)位置都是(dr/rρ)' * (dr/dρ)，因此最后把所有小的hessian拼接起来的时候他们是可以相加的
		float H = p->Hdd_accAF + p->Hdd_accLF + p->priorF;
		if (H < 1e-10)
        {
			H = 1e-10;
        }
        p->data->idepth_hessian = H;
		p->HdiF = 1.0 / H;  //; 逆深度hessian的逆，所以这里取倒数

		//* 逆深度残差
        //; 1.这个应该就是深蓝PPT P36的舒尔补bd部分，也就是J'*e对应的这个点的残差那一行，是1x1的
        //!  7.26增：不是！这里其实是在算正规方程中b的部分关于逆深度的那个位置。PPT中P36的都是关于相机状态的，没有逆深度
        //; 2.bd_accAF: 正常计算的本次逆深度的b部分
        //;   bd_accLF：边缘化的逆深度b部分，同理前面调用accumulateLF_MY，在里面没有实质操作，会把它设置成0
		p->bdSumF = p->bd_accAF + p->bd_accLF;

        //; shiftPriorToZero调用传入的时候是true
		if (shiftPriorToZero)
        {
            //; 还加上逆深度的先验部分，很简单，就是b = J'*e
			p->bdSumF += p->priorF * p->deltaF;
        }

		//* 逆深度和内参的交叉项
        //; 1. 这里注意，在之前计算H中和逆深度同一行的W'(见深蓝PPT P12关于H矩阵的划分)的时候，只计算了内参部分
        //;    我感觉可能是因为那里计算的相机状态的H部分是相对状态量，而这里是绝对状态量，所以在那里计算无意义？
        //!  7.26增：不是的！
        //;     本质上还是因为所有帧都共享同一个相机内参，所以在之前计算H的时候直接就累加相机内参的舒尔补
        //;     是可行的。如果不累加那个舒尔补，在这里再次遍历点的所有残差的时候再累加也行，但是这样就需要
        //;     8*8的hessian中每个hessian都要维护相机内参的舒尔补，浪费空间
        //; 2. 同理，Hcd_accAF是active状态正常计算的；Hcd_accLF是调用accumulateLF_MY，在里面没有实质操作，会把它设置成0
		VecCf Hcd = p->Hcd_accAF + p->Hcd_accLF;

        // Step 执行舒尔补
		//* schur complement
        // Step 1 对相机内参部分舒尔补
        //   Step 1.1 当前这个点的逆深度对H中  (相机内参, 相机内参)  位置的舒尔补，对应深蓝学院PPT P36上面的accHc
        // Hcd * Hdd_inv * Hcd^T
		accHcc[tid].update(Hcd, Hcd, p->HdiF);
        //   Step 1.2 当前这个点的逆深度对b中  (相机内参, 0)  位置的舒尔补，对应深蓝学院PPT P36上面的accbc
        //;        注意：这里的列数位置一直是0，因为b部分是一个列向量
		// Hcd * Hdd_inv * bd
		accbc[tid].update(Hcd, p->bdSumF * p->HdiF);

		assert(std::isfinite((float)(p->HdiF)));

        // Step 2 当前这个点的逆深度对H和b中相机位姿、光度位置的舒尔补，对应深蓝学院PPT P36上面的accD accE accEB
        //; 总的帧数，正常运行状态下应该是8*8
		int nFrames2 = nframes[tid] * nframes[tid];
        //; 遍历当前点构成的所有残差，其实也就是当前点的host帧和可能的滑窗中其他关键帧之间构成的残差
		for (EFResidual *r1 : p->residualsAll)
		{
			if (!r1->isActive())
            {
				continue;
            }
            //; 这个残差在整个8*8的小hessian数组中的索引
            int r1ht = r1->hostIDX + r1->targetIDX * nframes[tid];

            // Step 3.1 当前这个点的逆深度对8*8的hessian中相应的构成这个残差的那个hessian的
            //   Step   (相机状态(包括位姿和光度), 相机状态(包括位姿和光度))  位置的舒尔补
            //! 疑问：这个地方属实没有怎么看懂，为什么同一个点下的residual要遍历两次？
			for (EFResidual *r2 : p->residualsAll)
			{
				if (!r2->isActive())
                {
					continue;
                }
                // Hfd_1 * Hdd_inv * Hfd_2^T,  f = [xi, a b]位姿 光度
                //; r1ht表示以当前残差为参照，后面的附加项表示其他所有残差(包括当前残差)，就相当于第1行的所有列
				accD[tid][r1ht + r2->targetIDX * nFrames2].update(r1->JpJdF, r2->JpJdF, p->HdiF);
			}

            // Step 3.2 当前这个点的逆深度对8*8的hessian中相应的构成这个残差的那个hessian的
            //   Step   (相机状态(包括位姿和光度), 相机内参)  位置的舒尔补
            //; 在accE[r1ht]中加上了针对当前residual（target, host）的Hfd * Hdd_inv * Hcd^T部分，和深蓝PPT可以对应
			// Hfd * Hdd_inv * Hcd^T
            //; 注意这里的JpJdF的前6维就是Hfd，即Hfd = JpJdF[0:6]
			accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);
            
            // Step 3.3 当前这个点的逆深度对8*8的hessian中相应的构成这个残差的那个b的
            //   Step   (相机状态(包括位姿和光度), 相机内参)  位置的舒尔补
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

		int nf = nframes[0];       // 8
		int nframes2 = nf * nf;    // 8*8 = 64

        //; 遍历所有的8*8帧
		for (int k = min; k < max; k++)
		{
			int i = k % nf;   //; i是host帧索引
			int j = k / nf;   //; j是target帧索引

            //; 根据host和target索引，找到他们在最后的H和b中的位置
			int iIdx = CPARS + i * 8;
			int jIdx = CPARS + j * 8;
			int ijIdx = i + nf * j;  // 在整个8*8中的位置，实际就是k，不知道为啥要重新算一遍

            //; 这两个就是深蓝PPT P36对应的accE和accEB部分，即(相机状态, 相机内参) 位置的舒尔补
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

            //; Hfc部分，即在H中(相机状态(位姿、光度)，相机内参) 位置的舒尔补，注意相对H转成绝对H
			H[tid].block<8, CPARS>(iIdx, 0) += EF->adHost[ijIdx] * Hpc;
			H[tid].block<8, CPARS>(jIdx, 0) += EF->adTarget[ijIdx] * Hpc;

            //; bf部分，即在b中(相机状态(位姿、光度)，相机内参) 位置的舒尔补，注意相对b转成绝对b
			b[tid].segment<8>(iIdx) += EF->adHost[ijIdx] * bp;
			b[tid].segment<8>(jIdx) += EF->adTarget[ijIdx] * bp;

            //! 疑问：这个循环就是在计算Hff部分的舒尔补，和计算accD的时候一样，也是需要累加，这里还是没有弄明白
			for (int k = 0; k < nf; k++)
			{
				int kIdx = CPARS + k * 8;  //; k是其他的任意一帧
                //; ijIdx是host和target构成的相对H在8*8中的索引，再+k干啥？这里还是不太懂啊...
				int ijkIdx = ijIdx + k * nframes2;  
				int ikIdx = i + nf * k;  //; i是host帧

				Mat88 accDM = Mat88::Zero();

				for (int tid2 = 0; tid2 < toAggregate; tid2++)
				{
					accD[tid2][ijkIdx].finish();
					if (accD[tid2][ijkIdx].num == 0)
						continue;
					accDM += accD[tid2][ijkIdx].A1m.cast<double>();
				}
				// Hff部分Schur
                //; (host, host)
				H[tid].block<8, 8>(iIdx, iIdx) += EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
				//; (target, 所有列)
                H[tid].block<8, 8>(jIdx, kIdx) += EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
				//; (target, host)
                H[tid].block<8, 8>(jIdx, iIdx) += EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
                //; (host, 所有列)
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
                //; 注意相机内参部分舒尔补之前算的时候就不是在8*8的每个小的hessian中算的，而是直接累加了总的，
                //; 所以这里直接加到总的H上即可
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
