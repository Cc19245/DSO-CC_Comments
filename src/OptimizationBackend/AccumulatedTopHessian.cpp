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

#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <iostream>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

	//@ 计算残差对应的 Hessian和Jres
    /**
     * @brief  计算每个点和其他帧上的点构成的残差的对应的hessian, 注意一共8*8个hessian,
     *   其内部会判断这个残差是和哪两帧构成的，从而在正确的8*8数组的位置累加计算的hessian
     *  参考：1.涂金戈博客 https://www.cnblogs.com/JingeTU/p/8586163.html
     *         其中一些变量是在之前线性化的时候计算的雅克比，变量定义见涂金戈博客：
     *          https://www.cnblogs.com/JingeTU/p/8395046.html
     *       2.深蓝学院PPT P36
     * @tparam mode 
     * @param[in] p 
     * @param[in] ef 
     * @param[in] tid 线程个数，默认实参=0
     */
	template <int mode>  // 0 = active, 1 = linearized, 2=marginalize
	void AccumulatedTopHessianSSE::addPoint(EFPoint *p, EnergyFunctional const *const ef, int tid) 
	{
		assert(mode == 0 || mode == 1 || mode == 2);

		VecCf dc = ef->cDeltaF; // 内参
		float dd = p->deltaF;	// 逆深度

		float bd_acc = 0;
		float Hdd_acc = 0;
		VecCf Hcd_acc = VecCf::Zero();

        //; 遍历这个点的所有残差
        //! 疑问：这个所有残差是什么意思？是这个点和其他帧上的点构成的残差？那岂不是同一个残差要多算一遍？
        //! 解答：目前感觉是这样的，因为看后面计算hessian的时候就能发现，他存储的hessian是8*8个，重复了一半
        //! 最新解答：上面的 说法是错的。因为一个点只有一个host帧，我们对点计算的残差，都是以host帧上的点为出发点，
        //!      所以不论这个点和其他多少个target帧构成残差，实际上这个残差都只会被计算一次，因为他只有一个host帧
		for (EFResidual *r : p->residualsAll) // 对该点所有残差遍历一遍
		{
            //; 正常计算这次的状态的hessain的时候就是mode=0, 也就是不涉及边缘化、大的先验等其他任何操作
			if (mode == 0) // 只计算新加入的残差
			{
				if (r->isLinearized || !r->isActive())
                {
					continue;
                }
			}
			if (mode == 1) // bug: 这个条件就一直满足 计算旧的残差, 之前计算过得
			{
                //; 这里注意，确实是上面公益群注释的那样。因为mode=1的模式下，比如在计算完正常的Hessian之后
                //; 调用accumulateLF_MT计算线性化的Hessian的时候，传入的mode=1。但是其实由于边缘化的点全部
                //; 被丢掉了，所以这里isLinearized一直为false，也就是下面一直continue，所以就不会计算这个点
                //; 的结果。因此，最后赋值给p->Hdd_accLF、p->bd_accLF = bd_acc、p->Hcd_accLF = Hcd_acc
                //; 全是0
				if (!r->isLinearized || !r->isActive())
                {
                    continue;
                }	
			}
			if (mode == 2) // 边缘化计算的情况
			{
				if (!r->isActive())
					continue;
				assert(r->isLinearized);
			}
			// if(mode == 1)
			// {
			// 	printf("yeah I'm IN !");
			// }

            //; 取出之间计算的雅克比中间量
			RawResidualJacobian *rJ = r->J; // 导数
			//* ID 来控制不同帧之间的变量, 区分出相同两帧 但是host target角色互换的
            //; htIDX就得到了这个残差在acc累加器数组中在那个位置
			int htIDX = r->hostIDX + r->targetIDX * nframes[tid];  //; 注意tid默认实参=0，nframes[0]就是所有关键帧个数
			//; 这个是当前状态相比线性化点的状态所增加的状态量，也是在之前调用 setPrecalcValues 预先计算的
            Mat18f dp = ef->adHTdeltaF[htIDX]; // 位姿+光度a b

            //; VecNRf 8x1向量
			VecNRf resApprox;  // 8x1的残差，就是pattern周围的残差
            //; active活跃点的情况最简单，直接就是之前计算的雅克比
			if (mode == 0)  
				resApprox = rJ->resF;
            //; marginalize边缘化使用的res_toZeroF 在EFResidual::fixLinearizationF()赋值
			if (mode == 2)  
				resApprox = r->res_toZeroF;

            //; linearized线性化的情况，涂金戈的博客(https://www.cnblogs.com/JingeTU/p/8586163.html)
            //; 中详细阐述了，实际上这个if是一定不会满足的，但是博客中他仍然给出了下面代码的解析
            //? 上面的解释不太对，这个mode是可以根据不同的模板值来传入的，所以这里还是可以进入这个if函数的
			if (mode == 1)  
			{
				//* 因为计算的是旧的, 由于更新需要重新计算
				// compute Jp*delta
				__m128 Jp_delta_x = _mm_set1_ps(rJ->Jpdxi[0].dot(dp.head<6>()) + rJ->Jpdc[0].dot(dc) + rJ->Jpdd[0] * dd);
				__m128 Jp_delta_y = _mm_set1_ps(rJ->Jpdxi[1].dot(dp.head<6>()) + rJ->Jpdc[1].dot(dc) + rJ->Jpdd[1] * dd);
				__m128 delta_a = _mm_set1_ps((float)(dp[6]));
				__m128 delta_b = _mm_set1_ps((float)(dp[7]));

				for (int i = 0; i < patternNum; i += 4)
				{
					// PATTERN: rtz = res_toZeroF - [JI*Jp Ja]*delta.
					//* 线性更新b值, 边缘化量, 每次在res_toZeroF上减
					__m128 rtz = _mm_load_ps(((float *)&r->res_toZeroF) + i);
					rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx)) + i), Jp_delta_x));
					rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JIdx + 1)) + i), Jp_delta_y));
					rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF)) + i), delta_a));
					rtz = _mm_add_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(rJ->JabF + 1)) + i), delta_b));
					_mm_store_ps(((float *)&resApprox) + i, rtz);
				}
			}

            // Step 计算Hessian矩阵， 对应涂金戈上述博客，也对应深蓝PPT中P36
            /* 
             * 1.这里的 Hessian 矩阵是存储了两个帧之间的相互信息，所有的信息存储在 AccumulatedTopHessianSSE::acc 
             *   中，acc是一个数组，大小是 8*8 个，位置 (i, j) 上对应的是 i 帧与 j 帧的相互信息。
             * 2.AccumulatorApprox 也就是AccumulatedTopHessianSSE::acc 变量的“基础”类型。
             *   这个类型对应着 13x13 的矩阵
             */
			// need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
			Vec2f JI_r(0, 0);
			Vec2f Jab_r(0, 0);
			float rr = 0;
            // Step 1.1 注意这里把残差乘到了梯度上，是为了计算13x13的hessian的b部分做准备的，也就是最右边一列，
            //?         因为这部分需要乘以r，所以下面乘了resApprox，而resApprox其实就是pattern的8x1残差
			for (int i = 0; i < patternNum; i++)
			{
                //; resApprox是每个点周围8个pattern的残差，8x1向量
                //; 这里涂金戈博客好像说的不对，这里就是2x8的残差对像素坐标 * 8x1的残差，得到残差对像素坐标的雅克比的和
				JI_r[0] += resApprox[i] * rJ->JIdx[0][i];
				JI_r[1] += resApprox[i] * rJ->JIdx[1][i];
                //; 同理这里就是2x8的残差对光度系数 * 8x1的残差，得到残差对光度系数的和
				Jab_r[0] += resApprox[i] * rJ->JabF[0][i];
				Jab_r[1] += resApprox[i] * rJ->JabF[1][i];
                //; 这里就是1x8的残差 * 8x1的残差，得到最后残差的和
				rr += resApprox[i] * resApprox[i];
			}

            // Step 1.2 计算H和b, 在同一个13x13的矩阵中计算
            //; 计算左上角10x10矩阵，传入的是 target帧像素对相机内参偏导、target帧像素对相对位姿偏导、残差对target帧像素偏导
			//* 计算hessian 10*10矩阵, [位姿+相机参数]
			acc[tid][htIDX].update(
                //; Jpdc = dx/dc, 2x4; Jpdxi = dx/dξ, 2x6; JIdx2 = (dr/dx)'*(dr/dx)^2 = 2x2
                //; r残差, x是target帧像素坐标，c相机内参，ξ相机位姿
				rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
				rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
                //; 注意这里传入的残差对像素坐标的偏导，已经是一个pattern内的和了
				rJ->JIdx2(0, 0), rJ->JIdx2(0, 1), rJ->JIdx2(1, 1));
            
            //; 计算右下角3x3矩阵
			//* 计算 3*3 矩阵, [光度a, 光度b, 残差r]
			acc[tid][htIDX].updateBotRight(
                //; Jab2 = (dr/dl)' * (dr/dl), 2x2; Jab_r = (dr/dl)' * r, 2x1; rr=r'*r, 1x1
                //; r残差，l光度系数
				rJ->Jab2(0, 0), rJ->Jab2(0, 1), Jab_r[0],
				rJ->Jab2(1, 1), Jab_r[1], rr);

            //; 计算右上角10x3矩阵
			//* 计算 10*3 矩阵, [位姿+相机参数]*[光度a, 光度b, 残差r]
			acc[tid][htIDX].updateTopRight(
                //; Jpdc = dx/dc, 2x4; Jpdxi = dx/dξ, 2x6; JabJIdx = (dr/dl)' * (dr/dx), 2x2; 
                //; JI_r = (dr/dx)'*r, 2x1
                //; 
				rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
				rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
				rJ->JabJIdx(0, 0), rJ->JabJIdx(0, 1),
				rJ->JabJIdx(1, 0), rJ->JabJIdx(1, 1),
				JI_r[0], JI_r[1]);

            /**
             * 函数的目标除了计算不同帧之间的相互信息（变量acc），还需要计算每一个点对于所有 residual 的信息和。
             * 即EFPoint中的成员变量Hdd_accAF, bd_accAF, Hcd_accAF, Hdd_accLF, bd_accLF, Hcd_accLF，
             * 如果这个点是 active 点，那么设置AF相关的变量，否则设置LF相关变量，如果是 marginalize 点，
             * 清除AF相关变量的信息。这三个成员变量将用于计算逆深度的优化量。局部变量Hdd_acc, bd_acc, 
             * Hcd_acc对应着这些EFPoint的成员变量，最后赋值到成员变量。
             */
            // Step 2 计算后面舒尔补部分要用的，和深蓝PPT P36对应
            //; 7.26增：不是！这个地方和P36非常像，但不是一样的。P36涉及的状态只有位姿、内参、光度，不涉及逆深度，
            //;      因为最后舒尔消元得到的只是这些状态的正规方程而没有逆深度。但是最后求解完相机状态的正规方程
            //;      得到相机状态增量之后，还需要反带到逆深度的方程中求解逆深度，所以总的H和b中和逆深度同一行的
            //;      部分也是需要在构造H和b的时候构造出来的，所以这里就是在构造这部分。
            //?     下面使用 += 就是这个点和其他帧上的点构成的残差，都可以算作这个点的逆深度的hessain部分，所以是+=
			Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;  // 中间变量，无意义
            //; 1x1, (残差 * 残差对像素坐标的雅克比) * 像素坐标对逆深度的雅克比 = 残差 * 残差对逆深度的雅克比
            //;  这个就是b_d，即当前这个点的逆深度部分的b，因为b = J*e, 所以这里算的就是J*e
			bd_acc += JI_r[0] * rJ->Jpdd[0] + JI_r[1] * rJ->Jpdd[1];		  //* 残差*逆深度J
            //; 1x1, 这个实际就是 残差对逆深度雅克比' * 残差对逆深度雅克比，也就是Hessian中行列都是逆深度位置的部分
			Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);								  //* 光度对逆深度hessian
			//; 4x1, 这个确实是 残差对相机内参雅克比' * 残差对逆深度雅克比 = 4x1 * 1x1 = 4x1
            //;  这里其实就在算H中和当前点的逆深度在同一行的那部分。但是由于这里要注意只计算了相机内参的部分，并没有计算
            //;  相机位姿的部分，这是为何？我觉得还是因为这里求得还是相对位姿的雅克比，而最后求解用的是绝对位姿雅克比，
            //;  所以可能这样直接算出来没有意义？
            Hcd_acc += rJ->Jpdc[0] * Ji2_Jpdd[0] + rJ->Jpdc[1] * Ji2_Jpdd[1]; //* 光度对内参J*光度对逆深度J

			nres[tid]++;
		} // 遍历这个点构成的所有残差完毕

        //; active模式，则属于正常的普通状态，所以累加结果存到A相关的变量中
		if (mode == 0)
		{
			p->Hdd_accAF = Hdd_acc;
			p->bd_accAF = bd_acc;
			p->Hcd_accAF = Hcd_acc;
		}
        //; 线性化或者边缘化模式，都会设置L相关的变量
		if (mode == 1 || mode == 2)
		{
            //! 注意：在mode=1的时候，实际上面的遍历没有实际作用，所以这里全是0
			p->Hdd_accLF = Hdd_acc;
			p->bd_accLF = bd_acc;
			p->Hcd_accLF = Hcd_acc;
		}
		if (mode == 2) // 边缘化掉, 设为0
		{
			p->Hcd_accAF.setZero();
			p->Hdd_accAF = 0;
			p->bd_accAF = 0;
		}
	}
	// 实例化
	template void AccumulatedTopHessianSSE::addPoint<0>(EFPoint *p, EnergyFunctional const *const ef, int tid);
	template void AccumulatedTopHessianSSE::addPoint<1>(EFPoint *p, EnergyFunctional const *const ef, int tid);
	template void AccumulatedTopHessianSSE::addPoint<2>(EFPoint *p, EnergyFunctional const *const ef, int tid);


	//@ 对某一个线程进行的 H 和 b 计算, 或者是没有使用多线程
	void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const *const EF, bool usePrior, bool useDelta, int tid)
	{
		H = MatXX::Zero(nframes[tid] * 8 + CPARS, nframes[tid] * 8 + CPARS);
		b = VecX::Zero(nframes[tid] * 8 + CPARS);

		for (int h = 0; h < nframes[tid]; h++)
			for (int t = 0; t < nframes[tid]; t++)
			{
				int hIdx = CPARS + h * 8;
				int tIdx = CPARS + t * 8;
				int aidx = h + nframes[tid] * t;

				acc[tid][aidx].finish();
				if (acc[tid][aidx].num == 0)
					continue;

				MatPCPC accH = acc[tid][aidx].H.cast<double>();

				H.block<8, 8>(hIdx, hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adHost[aidx].transpose();

				H.block<8, 8>(tIdx, tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

				H.block<8, 8>(hIdx, tIdx).noalias() += EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();

				H.block<8, CPARS>(hIdx, 0).noalias() += EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);

				H.block<8, CPARS>(tIdx, 0).noalias() += EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);

				H.topLeftCorner<CPARS, CPARS>().noalias() += accH.block<CPARS, CPARS>(0, 0);

				b.segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);

				b.segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8, 1>(CPARS, 8 + CPARS);

				b.head<CPARS>().noalias() += accH.block<CPARS, 1>(0, 8 + CPARS);
			}

		// ----- new: copy transposed parts.
		for (int h = 0; h < nframes[tid]; h++)
		{
			int hIdx = CPARS + h * 8;
			H.block<CPARS, 8>(0, hIdx).noalias() = H.block<8, CPARS>(hIdx, 0).transpose();

			for (int t = h + 1; t < nframes[tid]; t++)
			{
				int tIdx = CPARS + t * 8;
				H.block<8, 8>(hIdx, tIdx).noalias() += H.block<8, 8>(tIdx, hIdx).transpose();
				H.block<8, 8>(tIdx, hIdx).noalias() = H.block<8, 8>(hIdx, tIdx).transpose();
			}
		}

		if (usePrior)
		{
			assert(useDelta);
			H.diagonal().head<CPARS>() += EF->cPrior;
            //; 这里也可以看出来，cDeltaF是相机当前内参相对先验内参的增量
			b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
			for (int h = 0; h < nframes[tid]; h++)
			{
				H.diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior;
				b.segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
			}
		}
	}


	//@ 构造Hessian矩阵, b=Jres矩阵
    /**
     * @brief 把8*8个小的hessian矩阵，拼接成最后大的hessian矩阵，为后面求解做准备
     * 
     * @param[in] H  输出，拼接完成的H，维度68x68
     * @param[in] b  输出，拼接完成的b，维度68x1
     * @param[in] EF 能量函数对象的指针
     * @param[in] usePrior 是否使用先验信息，调用的时候传入的是true
     * @param[in] min  从最小哪帧开始遍历
     * @param[in] max  遍历截止到哪帧
     * @param[in] stats 
     * @param[in] tid 对于单线程，传入-1
     */
	void AccumulatedTopHessianSSE::stitchDoubleInternal(
		MatXX *H, VecX *b, EnergyFunctional const *const EF, bool usePrior,
		int min, int max, Vec10 *stats, int tid)
	{
		int toAggregate = NUM_THREADS;
		// 不用多线程, 为啥不能统一一下
        //; 如果=-1，说明是单线程
		if (tid == -1)
		{
			toAggregate = 1;
			tid = 0;
		} // special case: if we dont do multithreading, dont aggregate.
		if (min == max)
			return;
        
        // Step 循环是遍历所有可能的 (host_frame,target_frame) 组合
		for (int k = min; k < max; k++) // 帧的范围 最大nframes[0]*nframes[0]
		{
            //; 一个取整，一个取余，得到host_id和target_id
			int h = k % nframes[0]; // 和两个循环一样
			int t = k / nframes[0];

            //; 在整个大的H中的索引，整个大的H中左上角存储相机内参部分
			int hIdx = CPARS + h * 8; // 起始元素id
			int tIdx = CPARS + t * 8;
            // 总id，靠，这不就是k吗？
			int aidx = h + nframes[0] * t; // 总的id

			assert(aidx == k);   // 确实是k, 无语，再算一下干啥？

			MatPCPC accH = MatPCPC::Zero(); // (8+4+1)*(8+4+1)矩阵，也就是13*13的hessian矩阵

            // Step 1 内层循环累积计算accH，这个循环是用于累加多个线程的结果，accH就是acc[h+nframes*t]
			for (int tid2 = 0; tid2 < toAggregate; tid2++)
			{
				acc[tid2][aidx].finish();  //; 注意这里就是从acc中拿出来对应的8*8位置的那个13x13的hessian矩阵
				if (acc[tid2][aidx].num == 0)
					continue;
				accH += acc[tid2][aidx].H.cast<double>(); // 不同线程之间的加起来
			}

			// Step 相对的量通过adj变成绝对的量, 并累加到 H, b 中
            //?   关于相对相对量和绝对量转换这部分，在涂金戈博客最后有讲解：https://www.cnblogs.com/JingeTU/p/8586163.html
            //; 注意下面为什么从(4,4)开始取？因为每个小的hessian都会计算关于内参部分的hessian，并且这部分位于13x13的
            //;  左上角4x4位置。而在整个68x68的大hessian中，只有一个4x4的内参，所这里先累加帧帧之间的位姿+光度(8x8)部分，
            //;  最后再累加相机内参部分。
            // Step 2 H部分累加
            //! 注意：好像仍然只是算了对角线的一边，另一边没有算？
            //   Step 2.1 单纯的相机位姿+光度部分
            // 1.host-host部分，比如(4+8*2, 4+8*2)位置
			H[tid].block<8, 8>(hIdx, hIdx).noalias() += 
                EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adHost[aidx].transpose();
            // 2.target-target部分，比如(4+8*5, 4+8*5)位置
			H[tid].block<8, 8>(tIdx, tIdx).noalias() += 
                EF->adTarget[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();
            // 3.host-target部分，比如(4+8*2, 4+8*5)位置
			H[tid].block<8, 8>(hIdx, tIdx).noalias() += 
                EF->adHost[aidx] * accH.block<8, 8>(CPARS, CPARS) * EF->adTarget[aidx].transpose();
            
            //   Step 2.2 相机位姿+光度 和 相机内参交叉的部分
            // 1.host-target部分，比如(4+8*2, 0)位置
			H[tid].block<8, CPARS>(hIdx, 0).noalias() += 
                EF->adHost[aidx] * accH.block<8, CPARS>(CPARS, 0);
            // 2.target-cam部分，比如(4+8*5, 0)位置
			H[tid].block<8, CPARS>(tIdx, 0).noalias() += 
                EF->adTarget[aidx] * accH.block<8, CPARS>(CPARS, 0);
            
            //   Step 2.3 相机内参单独部分
			H[tid].topLeftCorner<CPARS, CPARS>().noalias() += 
                accH.block<CPARS, CPARS>(0, 0);


            // Step 3 b部分累加
            // 1.host部分，比如(4+8*2, 0)位置
			b[tid].segment<8>(hIdx).noalias() += 
                EF->adHost[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);
            // 2.target部分，比如(4+8*5, 0)位置
			b[tid].segment<8>(tIdx).noalias() += 
                EF->adTarget[aidx] * accH.block<8, 1>(CPARS, CPARS + 8);
            // 3.内参部分，比如(0, 0)位置
			b[tid].head<CPARS>().noalias() += 
                accH.block<CPARS, 1>(0, CPARS + 8); // 残差 * 内参
		}

		// only do this on one thread.
        //; usePrior调用的时候传入的是true，表示有大的先验信息
		if (min == 0 && usePrior)
		{
            //; 以下内容参见深蓝PPT P40, 可以对应的很好
            // 1.相机内参的先验信息
			H[tid].diagonal().head<CPARS>() += EF->cPrior;		// hessian先验
            //; b就是先验 H * 当前值和先验的残差e
			b[tid].head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>()); // H*delta 更新残差
			
            // 2.每个帧的位姿和光度先验信息
            for (int h = 0; h < nframes[tid]; h++)
			{
				H[tid].diagonal().segment<8>(CPARS + h * 8) += EF->frames[h]->prior; // hessian先验
				b[tid].segment<8>(CPARS + h * 8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
			}
		}
	}

}
