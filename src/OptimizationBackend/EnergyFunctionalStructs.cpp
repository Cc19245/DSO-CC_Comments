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

#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

    /**
     * @brief  计算优化过程中后面构造H矩阵的时候，需要的中间变量
     *  参考博客1：https://blog.csdn.net/xxxlinttp/article/details/90640350?spm=1001.2014.3001.5502
     *    博客中4.7.6节对下面的变量代表的物理含义进行了介绍
     *  参考博客2：https://www.cnblogs.com/JingeTU/p/8395046.html
     *    涂金戈的博客，对应最后的10这个小节
     */
	void EFResidual::takeDataF()
	{
        // Step 1 获取前端计算的雅克比
        //; 这里就是把持有的PointFrameResidual *data内部的雅克比RawResidualJacobian，
        //;  交换给的当前类的成员变量RawResidualJacobian* J，从而获得从前端计算的雅克比
		std::swap<RawResidualJacobian *>(J, data->J);

        // Step 2 利用前端计算的雅克比，再算一些计算hessian的时候需要的中间量，即JpJdF
    //! 7.26增：JpJdF[1:6]代表hessian中的Hfd, JpJdF[7:8]代表b中的bf，这个是给后端逆深度舒尔消元使用的
        //; 1.注意下面的计算和两个博客中的是吻合的，涂金戈博客中写的是  图像导数 * (图像导数 * 逆深度导数) = 2x8 * 8x1 = 2x1
        //;   这不过这里代码中实际用的是 (图像导数 * 图像导数) * 逆深度导数 = 2x2 * 2x1 = 2x1
		// 图像导数 * 图像导数 * 逆深度导数
		Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;
		
        //; 2.Vec8f JpJdF
        //;   同理这里和涂金戈中博客稍有不同，涂金戈博客中化简成了 (位姿导数 * 图像导数 * 图像导数) * 逆深度导数
        //;   而这里复用了上面刚计算的结果，是计算了 位姿导数 * (图像导数 * 图像导数 * 逆深度导数)
        // 位姿导数 * 图像导数 * 图像导数 * 逆深度导数
		for (int i = 0; i < 6; i++)
        {
            JpJdF[i] = J->Jpdxi[0][i] * JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];
        }   

        //; 3.同理这里仍然是和涂金戈博客中稍有不同，涂金戈博客简化成了  图像导数 * (逆深度导数 * 光度导数)
        //;   但是这里用的是(图像导数 * 逆深度导数) * 光度导数
		// 图像导数 * 逆深度导数 * 光度导数
		JpJdF.segment<2>(6) = J->JabJIdx * J->Jpdd;
	}


	//@ 从 FrameHessian 中提取数据
    /**
     * @brief 从前端帧中提取数据，存储到类成员变量中。
     *   注意：此函数是在构造函数中被调用
     * 
     */
	void EFFrame::takeData()
	{
        // 1.得到这一帧的先验hessian, 主要是光度仿射变换的hessian
		prior = data->getPrior().head<8>();	  

        // 2.状态 与 FEJ线性化点 之差, state - state_zero
		delta = data->get_state_minus_stateZero().head<8>();	

        // 3.状态 与 先验在线性化点 之差, state - 0
        //! 疑问：这里不太明白
		delta_prior = (data->get_state() - data->getPriorZero()).head<8>(); 

		assert(data->frameID != -1);

        //; 拷贝这个能量帧在历史所有关键帧中的序号
		frameID = data->frameID; // 所有帧的ID序号
	}


	//@ 从PointHessian读取先验和当前状态信息
	void EFPoint::takeData()
	{
		priorF = data->hasDepthPrior ? setting_idepthFixPrior * SCALE_IDEPTH * SCALE_IDEPTH : 0;
		if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
			priorF = 0;

		//TODO 每次都更新线性化点，这不一直是零？？
		deltaF = data->idepth - data->idepth_zero; // 当前状态逆深度减去线性化处
	}

	//@ 计算线性化更新后的残差,
	/**
	 * @brief 计算某个点构成的能量残差在线性化点的处的残差，因为优化之后重新把他线性化了
	 * 
	 * @param[in] ef 
	 */
	void EFResidual::fixLinearizationF(EnergyFunctional *ef)
	{
		Vec8f dp = ef->adHTdeltaF[hostIDX + ef->nFrames * targetIDX]; // 得到hostIDX --> targetIDX的状态增量

		// compute Jp*delta
        //; Jpdxi: dx/dξ, dp:前六维就是dξ； Jpdc: dx/dc, cDeltaF: 相机内参相对先验的增量； 
        //; Jpdd:  dx/d_rou, deltaF: 逆深度增量
		__m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>()) + J->Jpdc[0].dot(ef->cDeltaF) + J->Jpdd[0] * point->deltaF);
		__m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>()) + J->Jpdc[1].dot(ef->cDeltaF) + J->Jpdd[1] * point->deltaF);
		__m128 delta_a = _mm_set1_ps((float)(dp[6]));  //; 这两个是光度的增量
		__m128 delta_b = _mm_set1_ps((float)(dp[7]));

        //; 算这个点周围一个pattern一共8个点构成的值
		for (int i = 0; i < patternNum; i += 4)
		{
			// PATTERN: rtz = resF - [JI*Jp Ja]*delta.
            //; resF就是光度残差，是一个8x1向量，但是这里用指针把其中每个元素都取出来了
			__m128 rtz = _mm_load_ps(((float *)&J->resF) + i); // 光度残差
			
            //? res - J * delta_x
            //; 下面的公式确实就是resF - J*dx。 JIdx: dr/dx;  JabF: dr/dl
			rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JIdx)) + i), Jp_delta_x));
			rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JIdx + 1)) + i), Jp_delta_y));
			rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JabF)) + i), delta_a));
			rtz = _mm_sub_ps(rtz, _mm_mul_ps(_mm_load_ps(((float *)(J->JabF + 1)) + i), delta_b));
			
            //; 把上面计算的残差结果存到res_toZeroF中
            _mm_store_ps(((float *)&res_toZeroF) + i, rtz); // 存储在res_toZeroF
  
		}
		isLinearized = true;
	}

}
