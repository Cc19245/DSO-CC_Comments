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
#include "vector"
#include <math.h>
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso
{

	class PointFrameResidual;
	class CalibHessian;
	class FrameHessian;
	class PointHessian;

	class EFResidual;
	class EFPoint;
	class EFFrame;
	class EnergyFunctional;

	class EFResidual
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

		inline EFResidual(PointFrameResidual *org, EFPoint *point_, EFFrame *host_, EFFrame *target_) : 
            data(org), point(point_), host(host_), target(target_)
		{
			isLinearized = false;  //; 刚构造，不是上次FEJ边缘化的残差
			isActiveAndIsGoodNEW = false;   //; 刚构造，不是激活残差
			J = new RawResidualJacobian();
			assert(((long)this) % 16 == 0);
			assert(((long)J) % 16 == 0);
		}
		inline ~EFResidual()
		{
			delete J;
		}

		void takeDataF();

		void fixLinearizationF(EnergyFunctional *ef);

		// structural pointers
        //; 这个点残差是由哪个能量点构成的
		PointFrameResidual *data;

        //; 这个残差关联的host/target能量帧在后端能量帧数组中的索引
		int hostIDX, targetIDX; //< 残差对应的 host 和 Target ID号
		EFPoint *point;			//< 这个残差是由哪个点构成的
		EFFrame *host;			//< 这个点的host帧
		EFFrame *target;		//< 这个点的target帧
		int idxInAll;			//< 这个残差在由这个点构成的所有残差中的id

		RawResidualJacobian *J; //< 用来计算jacob, res值

		VecNRf res_toZeroF; //< 更新delta后的线性残差
		Vec8f JpJdF;		//< 逆深度Jaco和位姿+光度Jaco的Hessian

		// status.
        //; 这个点能量残差是否被线性化了，也就是是否是上次FEJ保留下来的
		bool isLinearized; //< 计算完成res_toZeroF

        //; 从下面看isActive，反应这个变量就是表示这个点能量残差是否是active的
		// if residual is not OOB & not OUTLIER & should be used during accumulations
		bool isActiveAndIsGoodNEW;	//< 激活的还可以参与优化
		inline const bool &isActive() const { return isActiveAndIsGoodNEW; } //< 是不是激活的取决于残差状态
	};

	enum EFPointStatus
	{
		PS_GOOD = 0,
		PS_MARGINALIZE,
		PS_DROP
	};

	class EFPoint
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		EFPoint(PointHessian *d, EFFrame *host_) : data(d), host(host_)
		{
			takeData();
			stateFlag = EFPointStatus::PS_GOOD;
		}
		void takeData();

		PointHessian *data; //< PointHessian数据

		float priorF; //< 逆深度先验信息矩阵, 初始化之后的有
		float deltaF; //< 当前逆深度和线性化处的差, 没有使用FEJ, 就是0

		// constant info (never changes in-between).
		int idxInPoints; //< 当前点在EFFrame中id
		EFFrame *host;

		// contains all residuals.
		std::vector<EFResidual *> residualsAll; //< 该点的所有残差

		float bdSumF;	 //< 当前残差 + 边缘化先验残差
		float HdiF;		 //< 逆深度hessian的逆, 协方差
		float Hdd_accLF; //< 边缘化, 逆深度的hessian
		VecCf Hcd_accLF; //< 边缘化, 逆深度和内参的hessian
		float bd_accLF;	 //< 边缘化, J逆深度*残差
		float Hdd_accAF; //< 正常逆深度的hessian
		VecCf Hcd_accAF; //< 正常逆深度和内参的hessian
		float bd_accAF;	 //< 正常 J逆深度*残差

		EFPointStatus stateFlag; //< 点的状态
	};

	class EFFrame
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		EFFrame(FrameHessian *d) : data(d)
		{
			takeData();  
		}
		void takeData();


		//! 注意以下的三个Vec8都对应位姿 0-5, 光度ab 6-7，但是有的是hessian，有的是状态增量
        //; 位姿和光度的先验hessian，注意是hessian不是状态!
        //TODO 服了，用8x8矩阵写清楚点不好吗？
		Vec8 prior;		  //< 位姿只有第一帧有先验 prior hessian (diagonal)
        
        //; 当前帧状态 - 先验相机状态(实际上都给了0)，其实结果就是当前帧状态
		Vec8 delta_prior; //< 相对于先验的增量   // = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
        //; 当前帧的状态 - 上一次线性化点的状态的增量
        Vec8 delta;		  //< 相对于线性化点位姿, 光度的增量  // state - state_zero.

		std::vector<EFPoint *> points; //< 帧上所有点
		FrameHessian *data;			   //< 对应FrameHessian数据

		//? 和FrameHessian中的idx有啥不同
        //! CC: 确实，我感觉应该是一样的吧？但是赋值并没有使用前端FrameHessian中的帧索引
		int idx; //< 在能量函数中帧id // idx in frames.

		int frameID; //< 所有历史帧ID
	};

}
