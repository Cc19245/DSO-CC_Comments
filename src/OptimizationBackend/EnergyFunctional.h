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
#include "vector"
#include <math.h>
#include "map"

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
	class AccumulatedTopHessian;
	class AccumulatedTopHessianSSE;
	class AccumulatedSCHessian;
	class AccumulatedSCHessianSSE;

	extern bool EFAdjointsValid;
	extern bool EFIndicesValid;
	extern bool EFDeltaValid;

	class EnergyFunctional
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
		friend class EFFrame;
		friend class EFPoint;
		friend class EFResidual;
		friend class AccumulatedTopHessian;
		friend class AccumulatedTopHessianSSE;
		friend class AccumulatedSCHessian;
		friend class AccumulatedSCHessianSSE;

		EnergyFunctional();
		~EnergyFunctional();

		EFResidual *insertResidual(PointFrameResidual *r);
		EFFrame *insertFrame(FrameHessian *fh, CalibHessian *Hcalib);
		EFPoint *insertPoint(PointHessian *ph);

		void dropResidual(EFResidual *r);
		void marginalizeFrame(EFFrame *fh);
		void removePoint(EFPoint *ph);

		void marginalizePointsF();
		void dropPointsF();
		void solveSystemF(int iteration, double lambda, CalibHessian *HCalib);
		double calcMEnergyF();
		double calcLEnergyF_MT();

		void makeIDX();

		void setDeltaF(CalibHessian *HCalib);

		void setAdjointsF(CalibHessian *Hcalib);

        //; 滑窗中的所有能量帧
		std::vector<EFFrame *> frames;	  //< 能量函数中的帧
		int nPoints, nFrames, nResiduals; //< EFPoint的数目, EFframe关键帧数, 残差数

		MatXX HM; //< 优化的Hessian矩阵, 边缘化掉逆深度
		VecX bM;  //< 优化的Jr项, 边缘化掉逆深度

		int resInA, resInL, resInM; //< 分别是在计算A, L, 边缘化H和b中残差的数量
		MatXX lastHS;
		VecX lastbS;
		VecX lastX;

		std::vector<VecX> lastNullspaces_forLogging;
		std::vector<VecX> lastNullspaces_pose;
		std::vector<VecX> lastNullspaces_scale;
		std::vector<VecX> lastNullspaces_affA;
		std::vector<VecX> lastNullspaces_affB;

		IndexThreadReduce<Vec10> *red;

        //< 关键帧之间的连接关系。first: 前32表示host ID, 后32位表示target ID; 
        // second:数目 [0] 两帧之间普通的残差, [1] 两帧之间边缘化的残差
		std::map<uint64_t, Eigen::Vector2i, 
                 //; 后面这俩属于内存操作符，不用管
                 std::less<uint64_t>,
				 Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>> // 64位对齐
				>
                connectivityMap; 

	private:
		VecX getStitchedDeltaF() const;

		void resubstituteF_MT(VecX x, CalibHessian *HCalib, bool MT);
		void resubstituteFPt(const VecCf &xc, Mat18f *xAd, int min, int max, Vec10 *stats, int tid);

		void accumulateAF_MT(MatXX &H, VecX &b, bool MT);
		void accumulateLF_MT(MatXX &H, VecX &b, bool MT);
		void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

		void calcLEnergyPt(int min, int max, Vec10 *stats, int tid);

		void orthogonalize(VecX *b, MatXX *H);

        //; 当前状态两帧之间的相对位姿 - 线性化点两帧之间的相对位姿 
		Mat18f *adHTdeltaF; //< host和target之间位姿的增量, 一共帧数×帧数个

		Mat88 *adHost; //< 伴随矩阵, double
		Mat88 *adTarget;

		//; 伴随矩阵部分，就是求 相对状态 对 绝对状态的雅克比
		Mat88f *adHostF; //< 伴随矩阵, float
		Mat88f *adTargetF;

        //; 相机内参先验对应的Hessian部分，也就是先验对于对应的权重
        //;   草，用4x4矩阵能死是吧？增量也用向量、H也用向量，真实唯恐弄不乱
		VecC cPrior;   //< setting_initialCalibHessian 信息矩阵

        //; 相机内参增量 = 相机当前内参 - 先验内参(或线性化点的内参)
		VecCf cDeltaF; //< 相机内参增量
		VecCf cPriorF; // float型

		AccumulatedTopHessianSSE *accSSE_top_L; //<
		AccumulatedTopHessianSSE *accSSE_top_A; //<

		AccumulatedSCHessianSSE *accSSE_bot; 

        //; 滑窗中的所有能量点，不管这个点的host帧是哪个
		std::vector<EFPoint *> allPoints;
		std::vector<EFPoint *> allPointsToMarg;

		float currentLambda;
	};
}
