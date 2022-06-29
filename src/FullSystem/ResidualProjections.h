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
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "util/settings.h"

namespace dso
{

	//@ 返回逆深度的导数值
	EIGEN_STRONG_INLINE float derive_idepth(
		const Vec3f &t, const float &u, const float &v,
		const int &dx, const int &dy, const float &dxInterp,
		const float &dyInterp, const float &drescale)
	{
		return (dxInterp * drescale * (t[0] - t[2] * u) + dyInterp * drescale * (t[1] - t[2] * v)) * SCALE_IDEPTH;
	}

	//@ 把host上的点变换到target上
	EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt, const float &v_pt,
		const float &idepth,
		const Mat33f &KRKi, const Vec3f &Kt,
		float &Ku, float &Kv)
	{
		Vec3f ptp = KRKi * Vec3f(u_pt, v_pt, 1) + Kt * idepth; // host上点除深度
		Ku = ptp[0] / ptp[2];
		Kv = ptp[1] / ptp[2];
		return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G; // 不在边缘
	}

	//@ 将host帧投影到新的帧, 且可以设置像素偏移dxdy, 得到:
	//@ 参数: [drescale 新比旧逆深度] [uv 新的归一化平面]
	//@		[kukv 新的像素平面] [KliP 旧归一化平面] [new_idepth 新的逆深度]
	/**
	 * @brief  参考博客：epsilonjohn.club/2020/03/16/SLAM代码课程/DSO/DSO-1-系统框架与初始化/
	 * 
	 * @param[in] u_pt 
	 * @param[in] v_pt 
	 * @param[in] idepth   i帧下的逆深度
	 * @param[in] dx 
	 * @param[in] dy 
	 * @param[in] HCalib 
	 * @param[in] R 
	 * @param[in] t 
	 * @param[in] drescale 
	 * @param[in] u 
	 * @param[in] v 
	 * @param[in] Ku 
	 * @param[in] Kv 
	 * @param[in] KliP 
	 * @param[in] new_idepth 
	 * @return EIGEN_STRONG_INLINE 
	 */
	EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt, const float &v_pt,
		const float &idepth,
		const int &dx, const int &dy,
		CalibHessian *const &HCalib,
		const Mat33f &R, const Vec3f &t,
		float &drescale, float &u, float &v,
		float &Ku, float &Kv, Vec3f &KliP, float &new_idepth)
	{
		// host上归一化平面点
		KliP = Vec3f(
			(u_pt + dx - HCalib->cxl()) * HCalib->fxli(),
			(v_pt + dy - HCalib->cyl()) * HCalib->fyli(),
			1);

		//; 这里其实没有什么复杂的，就是正常来说RP+t算坐标变换时，P点应该用i帧相机坐标系下的正常表示的点，而不是i帧相机归一化
		//; 平面上的点，即应该是R*(KliP/idepth)+t, 其中idepth是i帧下的逆深度，这样得到的就是j帧相机坐标系下的正常点。但是
		//; 这里又处理了一步，把上面的结果再乘以idepth，就相当于把j帧坐标系下的正常点坐标又做了一个缩放，即又除以了i帧坐标下
		//; 的深度（注意不是j帧坐标下的深度，因此此时得到的ptp并不是j帧归一化平面上的点）。其实此时要把ptp点投影到j帧像素坐标
		//; 很简单，直接除以它的第3维度的坐标把他转到j帧相机归一化平面上即可（也就是求的drescale，注意这并不是j帧下的逆深度），
		//; 这个和j帧相机坐标系下的正常点的处理没有什么区别
		Vec3f ptp = R * KliP + t * idepth;   // 这就是虚拟点P_j^{virtual}，也就是j帧相机坐标系下的坐标在乘以i帧下的逆深度
		drescale = 1.0f / ptp[2];		// target帧逆深度 比 host帧逆深度
		new_idepth = idepth * drescale; // 新的帧上逆深度

		if (!(drescale > 0))
			return false;

		// 归一化平面，将虚拟点P_j^{virtual}转到归一化平面上
		u = ptp[0] * drescale;
		v = ptp[1] * drescale;
		// 像素平面
		Ku = u * HCalib->fxl() + HCalib->cxl();
		Kv = v * HCalib->fyl() + HCalib->cyl();

		return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G;
	}

}
