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

#include "util/globalCalib.h"
#include "stdio.h"
#include <iostream>

//! 后面带G的是global变量
namespace dso
{
	int wG[PYR_LEVELS], hG[PYR_LEVELS];
	float fxG[PYR_LEVELS], fyG[PYR_LEVELS],
		cxG[PYR_LEVELS], cyG[PYR_LEVELS];

	float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS],
		cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

	Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];

	float wM3G; // w-3 global
	float hM3G;

	/**
	 * @brief 设置一些全局参数，注意这里主要是设置各个层的图像金字塔
	 * 
	 * @param[in] w  输出图像宽
	 * @param[in] h  输出图像高
	 * @param[in] K  输出图像 和 有效归一化平面范围之间的投影矩阵
	 */
	void setGlobalCalib(int w, int h, const Eigen::Matrix3f &K)
	{
		int wlvl = w;
		int hlvl = h;
		pyrLevelsUsed = 1;  //; 最终的图像金字塔层数
		// Step 1 : 计算输出图像能够构造的金字塔层数（注意和输出图像的分辨率有关）
		//; 比如输出图像640*480，图像金字塔分别是640*480, 320*240, 160*120, 
		//; 80*60(执行完这个之后就不满足w*h>5000了)， 所以最后一共就是4层金字塔
		while (wlvl % 2 == 0 && hlvl % 2 == 0 && wlvl * hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS)  // PYR_LEVELS = 6
		{
			wlvl /= 2;
			hlvl /= 2;
			pyrLevelsUsed++;
		}
		printf("using pyramid levels 0 to %d. coarsest resolution: %d x %d!\n",
			   pyrLevelsUsed - 1, wlvl, hlvl);
		
		// Step 2 : 判断输出图像分辨率是否太高，如果最高层的宽和高都>100，
		//       只有可能金字塔达到了6层而退出上面的while循环，说明输出图像分辨率太大了
		if (wlvl > 100 && hlvl > 100)
		{
			printf("\n\n===============WARNING!===================\n "
				   "using not enough pyramid levels.\n"
				   "Consider scaling to a resolution that is a multiple of a power of 2.\n");
		}
		
		// Step 3 : 判断输出图像分辨率是否太低
		if (pyrLevelsUsed < 3)
		{
			printf("\n\n===============WARNING!===================\n "
				   "I need higher resolution.\n"
				   "I will probably segfault.\n");
		}

		wM3G = w - 3;
		hM3G = h - 3;

		// Step 4 : 计算金字塔各个层的参数
		//; 1.第0层（原始输出图像）的宽、高、内参矩阵
		wG[0] = w;
		hG[0] = h;
		KG[0] = K;
		fxG[0] = K(0, 0);
		fyG[0] = K(1, 1);
		cxG[0] = K(0, 2);
		cyG[0] = K(1, 2);
		KiG[0] = KG[0].inverse();  //; 投影矩阵的逆
		fxiG[0] = KiG[0](0, 0);
		fyiG[0] = KiG[0](1, 1);
		cxiG[0] = KiG[0](0, 2);
		cyiG[0] = KiG[0](1, 2);
		
		//; 2.计算金字塔其他所有层的宽、高、内参矩阵
		for (int level = 1; level < pyrLevelsUsed; ++level)
		{
			// 宽高移位就是/2
			wG[level] = w >> level;
			hG[level] = h >> level;

			// 内参矩阵也都满足/2的操作
			fxG[level] = fxG[level - 1] * 0.5;
			fyG[level] = fyG[level - 1] * 0.5;
			cxG[level] = (cxG[0] + 0.5) / ((int)1 << level) - 0.5;
			cyG[level] = (cyG[0] + 0.5) / ((int)1 << level) - 0.5;

			KG[level] << fxG[level], 0.0, cxG[level], 0.0, fyG[level], cyG[level], 0.0, 0.0, 1.0; // synthetic
			KiG[level] = KG[level].inverse();

			fxiG[level] = KiG[level](0, 0);
			fyiG[level] = KiG[level](1, 1);
			cxiG[level] = KiG[level](0, 2);
			cyiG[level] = KiG[level](1, 2);
		}
	}

}
