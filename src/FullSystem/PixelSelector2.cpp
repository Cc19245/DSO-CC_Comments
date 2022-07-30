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

#include "FullSystem/PixelSelector2.h"

//

#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace dso
{

	PixelSelector::PixelSelector(int w, int h)
	{
		randomPattern = new unsigned char[w * h];
		std::srand(3141592); // want to be deterministic.
		for (int i = 0; i < w * h; i++)
			randomPattern[i] = rand() & 0xFF; // 随机数, 取低8位

		currentPotential = 3;

		// 划分的所有32*32大小的block进行计算阈值
		gradHist = new int[100 * (1 + w / 32) * (1 + h / 32)];
		ths = new float[(w / 32) * (h / 32) + 100];  //; 每个block的阈值（为啥要+100？）
		thsSmoothed = new float[(w / 32) * (h / 32) + 100];   //; 均值滤波之后的block阈值

		allowFast = false;
		gradHistFrame = 0;
	}

	PixelSelector::~PixelSelector()
	{
		delete[] randomPattern;
		delete[] gradHist;
		delete[] ths;
		delete[] thsSmoothed;
	}

	/**
	 * @brief 求32*32block中的梯度直方图的阈值
	 * 
	 * @param[in] hist   传入的梯度1-49个格的直方图
	 * @param[in] below  计算阈值占比多少，传入的是0.5，也就是50%
	 * @return int  计算得到的阈值是多少
	 */
	int computeHistQuantil(int *hist, float below)
	{
		// hist[0]是block中所有像素的个数，*below就得到了梯度的阈值
		int th = hist[0] * below + 0.5f; // 最低的像素个数
		for (int i = 0; i < 90; i++)	 // 90? 这么随便....（其实已经确定了hist最大索引49，这里用90不会出问题，不过确实随意了点）
		{
			th -= hist[i + 1]; // 梯度值为0-i的所有像素个数占 below %
			if (th < 0)
				return i;
		}
		return 90;
	}

	//* 生成梯度直方图, 为每个block计算阈值
	void PixelSelector::makeHists(const FrameHessian *const fh)
	{
		gradHistFrame = fh;  //; 赋值给类成员变量，防止下一次再次计算梯度直方图
		float *mapmax0 = fh->absSquaredGrad[0]; //第0层梯度平方和

		// weight and height
		int w = wG[0];
		int h = hG[0];

		//!还是每个blocks大小为32*32, 不是论文里的32*32个网格
		int w32 = w / 32;  //; 一共可以划分成多少个32*32的网格
		int h32 = h / 32;
		thsStep = w32;

		// Step 1 遍历所有的32*32网格, 求每个网格内的梯度的阈值
		for (int y = 0; y < h32; y++)
		{
			for (int x = 0; x < w32; x++)
			{
				float *map0 = mapmax0 + 32 * x + 32 * y * w; // y行x列的格
				// 利用数组hist0[]存储梯度平方和（从1-49）相同的像素个数，hist0[0]存储整个块中的像素数量
				int *hist0 = gradHist;						 // + 50*(x+y*w32);
				memset(hist0, 0, sizeof(int) * 50);			 // 分成50格

				for (int j = 0; j < 32; j++)
				{
					for (int i = 0; i < 32; i++)
					{
						int it = i + 32 * x; // 该格里第(j,i)像素的整个图像坐标
						int jt = j + 32 * y;
						if (it > w - 2 || jt > h - 2 || it < 1 || jt < 1)
							continue;					//内
						int g = sqrtf(map0[i + j * w]); // 梯度平方和开根号
						if (g > 48)
							g = 48;		//? 为啥是48这个数，因为一共分为了50格
						hist0[g + 1]++; // 1-49 存相应梯度个数
						hist0[0]++;		// 所有的像素个数
					}
				}
				// 得到每一block的梯度阈值， setting_minGradHistCut = 0.5，setting_minGradHistAdd = 7
				//; 但是这里注意，从block中算出来的梯度中位数并不是最终结果，还人工添加了一个梯度偏置7
				ths[x + y * w32] = computeHistQuantil(hist0, setting_minGradHistCut) + setting_minGradHistAdd;
			}
		}

		// Step 2 遍历所有的block, 使用3*3的窗口求平均值来平滑求得的梯度阈值
		for (int y = 0; y < h32; y++)
		{
			for (int x = 0; x < w32; x++)
			{
				float sum = 0, num = 0;
				if (x > 0)
				{
					if (y > 0)
					{
						num++;
						sum += ths[x - 1 + (y - 1) * w32];
					}
					if (y < h32 - 1)
					{
						num++;
						sum += ths[x - 1 + (y + 1) * w32];
					}
					num++;
					sum += ths[x - 1 + (y)*w32];
				}

				if (x < w32 - 1)
				{
					if (y > 0)
					{
						num++;
						sum += ths[x + 1 + (y - 1) * w32];
					}
					if (y < h32 - 1)
					{
						num++;
						sum += ths[x + 1 + (y + 1) * w32];
					}
					num++;
					sum += ths[x + 1 + (y)*w32];
				}

				if (y > 0)
				{
					num++;
					sum += ths[x + (y - 1) * w32];
				}
				if (y < h32 - 1)
				{
					num++;
					sum += ths[x + (y + 1) * w32];
				}
				num++;
				sum += ths[x + y * w32];

				thsSmoothed[x + y * w32] = (sum / num) * (sum / num);
			}
		}
	}

	/********************************
	 * @ function:
	 * 
	 * @ param: 	fh				帧Hessian数据结构
	 * @			map_out			选出的地图点
	 * @			density		 	每一金字塔层要的点数(密度)，其实这个函数只有在第0层调用，传参是0.03*w*h
	 * @			recursionsLeft	最大递归次数, 调用是1
	 * @			plot			是否画图（也就是画出来提取的像素点）,调用是false
	 * @			thFactor		阈值因子，调用是2
	 * @
	 * @ note:		使用递归
	 *******************************/
	int PixelSelector::makeMaps(
		const FrameHessian *const fh,
		float *map_out, float density, int recursionsLeft, bool plot, float thFactor)
	{
		float numHave = 0;
		float numWant = density;  //; 想提取的点个数（0.03*w*h）
		float quotia;
		int idealPotential = currentPotential;

		{
			// the number of selected pixels behaves approximately as
			// K / (pot+1)^2, where K is a scene-dependent constant.
			// we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.
			// Step 1 没有计算直方图, 以及选点的阈值, 则调用函数生成block阈值
			if (fh != gradHistFrame)
			{
				//; 生成梯度直方图, 为每个block计算阈值。简单理解就是先给每一个小块计算阈值，然后用3x3的直方图对阈值进行滤波
				makeHists(fh); // 第一次进来，求梯度直方图的frame不是fh，则生成直方图
			}

			// select!
			// Step 2 在当前帧上选择符合条件的像素（利用上一步求的梯度直方图作为阈值）
			Eigen::Vector3i n = this->select(fh, map_out, currentPotential, thFactor);

			// sub-select!
			numHave = n[0] + n[1] + n[2]; // 选择得到的点
			quotia = numWant / numHave;	  // 想要的 与 得到的 比例

			// Step 3 计算新的采像素点的, 范围大小, 相当于动态网格了, pot越小取得点越多
            //   Step （因为在输出图像上一个pot里面只取阈值最大的点）
			// by default we want to over-sample by 40% just to be sure.
			//! 问题：这里为啥要+1？
			float K = numHave * (currentPotential + 1) * (currentPotential + 1); // 相当于覆盖的面积, 每一个像素对应一个pot*pot
			idealPotential = sqrtf(K / numWant) - 1;							 // round down.
			if (idealPotential < 1)
				idealPotential = 1;

			// Step 4 想要的数目和已经得到的数目, 大于或小于0.25都会重新采样一次
			// recursionsLeft递归剩余次数，传入是1
			if (recursionsLeft > 0 && quotia > 1.25 && currentPotential > 1)
			{
				//re-sample to get more points!
				// potential needs to be smaller
				if (idealPotential >= currentPotential)	   // idealPotential应该小
					idealPotential = currentPotential - 1; // 减小,多采点

				//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
				//				100*numHave/(float)(wG[0]*hG[0]),
				//				100*numWant/(float)(wG[0]*hG[0]),
				//				currentPotential,
				//				idealPotential);
				currentPotential = idealPotential;
				
				//; 这里可以发现，此时传入的递归次数是0，所以进入之后只再重新采样一次
				return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor); //递归
			}
			else if (recursionsLeft > 0 && quotia < 0.25)
			{
				// re-sample to get less points!

				if (idealPotential <= currentPotential)	   // idealPotential应该大
					idealPotential = currentPotential + 1; // 增大, 少采点

				//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
				//				100*numHave/(float)(wG[0]*hG[0]),
				//				100*numWant/(float)(wG[0]*hG[0]),
				//				currentPotential,
				//				idealPotential);
				currentPotential = idealPotential;
				return makeMaps(fh, map_out, density, recursionsLeft - 1, plot, thFactor);
			}
		}

		// Step 5 现在提取的还是过多（容忍范围5%）, 随机删除一些点
		int numHaveSub = numHave;
		if (quotia < 0.95)
		{
			int wh = wG[0] * hG[0];
			int rn = 0;
			unsigned char charTH = 255 * quotia;
			for (int i = 0; i < wh; i++)
			{
				if (map_out[i] != 0)
				{
					//; randomPattern是对应图像宽、高大小的随机数数组
					if (randomPattern[rn] > charTH)
					{
						map_out[i] = 0;
						numHaveSub--;
					}
					rn++;
				}
			}
		}

		//	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
		//			100*numHave/(float)(wG[0]*hG[0]),
		//			100*numWant/(float)(wG[0]*hG[0]),
		//			currentPotential,
		//			idealPotential,
		//			100*numHaveSub/(float)(wG[0]*hG[0]));
		currentPotential = idealPotential; //???

		// 画出选择结果
		if (plot)
		{
			int w = wG[0];
			int h = hG[0];

			MinimalImageB3 img(w, h);

			for (int i = 0; i < w * h; i++)
			{
				float c = fh->dI[i][0] * 0.7; // 像素值
				if (c > 255)
					c = 255;
				img.at(i) = Vec3b(c, c, c);
			}
			IOWrap::displayImage("Selector Image", &img);

			// 安照不同层数的像素, 画上不同颜色
			for (int y = 0; y < h; y++)
				for (int x = 0; x < w; x++)
				{
					int i = x + y * w;
					if (map_out[i] == 1)
						img.setPixelCirc(x, y, Vec3b(0, 255, 0));
					else if (map_out[i] == 2)
						img.setPixelCirc(x, y, Vec3b(255, 0, 0));
					else if (map_out[i] == 4)
						img.setPixelCirc(x, y, Vec3b(0, 0, 255));
				}
			IOWrap::displayImage("Selector Pixels", &img);
		}

		return numHaveSub;
	}

	//? 这个选点到底是不同层上, 还是论文里提到的不同阈值, 不同block???
	//! CC解答：是在同一个最大的12x12的block中，利用同一层上的阈值选择点。比如在12x12的范围内，所有选择的点都
	//!  必须使用的是同一层上的阈值，比如要么全是使用第0层的阈值，要么全是使用第1层的阈值，要么全是使用第2层的阈值。
	//!  但是最后选择的点，都是在输出图像上的坐标，也就是金字塔第0层的
	/********************************
	 * @ function:		根据阈值选择不同层上符合要求的像素
	 * 
	 * @ param: 		fh						传入的帧hessian矩阵
	 * @				map_out					选中的像素点及所在层
	 * @				pot(currentPotential)	选点的范围大小, 一个pot内选一个，调用是3
	 * @				thFactor				阈值因子(乘数)，调用是2，就是梯度直方图的阈值还要x2才作为最后阈值
	 * 
	 * @ note:			返回的是每一层选择的点的个数
	 *******************************/
	Eigen::Vector3i PixelSelector::select(const FrameHessian *const fh,
										  float *map_out, int pot, float thFactor)
	{
		//const 在*左, 指针内容不可改, 在*右指针不可改
		// 等价const Eigen::Vector3f * const
		Eigen::Vector3f const *const map0 = fh->dI;  //; 一个啥都不能改的指针

		// 0, 1, 2层的梯度平方和
		float *mapmax0 = fh->absSquaredGrad[0];
		float *mapmax1 = fh->absSquaredGrad[1];
		float *mapmax2 = fh->absSquaredGrad[2];

		// 0 1 2层的图像大小
		int w = wG[0];
		int w1 = wG[1];
		int w2 = wG[2];
		int h = hG[0];

		//; 随机的方向，通过把梯度往这些方向上投影，作为选择的梯度的值
		//? 这个是为了什么呢,
		//! 随机选这16个对应方向上的梯度和阈值比较
		//! 每个pot里面的方向随机选取的, 防止特征相同, 重复

		// 模都是1，也就是说在单位圆上找随机的方向
		const Vec2f directions[16] = {
			Vec2f(0, 1.0000),
			Vec2f(0.3827, 0.9239),
			Vec2f(0.1951, 0.9808),
			Vec2f(0.9239, 0.3827),
			Vec2f(0.7071, 0.7071),
			Vec2f(0.3827, -0.9239),
			Vec2f(0.8315, 0.5556),
			Vec2f(0.8315, -0.5556),
			Vec2f(0.5556, -0.8315),
			Vec2f(0.9808, 0.1951),
			Vec2f(0.9239, -0.3827),
			Vec2f(0.7071, -0.7071),
			Vec2f(0.5556, 0.8315),
			Vec2f(0.9808, -0.1951),
			Vec2f(1.0000, 0.0000),
			Vec2f(0.1951, -0.9808)};

		//? 在哪改变的状态 PixelSelectorStatus ?
		//; 注意PixelSelectorStatus是一个枚举类型，里面有4个分量，所以这里应该就是4
		memset(map_out, 0, w * h * sizeof(PixelSelectorStatus)); // 不同选择状态的数目不同

		// 金字塔层阈值的减小倍数， setting_gradDownweightPerLevel = 0.75
		float dw1 = setting_gradDownweightPerLevel; // 第二层，0.75
		float dw2 = dw1 * dw1;						// 第三层，0.75*0.75（可以看到这个权重是相对第0层的缩放倍数）

		// 第2层1个pot对应第1层4个pot, 第1层1个pot对应第0层的4个pot,
		// 第0层的4个pot里面只要选出来一个像素, 就不在对应高层的pot里面选了,
		// 但是还会在第0层的每个pot里面选大于阈值的像素
		// 阈值随着层数增加而下降
		// 从顶层向下层遍历, 写的挺有意思!

		int n2 = 0, n3 = 0, n4 = 0;   //; 记录第0/1/2层金字塔选择的像素个数
		
		/* 原文链接：https://blog.csdn.net/tanyong_98/article/details/106177134
		1.像素点选取的标准是：像素点在第0层图像的梯度平方和大于pixelTH0*thFactor，像素点在第1层图像的
		  梯度平方和大于pixelTH1*thFactor，像素点在第2层图像的梯度平方和大于pixelTH2*thFactor。
		  这三个条件是有先后顺序的，先判断前面，如果满足就不判断后面的条件，并且并不是满足此三个条件就会
		  被选择，而是有资格被选择，具体下面分析。
		2.像素点选取方法是：利用for循环实现4倍步长，2倍步长，1倍步长的遍历像素点选择满足上述像素点选取
		  标准的像素点，然后像素点的dirNorm还需大于在当前步长区域中上一个选取的像素点dirNorm。（所以
		  并不是选取最大的，而是要大于在当前步长区域内已选择的像素点）通过不同标准选取的像素点，其对应
		  位置的statusMap值不一样，条件1的值为1,条件2的值为2,条件3的值为4。
		最后，会对选取点的数量进行判断，然后调整步长重新选点。若选点过多，则增大步长，反之则减小步长。
		 （这个函数里好像没有体现这一点）
		*/

		//; 自己的理解解释下面的遍历：
		/* 1.首先明确下面的遍历其实是在输出图像上 逐像素 遍历的，也就是不论最后特征点是在金字塔哪层选择出来的，输出图像的
		     每一个像素都被遍历了一遍（感觉有点耗时？）
		   2.pot范围划分：定义在不同金字塔层上的pot对应输出图像的像素范围，第0层（即输出图像）pot大小是原始像素的3x3范围；
		     第1层图像缩小2倍，那么反过来对应第1层pot大小对应原始像素就是6x6范围；同理第2层金字塔pot对应原始像素12x12大小。
		   3.遍历顺序：既然是在输出图像上逐个像素遍历，所以for循环最外层就是pot范围最大的，即12x12（注意此时和金字塔第2层
		     无关，因为我们是在输出图像上遍历）。然后12x12范围内在划分成4个第1层金字塔对应的pot大小，即4个6x6。然后6x6
			 范围内再划分成4个第0层（即输出图像）金字塔对应的pot大小，即4个3x3（这就是最内层的for循环）。
		   4.对3中划分出来的每一个最小的3x3像素范围，对其中的 每一个像素 判断是否超过阈值。如果超过了，那么这个3x3像素范围
		     内至少选择出来了一个点，那么就不用把这9个像素变换到第1层再去判断了。如果恰巧9个点在第0层都没有被选择出来，那么
			 每一次他们都会变换到第1层，利用第1层的阈值再去筛选，如果筛选出来就不用再变换到第2层继续筛选了；反之同理。
		   5.也就是说，实际上输出图像上的每个像素点都会被遍历一遍，只不过为了筛选出特征点，如果使用底层的阈值没有选择出来，
		     会把这个像素转到高层，利用高层的插值像素的梯度和高层更低的梯度阈值继续进行筛选。如果在高层筛选出来了，注意本质上
			 这个点还是在输出图像上，但是这个点在输出图像上的梯度不满足阈值要求，缩放到高层金字塔上才满足阈值要求。
           6.另外注意：在输出图像上遍历3x3的范围的时候，如果选择出来一个点，那么在这个3x3所在的6x6、12x12范围内的其他3x3
		     范围内的点，都必须在第0层的阈值上进行筛选，而不会再变换到第1、2层进行筛选。这个是有道理的，因为在这个12x12的
			 范围内既然选出来了一个满足第0层阈值的点，那么其他像素就也都应该在第0层的阈值里面选。如果第0层阈值没有选择出来点，
			 而在第1层阈值上选择出来了，那么同理12x12范围内的其他像素的点也都应该在第1层的阈值里面选，而不能用第2层的阈值
			 选择出来的点。
			 也就是说：如果在一个12x12的范围内，有一个点在第0层的阈值上被选了出来，那么这个范围内的其他所有点也都应该使用第0
			 层的阈值来选择，而不能使用第1、2层的阈值来选择。
		*/
		// 在第2层金字塔中，每隔pot选一个点遍历。选择初始帧的时候，pot=3, 4*pot=12
		for (int y4 = 0; y4 < h; y4 += (4 * pot))
		{
			for (int x4 = 0; x4 < w; x4 += (4 * pot))
			{
				// 该点的邻域(向上取4pot或末尾余数)大小 
				//; 这里只要没有到达图像边界，一般都是4*pot=12
				int my3 = std::min((4 * pot), h - y4);  //; 这个应该就是不让出图像边界？
				int mx3 = std::min((4 * pot), w - x4);
				int bestIdx4 = -1;
				float bestVal4 = 0;
				// 随机系数
				Vec2f dir4 = directions[randomPattern[n2] & 0xF]; // 取低4位, 0-15, 和directions对应
				//* 上面的领域范围内, 在第1层进行遍历, 每隔pot一个点
				for (int y3 = 0; y3 < my3; y3 += (2 * pot))
				{
					for (int x3 = 0; x3 < mx3; x3 += (2 * pot))
					{
						//; 注意x4/y4就是第0层真实的像素坐标，而x3/y3是对应第0层的这个pot里面的小坐标，
						//; 所以二者加起来就是最后第1层金字塔对应的第0层的真实坐标
						int x34 = x3 + x4; // 对应第0层坐标
						int y34 = y3 + y4;
						// 继续确定该层上的邻域
						//; 同样这里一般都是2*3=6
						int my2 = std::min((2 * pot), h - y34);
						int mx2 = std::min((2 * pot), w - x34);
						int bestIdx3 = -1;
						float bestVal3 = 0;
						Vec2f dir3 = directions[randomPattern[n2] & 0xF];
						//* 上面的邻域范围内, 变换到第0层, 每隔pot遍历
						//! 每个pot大小格里面一个大于阈值的最大的像素
						for (int y2 = 0; y2 < my2; y2 += pot)
						{
							for (int x2 = 0; x2 < mx2; x2 += pot)
							{
								//; 这里就是在第0层金字塔（pot=3，范围最小）里面选择像素点
								int x234 = x2 + x34; // 坐标
								int y234 = y2 + y34;
								int my1 = std::min(pot, h - y234);
								int mx1 = std::min(pot, w - x234);
								//; 以下两个变量是在第0层的3x3像素范围内，选择的目标点的索引和它的梯度值
								int bestIdx2 = -1;
								float bestVal2 = 0;
								Vec2f dir2 = directions[randomPattern[n2] & 0xF];
								//* 第0层中的,pot大小邻域内遍历
								for (int y1 = 0; y1 < my1; y1 += 1)
								{
									for (int x1 = 0; x1 < mx1; x1 += 1)
									{
										assert(x1 + x234 < w);
										assert(y1 + y234 < h);
										int idx = x1 + x234 + w * (y1 + y234); // 像素id
										int xf = x1 + x234;					   // 像素坐标
										int yf = y1 + y234;
										//; 距离图像边界的4个点不选
										if (xf < 4 || xf >= w - 5 || yf < 4 || yf > h - 4)
											continue;

										// 直方图求得阈值, 除以32（>>5）确定的当前像素在直方图的哪个竖条里，
										//   也就是知道这个像素对应选择的梯度阈值是多少
										//! 可以确定是每个grid, 32格大小
										// thsStep是类成员变量，表示求梯度直方图的时候，可以划分成多少个32*32的网格
										float pixelTH0 = thsSmoothed[(xf >> 5) + (yf >> 5) * thsStep];
										float pixelTH1 = pixelTH0 * dw1;
										float pixelTH2 = pixelTH1 * dw2;

										// Step 1 先判断第0层里面能不能找到梯度满足阈值的点
										float ag0 = mapmax0[idx]; // 第0层梯度模
										//? 为啥这里还要把求得梯度阈值*一个倍数？
										if (ag0 > pixelTH0 * thFactor)
										{
											Vec2f ag0d = map0[idx].tail<2>();				// 后两位是像素梯度
											//; 还要算这个像素在随机方向上的梯度
											float dirNorm = fabsf((float)(ag0d.dot(dir2))); // 以这个方向上的梯度来判断
											//; setting_selectDirectionDistribution = true，所以这里不进入if
											if (!setting_selectDirectionDistribution)
												dirNorm = ag0;

											if (dirNorm > bestVal2) // 取梯度最大的
											{
												bestVal2 = dirNorm;
												bestIdx2 = idx;
												//; 只要在第0层选择到了像素超过阈值的点，那么就不在第1层和第2层选择了
												bestIdx3 = -2;
												bestIdx4 = -2;
											}
										}

										if (bestIdx3 == -2)
											continue; // 有了则不在其它层选点, 但是还会在该pot里选最大的

										// Step 2 第0层没找到，再到第1层找
										//; 当前像素变换到第1层，对应位置的像素的梯度
										float ag1 = mapmax1[(int)(xf * 0.5f + 0.25f) + (int)(yf * 0.5f + 0.25f) * w1]; // 第1层
										if (ag1 > pixelTH1 * thFactor)
										{
											Vec2f ag0d = map0[idx].tail<2>();
											float dirNorm = fabsf((float)(ag0d.dot(dir3)));
											if (!setting_selectDirectionDistribution)
												dirNorm = ag1;

											if (dirNorm > bestVal3)
											{
												bestVal3 = dirNorm;
												bestIdx3 = idx;
												bestIdx4 = -2;
											}
										}
										if (bestIdx4 == -2)
											continue;

										// Step 3 第1层没找到，再到第2层找
										float ag2 = mapmax2[(int)(xf * 0.25f + 0.125) + (int)(yf * 0.25f + 0.125) * w2]; // 第2层
										if (ag2 > pixelTH2 * thFactor)
										{
											Vec2f ag0d = map0[idx].tail<2>();
											float dirNorm = fabsf((float)(ag0d.dot(dir4)));
											if (!setting_selectDirectionDistribution)
												dirNorm = ag2;

											if (dirNorm > bestVal4)
											{
												bestVal4 = dirNorm;
												bestIdx4 = idx;
											}
										}
									}
								}  // 遍历完第0层的3x3像素

								// 第0层的pot循环完, 若有则添加标志
								if (bestIdx2 > 0)
								{
									map_out[bestIdx2] = 1;
									// 高层pot中有更好的了，满足更严格要求的，就不用满足pixelTH1的了
									// bug bestVal3没有什么用，因为bestIdx3=-2直接continue了
									bestVal3 = 1e10; // 第0层找到了, 就不在高层找了
									n2++;			 // 计数
								}
							}
						}  // 遍历完第1层金字塔覆盖范围的6x6个像素（也就是对应第1层金字塔的3x3像素范围）

						// 第0层没有, 则在第1层选
						//; 1.遍历完第0层的4块3*3个像素之后，没有找到符合阈值的点。但是把这个点变换到第1层金字塔满足
						//;   第1层的金字塔要求，那么也可以选择为特征点
						//; 2.注意：只要在第0层找到了，bestIdx3就是-2；如果第0层没找到，第1层也没找到，那么
						//;   bestIdx3就还是初始值-1
						if (bestIdx3 > 0)
						{
							map_out[bestIdx3] = 2;
							bestVal4 = 1e10;
							n3++;
						}
					}
				}  // 遍历完第2层金字塔覆盖范围的12x12个像素（也就是对应第2层金字塔的3x3像素范围）
				// 第1层没有, 则在第2层选
				if (bestIdx4 > 0)
				{
					map_out[bestIdx4] = 4;
					n4++;
				}
			}
		}
		return Eigen::Vector3i(n2, n3, n4); // 第0, 1, 2层选点的个数
	}
}
