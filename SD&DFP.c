/*
* 名称：最速下降法和DFP拟牛顿法
* 语言：C语言
* 作者：keke2014
* 时间：2013.01.10
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define pi 3.1415926

/* 求函数1在指定点的值 */
double fun1(double **vars, int n)
{
	int i;
	double sum = 0;
	for(i=0; i<n; ++i)
		sum += vars[i][0]*vars[i][0] - 10*cos(2*pi*vars[i][0]) + 10;
	return sum;
}

/* 求函数2在指定点的值 */
double fun2(double **vars, int n)
{
	int i;
	double sum = 0, temp = 1;
	for(i=0; i<n; ++i)
	{
		temp *= cos(vars[i][0]/sqrt(i+1));
		sum += vars[i][0]*vars[i][0];
	}
	sum = sum/4000 - temp + 1;
	return sum;
}

// 还可以继续添加函数……

/* 根据给定的行列数创建矩阵 */
double ** cre_mat(int x, int y)
{
	int i, j;
	double **m = (double **)malloc(x * sizeof(double *));
	if(m == NULL) return 0;
	for(i=0; i<x; ++i)
	{
		m[i] = (double *)malloc(y * sizeof(double));
		if(m[i] == NULL) return 0;
	}

	/* 初始化 */
	for(i=0; i<x; ++i)
		for(j=0; j<y; ++j)
			m[i][j] = 0.0;

	return m;
}

/* 释放矩阵暂用的内存 */
void free_mat(double **m, int x)
{
	int i;
	for(i=0; i<x; ++i)
		free(m[i]);
	free(m);
}

/* 求矩阵的逆 */
void mat_tra(double **m1, double **m2, int x, int y)
{
	int i, j;
	for(i=0; i<x; ++i)
		for(j=0; j<y; ++j)
			m2[j][i] = m1[i][j];
}

/* 复制矩阵 */
void cpy_mat(double **m1, double **m2, int x, int y)
{
	int i, j;
	for(i=0; i<x; ++i)
		for(j=0; j<y; ++j)
			m2[i][j] = m1[i][j];
}

/* 矩阵相加 */
void mat_add(double **m1, double **m2, double **m3, int x, int y)
{
	int i, j;
	for(i=0; i<x; ++i)
		for(j=0; j<y; ++j)
			m3[i][j] = m1[i][j] + m2[i][j];
}

/* 矩阵相减 */
void mat_sub(double **m1, double **m2, double **m3, int x, int y)
{
	int i, j;
	for(i=0; i<x; ++i)
		for(j=0; j<y; ++j)
			m3[i][j] = m1[i][j] - m2[i][j];
}

/* 矩阵相乘 */
void mat_mul(double **m1, double **m2, double **m3, int x, int y, int z)
{
	int i, j, k;

	/* 初始化输出矩阵 */
	for(i=0; i<x; ++i)
		for(j=0; j<z; ++j)
			m3[i][j] = 0.0;

	for(i=0; i<x; ++i)
		for(j=0; j<z; ++j)
			for(k=0; k<y; ++k)
				m3[i][j] += m1[i][k]*m2[k][j];
}

/* 矩阵数乘 */
void mat_sca_mul(double a, double **m, int x, int y)
{
	int i, j;
	for(i=0; i<x; ++i)
		for(j=0; j<y; ++j)
				m[i][j] = a * m[i][j];
}

/* 用极限方法求指定点的偏导/梯度 */
void differ(double (*f)(double **vars, int n), double **vars, double **grads, int n, double prec)
{
	int i, j;
	double delta_x = 3, delta_y;	// Δx和Δy，Δx初值设为3
	double y1, y2;
	double diff1, diff2;
	double **vars_temp = cre_mat(n, 1);

	/* 分别求x每个分量的偏导 */
	for(i=0; i<n; ++i)
	{
		y1 = f(vars, n);
		for(j=0; j<n; ++j)
		{
			vars_temp[j][0] = vars[j][0];
			if(i == j) vars_temp[j][0] += delta_x;	// 只针对当前的x分量加上Δx
		}
		y2 = f(vars_temp, n);
		delta_y = y2 - y1;
		diff1 = delta_y / delta_x;	// 求一个比值

		while(1)
		{
			delta_x = 0.5 * delta_x;	// 缩小Δx
			y1 = f(vars, n);
			for(j=0; j<n; ++j)
			{
				vars_temp[j][0] = vars[j][0];
				if(i == j)
					vars_temp[j][0] += delta_x;
			}
			y2 = f(vars_temp, n);
			delta_y = y2 - y1;
			diff2 = delta_y / delta_x;	// 再求一个比值

			/* 
			* 如果两次比值之差的绝对值小于设定的精度值，
			* 则第二个比值即为该分量的偏导，
			* 否者不断缩小Δx，直到满足条件的值出现
			*/
			if(fabs(diff2 - diff1) <= prec)
				break;
			else
				diff1 = diff2;
		}
		grads[i][0] = diff2;	// 保存求出的偏导值
	}

	/* 释放堆内存 */
	free_mat(vars_temp, n);
	vars_temp = NULL;
}

/* 成功失败法，用于一维搜索 */
void suc_fail(double (*f)(double **vars, int n), double **vars, double **d, int n, double prec, double h)
{
	int i;
	double alpha = 0;		// 阿尔法初值设为0（α≥0）
	double f1, f2;
	double **vars_temp1 = cre_mat(n, 1);
	double **vars_temp2 = cre_mat(n, 1);

	/* 根据阿尔法和步长求对应的x值 */
	for(i=0; i<n; ++i)
	{
		vars_temp1[i][0] = vars[i][0] - alpha * d[i][0];
		vars_temp2[i][0] = vars[i][0] - (alpha+h) * d[i][0];
	}
	f1 = f(vars_temp1, n);

	/* 核心循环 */
	while(1)
	{
		f2 = f(vars_temp2, n);

		/* f2小于f1，加大步长，下一次前进 */
		if(f2 < f1)		
		{
			alpha += h;		// 更新阿尔法
			for(i=0; i<n; ++i)	// 更新对应的x值
				vars_temp1[i][0] = vars[i][0] - alpha * d[i][0];
			f1 = f2;
			h = 2*h;	// 步长扩大一倍
			for(i=0; i<n; ++i)	// 更新对应的x值
				vars_temp2[i][0] = vars[i][0] - (alpha+h) * d[i][0];
		}

		/* 满足精度要求 */
		else if(fabs(h) <= prec)	
		{
			/* 保存阿尔法对应的x值 */
			for(i=0; i<n; ++i)
				vars[i][0] = vars_temp1[i][0];

			break;	// 算法结束，退出
		}

		/* f2大于f1，缩小步长，下一次后退 */
		else	
		{
			h = -(h/4);		// 更新步长
			for(i=0; i<n; ++i)	// 更新对应的x值
				vars_temp2[i][0] = vars[i][0] - (alpha+h) * d[i][0];
		}
	}	

	/* 释放用于暂存的堆内存 */
	free_mat(vars_temp1, n);
	vars_temp1 = NULL;
	free_mat(vars_temp2, n);
	vars_temp2 = NULL;
}

/* 最速下降法（Speedest Descent Method）*/
void SD(int fun, int n, double prec, double h)
{
	int i, iter = 0;
	double norm;	// 暂存梯度的范数

	double **vars = cre_mat(n, 1);	// 暂存变量x的矩阵
	double **grads = cre_mat(n, 1);	// 暂存梯度的矩阵

	/* 提示输入变量x的初值 */
	printf("Please input the initial value: ");
	fflush(stdin);	// 清空标准IO缓存，防止干扰下面变量的读取
	for(i=0; i<n; ++i)
		scanf("%lf", &vars[i][0]);

	printf("\nIterating...\n");	// 迭代中
	/* 核心循环 */
	while(1)
	{
		++iter;
		printf("iter=%d\n", iter);	// 输出iter，指示迭代次数

		switch(fun)	// 求指定函数的偏导
		{
		case 1:	differ(fun1, vars, grads, n, prec);break;
		case 2:	differ(fun2, vars, grads, n, prec);break;
		default: break;
		}

		/* 求梯度的二范数 */
		norm = 0;
		for(i=0; i<n; ++i)
			norm += grads[i][0] * grads[i][0];
		norm = sqrt(norm);

		/* 梯度范数满足精度 */
		if(norm <= prec)
		{
			/* 输出结果 */
			printf("\nThe answer: \n");
			for(i=0; i<n; ++i)
				printf("x%d = %g \n", i+1, vars[i][0]);
			switch(fun)
			{
			case 1:	printf("min(f1) = %g \n", fun1(vars, n));break;
			case 2:	printf("min(f2) = %g \n", fun2(vars, n));break;
			default: break;
			}

			/* 是否变量和梯度的内存 */
			free_mat(vars, n);
			vars = NULL;
			free_mat(grads, n);
			grads = NULL;

			break;	// 算法结束，退出
		}
		/* 梯度范数不满足精度，进行一维搜索 */
		else
		{
			switch(fun)	// 对指定函数进行一维搜索
			{
			case 1:	suc_fail(fun1, vars, grads, n, prec, h);break;
			case 2:	suc_fail(fun2, vars, grads, n, prec, h);break;
			default: break;
			}
		}
	}
	
}

/* DFP拟牛顿法 */
void DFP(int fun, int n, double prec, double h)
{
	int i, j, iter = 0, flag = 0;
	double norm, a, b;

	/* 变量x、梯度g、搜索方向d等的矩阵 */
	double **vars=cre_mat(n, 1), **grads=cre_mat(n, 1), **H=cre_mat(n, n), **d=cre_mat(n, 1);
	double **vars_temp=cre_mat(n, 1), **grads_temp=cre_mat(n, 1);

	/* 求ΔH的暂存矩阵 */
	double **a_temp=cre_mat(1, 1), **dgt_H=cre_mat(1, n), **b_temp=cre_mat(1, 1);
	double **delta_x=cre_mat(n, 1), **delta_g=cre_mat(n, 1);
	double **delta_x_t=cre_mat(1, n), **delta_g_t=cre_mat(1, n);
	double **H_dg=cre_mat(n, 1), **H_dg_dgt=cre_mat(n, n);
	double **H_temp1=cre_mat(n, n), **H_temp2=cre_mat(n, n);
	
	/* 提示输入变量x的初值 */
	printf("Please input the initial value: ");
	fflush(stdin);	// 清空标准IO缓存，防止干扰下面变量的读取
	for(i=0; i<n; ++i)
		scanf("%lf", &vars[i][0]);
	
	printf("\nIterating...\n");	// 迭代中
	/* 核心循环 */
	while(1)
	{
		++iter;	
		printf("iter=%d\n", iter);	// 输出iter，指示迭代次数
		++flag;	// 标识是否要求ΔH（iter≥2）以及是否要重新迭代

		if(1 == flag)
		{
			/* 初始化H为单位矩阵 */
			for(i=0; i<n; ++i)
				for(j=0; j<n; ++j)
					if(i == j)
						H[i][j] = 1;
		}

		cpy_mat(grads, grads_temp, n, 1);	// 备份梯度，用于求Δg
		switch(fun)	// 对指定函数求偏导
		{
		case 1:	differ(fun1, vars, grads, n, prec);break;
		case 2:	differ(fun2, vars, grads, n, prec);break;
		default: break;
		}

		/* 求梯度的二范数 */
		norm = 0;
		for(i=0; i<n; ++i)
			norm += grads[i][0] * grads[i][0];
		norm = sqrt(norm);

		/* 梯度范数满足精度要求 */
		if(norm <= prec)
		{
			/* 输出结果 */
			printf("\nThe answer: \n");
			for(i=0; i<n; ++i)
				printf("x%d = %g \n", i+1, vars[i][0]);
			switch(fun)
			{
			case 1:	printf("min(f1) = %g \n", fun1(vars, n));break;
			case 2:	printf("min(f2) = %g \n", fun2(vars, n));break;
			default: break;
			}

			// 释放内存（由于可能多次运行，必须释放，否者泄露导致程序异常退出）
			free_mat(vars, n);
			vars = NULL;
			free_mat(grads, n);
			grads = NULL;
			free_mat(H, n);
			H = NULL;
			free_mat(d, n);
			d = NULL;

			free_mat(a_temp, 1);
			a_temp = NULL;
			free_mat(dgt_H, 1);
			dgt_H = NULL;
			free_mat(b_temp, 1);
			b_temp = NULL;
			
			free_mat(vars_temp, n);
			vars_temp = NULL;
			free_mat(grads_temp, n);
			grads_temp = NULL;
			free_mat(delta_x, n);
			delta_x = NULL;
			free_mat(delta_g, n);
			delta_g = NULL;
			free_mat(delta_x_t, 1);
			delta_x_t = NULL;
			free_mat(delta_g_t, 1);
			delta_g_t = NULL;
				
			free_mat(H_dg, n);
			H_dg = NULL;
			free_mat(H_dg_dgt, n);
			H_dg_dgt = NULL;
			free_mat(H_temp1, n);
			H_temp1 = NULL;
			free_mat(H_temp2, n);
			H_temp2 = NULL;
			
			break;	// 算法结束，退出
		}

		/* 梯度范数不满足精度要求，继续迭代 */
		else
		{
			/* 求ΔH */
			if(flag > 1)
			{
				if(n == flag)	//	迭代次数达到n，开始新一轮迭代
					flag = 0;
				else
				{
					/* 求Δx和Δg */
					mat_sub(vars, vars_temp, delta_x, n, 1);
					mat_sub(grads, grads_temp, delta_g, n, 1);
					
					/* 求Δx和Δg的转置 */
					mat_tra(delta_x, delta_x_t, n, 1);
					mat_tra(delta_g, delta_g_t, n, 1);
					
					/* 分步求ΔH */
					mat_mul(delta_g_t, H, dgt_H, 1, n, n);
					mat_mul(dgt_H, delta_g, a_temp, 1, n, 1);
					a = a_temp[0][0];
					a = 1/a;
					
					mat_mul(delta_x_t, delta_g, b_temp, 1, n, 1);
					b = b_temp[0][0];
					b = 1/b;
					
					mat_mul(H, delta_g, H_dg, n, n, 1);
					mat_mul(H_dg, delta_g_t, H_dg_dgt, n, 1, n);
					mat_mul(H_dg_dgt, H, H_temp1, n, n, n);
					mat_sca_mul(a, H_temp1, n, n);
					
					mat_mul(delta_x, delta_x_t, H_temp2, n, 1, n);
					mat_sca_mul(b, H_temp2, n, n);
					
					/* 求最终的新H */
					mat_sub(H, H_temp1, H, n, n);
					mat_add(H, H_temp2, H, n, n);
				}
			}

			if(flag > 0)	// 如果是新一轮迭代(flag==0)，则跳过下面的处理
			{
				mat_mul(H, grads, d, n, n, 1);	// 求搜索方向（这里未加负号，在成功失败算法里加）
				cpy_mat(vars, vars_temp, n, 1);	// 备份变量，用于求Δx
				
				switch(fun)	// 对指定函数进行一维搜索
				{
				case 1:	suc_fail(fun1, vars, d, n, prec, h);break;
				case 2:	suc_fail(fun2, vars, d, n, prec, h);break;
				default: break;
				}
			}
		}
	}
}

/* 主函数 */
int main()
{
	int n, met, fun;	// met-method-方法，fun-function-函数
	double prec, h;	// 算法精度prec和一维搜索的步长h

	while(1)
	{
		do
		{	/* 选择方法：最速下降法或DFP */
			printf("\nPlease select the methord(1 or 2 , 0 to quit): \n");
			printf(" 0: quit \n 1: SD(steepest descent) \n 2: DFP \n");
			printf("Your selected methord: ");
			fflush(stdin);	// 清空标准IO缓存，防止干扰下面变量的读取
			scanf("%d", &met);
		}while(met<0 || met >2); // 选择错误，重新选择
		if(met == 0) break;	// 用户选择退出

		do
		{	/* 选择函数 */
			printf("\nPlease select the function(1 or 2): \n");
			printf(" 1: f1=Σ[x^2-10*cos(2*π*x)+10] \n");
			printf(" 2: f2=(Σx^2)/4000-∏cos(x/i)+1] \n");
			printf("Your selected function: ");
			fflush(stdin);	// 清空标准IO缓存，防止干扰下面变量的读取
			scanf("%d", &fun);
		}while(fun<1 || fun>2);	// 选择错误，重新选择
		
		do
		{
			/* 输入算法精度 */
			printf("\nPlease input the precision(prec>0): ");
			fflush(stdin);	// 清空标准IO缓存，防止干扰下面变量的读取
			scanf("%lf", &prec);
		}while(prec<=0);	// 维度有误，重新输入

		do
		{
			/* 输入算法精度 */
			printf("Please input the step size(h>0): ");
			fflush(stdin);	// 清空标准IO缓存，防止干扰下面变量的读取
			scanf("%lf", &h);
		}while(prec<=0);	// 维度有误，重新输入

		do
		{
			/* 输入变量维度 */
			printf("\nPlease input the dimension(n>=1): ");
			fflush(stdin);	// 清空标准IO缓存，防止干扰下面变量的读取
			scanf("%d", &n);
		}while(n<1);	// 维度有误，重新输入

		switch(met)	// 进入指定方法的函数
		{
		case 1: SD(fun, n, prec, h);break;
		case 2: DFP(fun, n, prec, h);break;
		default: break;
		}
	}
}
