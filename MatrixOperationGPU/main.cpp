

#include<memory>
#include<math.h>
#include"ipps.h"
#include"ippvm.h"

extern "C" double addnum(int *c, const int *a, const int *b, unsigned int dim0, unsigned int dim1);

int main()
{
	int dim[2] = { 512*512,512 };
	int N = dim[0]*dim[1]; //* 512 * 320;
	int* a = new int[N];
	int* b = new int[N];
	for (int i = 0; i < N; i++)
	{
		a[i] = 1;
		b[i] = 2;
	}
	int* c = new int[N];
	float* c1 = new float[N];
	float* c2 = new float[N];
	//addnum(c, b, a, dim[0], dim[1]);
	//for (int i = 0; i < N; i++)//274ms
	//{
	//	c[i] =pow( 2*a[i] * b[i],2);
	//}
	//ippsAdd_32s_Sfs(a, b, c, N, 0); //275ms
	ippsMulC_32s_ISfs(2, a, N, 0);
	ippsMul_32s_Sfs(a, b, c, N, 0);
	ippsConvert_32s32f(c, c1, N);
	ippsPowx_32f_A21(c1, 2, c2, N);
	int x = c2[0];
	return 0;
}