#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include"cuda.h"
 
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int dim0, unsigned int dim1);
 
__global__ void addKernel(int *c, const int *a, const int *b)
{
	//int i =(blockIdx.z*blockDim.x*blockDim.y+blockIdx.y*blockDim.x + blockIdx.x)+threadIdx.x;
	int i = blockIdx.y*gridDim.x*blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	c[i] = powf(2 * a[i] * b[i],2);
}
 
extern "C" double addnum(int *c, const int *a, const int *b, unsigned int dim0, unsigned int dim1)
{
	cudaError_t cudaStatus = addWithCuda(c, a, b, dim0, dim1);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
 
}
 
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int dim0,unsigned dim1)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	int size = dim0*dim1;
	cudaError_t cudaStatus;
 
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
 
	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
 
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
 
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
 
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
 
	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
 
	// Launch a kernel on the GPU with one thread for each element.
	dim3 DG(dim0 / 512, 512);
	addKernel<<<DG, 512>>>(dev_c, dev_a, dev_b);
 
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
 
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
 
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
 
Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
 
	return cudaStatus;
}