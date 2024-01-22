#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <stdio.h>

#define BLUR_SIZE 12

__global__ 
void matMulNaiveKernel(float* M, float* N, float* P, int m, int n, int k)
{
	// m x k, k x n
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col < n && row < m)
	{
		float pValue = 0;
		for (int i = 0; i < k; i++)
		{
			pValue += M[row * k + i] * N[n * i + col];
		}
		P[row * n + col] = pValue;
	}
}

inline unsigned int cdiv(unsigned int a, unsigned int b)
{
	return (a + b - 1) / b;
}

torch::Tensor matMulNaive(torch::Tensor A, torch::Tensor B) 
{
	// A: m x k 
	// B: k x n 
	assert(A.device().type() == torch::kCUDA);
	assert(B.device().type() == torch::kCUDA);
	assert(A.dtype() == torch::kF32);
	assert(B.dtype() == torch::kF32);

	const auto m = A.size(0);
	const auto k = A.size(1);
	const auto n = B.size(1);

	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
	auto result = torch::zeros({m, n}, options);

	dim3 threads(16, 16);
	dim3 blocks(cdiv(m, threads.x), cdiv(n, threads.y));


	matMulNaiveKernel<<<blocks, threads>>>(
			A.data_ptr<float>(),
			B.data_ptr<float>(),
			result.data_ptr<float>(),
			m,
			n,
			k
	);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	
	return result;
}
