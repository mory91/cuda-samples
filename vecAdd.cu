#include <stdio.h>

__global__ void vecAddKernel(float* A, float* B, float* C, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < n)
	{
		C[i] = A[i] + B[i];
	}
}

void vecAdd(float* A, float* B, float* C, int n)
{
	float *A_d, *B_d, *C_d;
	int size = n * sizeof(float);

	cudaMalloc((void**)&A_d, size);
	cudaMalloc((void**)&B_d, size);
	cudaMalloc((void**)&C_d, size);

	cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

	vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

	cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);
}

int main()
{
	float *A_h, *B_h, *C_h;
	int n = 200000;

	A_h = (float*)malloc(sizeof(float) * n);
	B_h = (float*)malloc(sizeof(float) * n);
	C_h = (float*)malloc(sizeof(float) * n);

	srand(time(NULL));
	for (int i = 0; i < n; i++)
	{
		A_h[i] = ((float)rand() / (float)RAND_MAX) * 100;
		B_h[i] = ((float)rand() / (float)RAND_MAX) * 100;
	}

	vecAdd(A_h, B_h, C_h, n);

	bool valid = true;
	for (int i = 0; i < n; i++)
		if (A_h[i] + B_h[i] != C_h[i])
		{
			valid = false;
			break;
		}

	if (!valid)
		printf("Not valid");
	else
		printf("Valid");

	free(A_h);
	free(B_h);
	free(C_h);
}
