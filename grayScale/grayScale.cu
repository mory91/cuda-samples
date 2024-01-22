#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>


__global__ 
void colorToGrayScaleConversionKernel(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if (col < width && row < height)
	{
		auto grayOffset = row * width + col;
		auto rgbOffset = grayOffset * 3;

		auto r = Pin[rgbOffset];
		auto g = Pin[rgbOffset + 1];
		auto b = Pin[rgbOffset + 2];

		Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
	}
}

inline unsigned int cdiv(unsigned int a, unsigned int b)
{
	return (a + b - 1) / b;
}

torch::Tensor toGrayscale(torch::Tensor image) 
{
	assert(image.device().type() == torch::kCUDA);
	assert(image.dtype() == torch::kByte);

	const auto height = image.size(0);
	const auto width = image.size(1);


	auto result = torch::empty({height, width, 1}, torch::TensorOptions().dtype(torch::kByte).device(image.device()));

	dim3 threads(16, 16);
	dim3 blocks(cdiv(width, threads.x), cdiv(height, threads.y));

	colorToGrayScaleConversionKernel<<<blocks, threads, 0, torch::cuda::getCurrentCUDAStream()>>>(
			result.data_ptr<unsigned char>(),
			image.data_ptr<unsigned char>(),
			width,
			height
	);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	
	return result;
}
