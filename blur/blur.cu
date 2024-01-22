#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define BLUR_SIZE 12

__global__ 
void blurKernel(unsigned char* in, unsigned char* out, int w, int h)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int ch = threadIdx.z;

	int baseOffset = w * h * ch;
	if (col < w && row < h)
	{
		auto pixVal = 0;
		auto pixels = 0;

		for (int blurRow=-BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++)
			for (int blurCol=-BLUR_SIZE; blurCol <= BLUR_SIZE; blurCol++)
			{
				int curRow = row + blurRow;
				int curCol = col + blurCol;

				if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w)
				{
					pixVal += in[baseOffset + curRow * w + curCol];
					pixels++;
				}
			}
		out[baseOffset + row * w + col] = (unsigned char) (pixVal / pixels);
	}
}

inline unsigned int cdiv(unsigned int a, unsigned int b)
{
	return (a + b - 1) / b;
}

torch::Tensor blur(torch::Tensor image) 
{
	assert(image.device().type() == torch::kCUDA);
	assert(image.dtype() == torch::kByte);

	const auto channels = image.size(0);
	const auto height = image.size(1);
	const auto width = image.size(2);


	auto result = torch::empty_like(image);

	dim3 threads(16, 16, channels);
	dim3 blocks(cdiv(width, threads.x), cdiv(height, threads.y));

	blurKernel<<<blocks, threads, 0, torch::cuda::getCurrentCUDAStream()>>>(
			image.data_ptr<unsigned char>(),
			result.data_ptr<unsigned char>(),
			width,
			height
	);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
	
	return result;
}
