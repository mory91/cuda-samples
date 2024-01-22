from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline

def compile_ext():
    cuda_src = Path("blur.cu").read_text()
    cpp_src = "torch::Tensor blur(torch::Tensor image);"

    grayscale_ext = load_inline(
        name="blur_ext",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["blur"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"]
    )

    return grayscale_ext


def main():
    ext = compile_ext()
    x = read_image("grace_hopper.jpg").contiguous().cuda()
    print("mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)

    assert x.dtype == torch.uint8

    y = ext.blur(x)

    print("Output image:", y.shape, y.dtype)
    print("mean", y.float().mean())
    write_png(y.cpu(), "output.png")


if __name__ == "__main__":
    main()
