from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline

def compile_ext():
    cuda_src = Path("matmul.cu").read_text()
    cpp_src = "torch::Tensor matMulNaive(torch::Tensor A, torch::Tensor B);"

    matmul_ext = load_inline(
        name="matmul_ext",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["matMulNaive"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"]
    )

    return matmul_ext


def main():
    ext = compile_ext()
    a = torch.randn((20, 20)).cuda()
    b = torch.randn((20, 30)).cuda()

    assert a.dtype == torch.float32
    assert b.dtype == torch.float32

    c = ext.matMulNaive(a, b)

    print("All close:", torch.allclose(c, a @ b))



if __name__ == "__main__":
    main()
