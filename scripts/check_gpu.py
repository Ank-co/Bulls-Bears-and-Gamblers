"""Sanity check of the GPU stack before any training. Exit code 0 only if CUDA works end to end."""
import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("FAIL torch is not installed")
        return 1

    print(f"torch          {torch.__version__}")
    print(f"built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL torch.cuda.is_available() is False")
        return 1

    dev = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device         {dev.name}")
    print(f"capability     sm_{cap[0]}{cap[1]}")
    print(f"VRAM           {dev.total_memory / 2**30:.1f} GiB")
    print(f"bf16 supported {torch.cuda.is_bf16_supported()}")

    arch_list = torch.cuda.get_arch_list()
    if f"sm_{cap[0]}{cap[1]}" not in arch_list:
        print(f"FAIL this torch build has no kernels for sm_{cap[0]}{cap[1]} (has {arch_list}). "
              "Install a build for CUDA 12.8 or newer.")
        return 1

    a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    b = (a @ a).float().sum().item()
    torch.cuda.synchronize()
    print(f"bf16 matmul    ok ({b:.3e})")
    print("OK GPU ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
