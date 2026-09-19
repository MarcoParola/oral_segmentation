"""Explicit CPU/CUDA/NPU selection; no accelerator fallback or transfer_to_npu."""
import importlib
import re
import torch


def resolve_device(spec):
    if not re.fullmatch(r"cpu|cuda(?::\d+)?|npu(?::\d+)?", spec):
        raise ValueError("Device must be cpu, cuda[:index], or npu[:index]")
    if spec == "cpu":
        return torch.device("cpu")
    kind, _, index = spec.partition(":")
    index = int(index or 0)
    if kind == "cuda":
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {spec}, but CUDA/device is unavailable; no fallback")
        torch.cuda.set_device(index)
    else:
        try:
            importlib.import_module("torch_npu")
        except (ImportError, OSError) as exc:
            raise RuntimeError("NPU requested: install matching torch_npu/CANN/driver and source set_env.sh; no fallback") from exc
        if not hasattr(torch, "npu") or not torch.npu.is_available() or index >= torch.npu.device_count():
            raise RuntimeError(f"Requested {spec}, but NPU/device is unavailable; no fallback")
        torch.npu.set_device(index)
        # FP32 eager initial bring-up. No JIT/AMP/compile/DDP migration yet.
        if hasattr(torch.npu, "set_compile_mode"):
            torch.npu.set_compile_mode(jit_compile=False)
    return torch.device(f"{kind}:{index}")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        torch.npu.synchronize()
