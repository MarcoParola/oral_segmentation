"""Bounded, read-only environment probe. No installation or environment mutation.

Exit codes: 0=pass, 2=warn/review needed, 1=fail. NPU compatibility is never
inferred from version strings alone: check the product-specific official matrix.
"""
import argparse
import importlib.metadata as metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

PACKAGES = ("torch", "torchvision", "torch-npu", "pytorch-lightning", "segmentation-models-pytorch", "numpy")


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def bounded_command(command, timeout=30):
    # No shell expansion; callers supply only fixed commands, never credentials.
    try:
        proc = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
        return {"exit_code": proc.returncode, "stdout": proc.stdout[-16000:], "stderr": proc.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "error": f"timeout after {timeout}s"}
    except OSError as exc:
        return {"exit_code": None, "error": type(exc).__name__}


def runtime_probe(device):
    """Executed only in a bounded child: isolate imports/driver discovery hangs."""
    import torch
    import torchvision
    facts = {"torch": torch.__version__, "torchvision": torchvision.__version__, "torch_cuda_build": torch.version.cuda}
    if device == "npu":
        import torch_npu
        facts["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
        facts["available"] = torch.npu.is_available()
        facts["device_count"] = torch.npu.device_count() if facts["available"] else 0
        if facts["available"]:
            facts["device_name"] = str(torch.npu.get_device_name(0))
    elif device == "cuda":
        facts["available"] = torch.cuda.is_available()
        facts["device_count"] = torch.cuda.device_count()
        if facts["available"]:
            facts["device_name"] = torch.cuda.get_device_name(0)
    else:
        facts["available"] = True
        facts["device_count"] = 1
    return facts


def parse_framework_version(value):
    """Small strict parser for published torch wheels, no packaging bootstrap dependency."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:(?:\.post|\.dev|rc)\d+)?(?:\+([a-z0-9.]+))?", value)
    if not match:
        raise ValueError("Unrecognized framework wheel version")
    return tuple(int(match[i]) for i in (1, 2, 3)), match[4]


def version_issues(versions, device):
    """Local consistency checks, NOT the CANN/firmware/product compatibility matrix."""
    issues = []
    required = ["torch", "torchvision"] + (["torch-npu"] if device == "npu" else [])
    for name in required:
        if not versions.get(name):
            issues.append(("fail", f"missing {name}"))
    if any(not versions.get(n) for n in required):
        return issues
    try:
        torch_v, torch_local = parse_framework_version(versions["torch"])
        vision_v, vision_local = parse_framework_version(versions["torchvision"])
        if torch_local and vision_local and torch_local != vision_local:
            issues.append(("fail", "torch/torchvision local platform build tags differ"))
        # This project only validated 2.1.x + 0.16.x. Other pairs require a new baseline.
        if torch_v[:2] == (2, 1) and vision_v[:2] != (0, 16):
            issues.append(("fail", "torch 2.1 requires torchvision 0.16.x for this baseline"))
        elif torch_v[:2] != (2, 1):
            issues.append(("warn", "outside tested torch 2.1 / torchvision 0.16 baseline; manual pair validation required"))
        elif torch_v[2:] != vision_v[2:]:
            issues.append(("fail", "torch/torchvision patch versions differ from paired baseline"))
        if device == "npu":
            npu_v, _ = parse_framework_version(versions["torch-npu"])
            if npu_v != torch_v:
                issues.append(("fail", "torch-npu base version does not match torch"))
            if torch_local and "cu" in torch_local:
                issues.append(("fail", "CUDA torch build must not be used for this NPU bring-up"))
    except ValueError:
        issues.append(("fail", "unparseable framework version; cannot protect dependencies"))
    return issues


def assess(device, versions, probe, hardware=None):
    checks = [{"status": s, "message": m} for s, m in version_issues(versions, device)]
    if probe.get("error"):
        checks.append({"status": "fail", "message": "framework/device probe failed: " + probe["error"]})
    elif not probe.get("available") or probe.get("device_count", 0) < 1:
        checks.append({"status": "fail", "message": f"requested {device} unavailable; no fallback"})
    else:
        checks.append({"status": "pass", "message": f"{device} import and device discovery succeeded"})
    if device == "npu":
        if probe.get("torch_cuda_build") is not None:
            checks.append({"status": "fail", "message": "CUDA-enabled torch rejected for NPU environment"})
        if not hardware or hardware.get("npu_smi", {}).get("exit_code") != 0:
            checks.append({"status": "warn", "message": "npu-smi unavailable/failed; chip identity unknown"})
        if not hardware or not hardware.get("version_files"):
            checks.append({"status": "warn", "message": "driver/CANN version files unknown; confirm installation path"})
        checks.append({"status": "warn", "message": "910 vs 910B product, architecture, driver/CANN/torch_npu official matrix requires manual confirmation; not certified compatible"})
    status = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "pass"
    return status, checks


def collect(device, timeout=30, ascend_root=Path("/usr/local/Ascend")):
    versions = package_versions()
    result = bounded_command([sys.executable, "-m", "scripts.preflight", "--worker", "--device", device], timeout)
    probe = {"error": result.get("error", "import/discovery failed")}
    if result.get("exit_code") == 0:
        try:
            probe = json.loads(result["stdout"].split("ORAL_PROBE_JSON=")[-1])
        except (ValueError, KeyError):
            probe = {"error": "invalid worker JSON"}
    hardware = None
    if device == "npu":
        files = {}
        # Only bounded known version files, no environment dump, config secrets or recursive system scan.
        for relative in ("driver/version.info", "ascend-toolkit/latest/version.cfg", "ascend-toolkit/latest/ascend_toolkit_install.info"):
            path = ascend_root / relative
            try:
                with path.open(encoding="utf-8", errors="replace") as stream:
                    files[relative] = stream.read(4096)
            except OSError:
                pass
        hardware = {"npu_smi": bounded_command(["npu-smi", "info"], timeout), "version_files": files,
                    "ascend_root": str(ascend_root), "chip_classification": "manual confirmation required"}
    status, checks = assess(device, versions, probe, hardware)
    return {"schema": 1, "status": status, "device": device, "scope": "single-device FP32 eager; no Lightning NPU/AMP/compile/DDP",
            "python": platform.python_version(), "platform": platform.platform(), "architecture": platform.machine(),
            "packages": versions, "runtime": probe, "probe_process": result, "ascend": hardware, "checks": checks,
            "official_matrix": "https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md",
            "compatibility_certified": False}


def write_new(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), default="cpu")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--ascend-root", type=Path, default=Path("/usr/local/Ascend"))
    parser.add_argument("--output", default="artifacts/preflight.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print("ORAL_PROBE_JSON=" + json.dumps(runtime_probe(args.device)))
        return
    if not 1 <= args.timeout <= 120:
        parser.error("timeout must be 1..120 seconds")
    report = collect(args.device, args.timeout, args.ascend_root)
    write_new(args.output, json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "checks": report["checks"], "output": args.output}, indent=2))
    raise SystemExit({"pass": 0, "warn": 2, "fail": 1}[report["status"]])


if __name__ == "__main__":
    main()
