"""Snapshot installed framework pins BEFORE installing runtime dependencies.

Does not run pip. Use constraints.txt with pip/uv, then run verify to detect
replacement. A version pin protects the installed version, not wheel provenance;
keep approved wheels/checksums and never use this as a CANN compatibility claim.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.preflight import collect, package_versions, write_new, parse_framework_version

FRAMEWORKS = ("torch", "torchvision", "torch-npu")


def protected_versions(versions, device):
    required = ("torch", "torchvision", "torch-npu") if device == "npu" else ("torch", "torchvision")
    if any(not versions.get(name) for name in required):
        raise ValueError("Required framework missing; install approved platform wheels first")
    return {name: versions[name] for name in FRAMEWORKS if versions.get(name)}


def constraints_text(versions):
    lines = ["# Generated from the already installed, approved platform environment.",
             "# Exact local build tags retained. Not a compatibility certification."]
    for name, version in sorted(versions.items()):
        parse_framework_version(version)  # fail closed for malformed/injected versions
        lines.append(f"{name}=={version}")
    return "\n".join(lines) + "\n"


def verify_guard(guard, current):
    mismatches = {name: {"expected": version, "actual": current.get(name)}
                  for name, version in guard["packages"].items() if current.get(name) != version}
    # Unexpected newly installed torch_npu also changes the platform.
    for name in FRAMEWORKS:
        if current.get(name) and name not in guard["packages"]:
            mismatches[name] = {"expected": None, "actual": current[name]}
    return {"status": "fail" if mismatches else "pass", "mismatches": mismatches,
            "meaning": "version stability only, not wheel provenance/driver/CANN compatibility"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    create = sub.add_parser("create")
    create.add_argument("--device", choices=("cpu", "cuda", "npu"), required=True)
    create.add_argument("--output", required=True, help="new output directory")
    create.add_argument("--timeout", type=int, default=30)
    verify = sub.add_parser("verify")
    verify.add_argument("guard", type=Path)
    verify.add_argument("--output", required=True)
    args = p.parse_args()
    if args.mode == "create":
        if not 1 <= args.timeout <= 120:
            p.error("timeout must be 1..120")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        report = collect(args.device, args.timeout)
        write_new(output / "preflight.json", json.dumps(report, indent=2))
        if report["status"] == "fail":
            print("FAIL: no constraints exported; fix missing/inconsistent platform first")
            raise SystemExit(1)
        pins = protected_versions(report["packages"], args.device)
        text = constraints_text(pins)
        write_new(output / "constraints.txt", text)
        guard = {"schema": 1, "device": args.device, "packages": pins, "preflight_status": report["status"],
                 "constraints_sha256": hashlib.sha256(text.encode()).hexdigest(), "compatibility_certified": False}
        write_new(output / "guard.json", json.dumps(guard, indent=2))
        print(json.dumps(guard, indent=2))
        raise SystemExit(2 if report["status"] == "warn" else 0)
    guard = json.loads(args.guard.read_text(encoding="utf-8"))
    if guard.get("schema") != 1:
        raise ValueError("Unsupported guard schema")
    # Detect accidental alteration of the constraints handed to pip.
    constraints = args.guard.parent / "constraints.txt"
    if hashlib.sha256(constraints.read_bytes()).hexdigest() != guard["constraints_sha256"]:
        raise ValueError("constraints.txt hash differs from guard")
    result = verify_guard(guard, package_versions())
    write_new(args.output, json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result["status"] == "fail" else 0)


if __name__ == "__main__":
    main()
