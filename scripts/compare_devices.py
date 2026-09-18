"""Same-weight, same-real-sample FP32 eager numerical acceptance.

No random-init comparison, no image export. Reference stores logits/loss only;
these still derive from medical data and must remain local/access-controlled.
Only load trusted local smoke model.pt files (weights_only=True, strict load).
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from scripts.preflight import write_new
from scripts.smoke import digest, check_finite
from src.datasets import BinarySegmentationDataset, MultiClassSegmentationDataset
from src.datasets.polygons import image_path
from src.devices import resolve_device, synchronize
from src.models.factory import build_model


SCHEMA = 1


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def pipeline_fingerprint(model_name):
    root = Path(__file__).resolve().parents[1]
    paths = ["src/datasets/polygons.py", "src/datasets/binarySegmentation.py", "src/datasets/multiClassSegmentation.py",
             "src/models/factory.py", f"src/models/{model_name}/{model_name}.py", "scripts/compare_devices.py"]
    # Normalize line endings for Windows -> Linux copies; do NOT hash absolute paths.
    return {name: hashlib.sha256((root / name).read_text(encoding="utf-8").encode()).hexdigest() for name in paths}


def load_weights(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("model") not in ("fcn", "deeplab", "unet"):
        raise ValueError("Expected trusted smoke model.pt with architecture metadata, not a Lightning checkpoint")
    if checkpoint.get("classes") not in (1, 4):
        raise ValueError("Only binary or 3 foreground classes + background supported")
    if checkpoint.get("encoder") not in ("resnet50", "efficientnet-b7"):
        raise ValueError("Unsupported encoder metadata")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Missing state_dict")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Non-tensor state entry {key}")
        check_finite(value, f"weight {key}")
    return checkpoint


def prepare(checkpoint_path, data, split, indices, size):
    checkpoint_path, data = Path(checkpoint_path), Path(data)
    checkpoint = load_weights(checkpoint_path)
    classes = checkpoint["classes"]
    annotation_path = data / f"{split}.json"
    transform = T.Compose([T.Resize((size, size)), T.ToTensor()])
    dataset = BinarySegmentationDataset(annotation_path, transform) if classes == 1 else MultiClassSegmentationDataset(annotation_path, transform, n_classes=3)
    if len(indices) != len(set(indices)) or not indices or any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError("Sample indices must be unique, non-empty and within split")
    images, masks, _ = next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices), num_workers=0)))
    check_finite(images, "images")
    check_finite(masks, "masks")
    records = [dataset.dataset["images"][i] for i in indices]
    identity = {"schema": SCHEMA, "checkpoint_sha256": digest(checkpoint_path),
                "model": checkpoint["model"], "classes": classes, "encoder": checkpoint["encoder"],
                "split": split, "annotation_sha256": digest(annotation_path), "indices": indices,
                "image_ids": [r["id"] for r in records],
                "image_sha256": [digest(image_path(annotation_path, r)) for r in records],
                "categories": dataset.dataset["categories"],
                "preprocessing": {"size": size, "image": "bilinear RGB ToTensor [0,1]", "mask": "nearest all polygons", "augmentation": None},
                "input_shape": list(images.shape), "target_shape": list(masks.shape),
                "input_sha256": tensor_digest(images), "target_sha256": tensor_digest(masks),
                "pipeline_sha256": pipeline_fingerprint(checkpoint["model"]),
                "precision": "FP32", "mode": "eval eager, no autocast", "loss": "BCEWithLogitsLoss" if classes == 1 else "CrossEntropyLoss"}
    return checkpoint, images, masks, identity


def require_identity(expected, actual):
    mismatch = [key for key in sorted(set(expected) | set(actual)) if expected.get(key) != actual.get(key)]
    if mismatch:
        raise ValueError("Reference identity mismatch: " + ", ".join(mismatch))


def compare_arrays(reference, actual, rtol, atol):
    if not all(math.isfinite(v) and v >= 0 for v in (rtol, atol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    if set(reference) != set(actual):
        raise ValueError("Numerical output keys differ")
    stats = {}
    for name in reference:
        expected, observed = np.asarray(reference[name]), np.asarray(actual[name])
        if expected.shape != observed.shape or expected.dtype != observed.dtype:
            raise ValueError(f"Output shape/dtype mismatch: {name}")
        if not np.isfinite(expected).all() or not np.isfinite(observed).all():
            raise ValueError(f"Nonfinite numerical output: {name}")
        diff = np.abs(observed.astype(np.float64) - expected.astype(np.float64))
        scale = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
        stats[name] = {"shape": list(expected.shape), "dtype": str(expected.dtype),
                       "max_abs_error": float(diff.max()), "max_rel_error": float((diff / scale).max()),
                       "mean_abs_error": float(diff.mean()),
                       "allclose": bool(np.allclose(observed, expected, rtol=rtol, atol=atol))}
    return {"success": all(s["allclose"] for s in stats.values()), "rtol": rtol, "atol": atol,
            "rule": "abs(actual-reference) <= atol + rtol*abs(reference)", "relative_denominator_floor": 1e-12, "outputs": stats}


def evaluate(checkpoint, images, masks, device):
    torch.set_num_threads(4)
    # Strict load replaces ALL parameters/buffers: random construction is never the reference.
    model = build_model(checkpoint["model"], checkpoint["classes"], weights=None, encoder_name=checkpoint["encoder"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    with torch.no_grad():
        logits = model(images.to(device))
        target = masks.to(device)
        loss = model.loss(logits, target if checkpoint["classes"] == 1 else target.argmax(1))
        check_finite(logits, "logits")
        check_finite(loss, "loss")
    synchronize(device)
    return {"logits": logits.cpu().numpy(), "loss": loss.cpu().numpy()}


def environment(device):
    packages = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "numpy", "pillow", "segmentation-models-pytorch")}
    return {"device": str(device), "packages": packages,
            "torch_npu": importlib.metadata.version("torch-npu") if device.type == "npu" else None}


def execute(args, output):
    device = resolve_device(args.device)
    if args.mode == "reference" and device.type != "cpu":
        raise ValueError("Reference must be generated on CPU from the SAME trusted checkpoint")
    checkpoint, images, masks, identity = prepare(args.checkpoint, args.data, args.split, args.indices, args.size)
    reference, manifest = None, None
    if args.mode == "compare":
        manifest = json.loads((Path(args.reference) / "reference.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA or manifest.get("environment", {}).get("device") != "cpu":
            raise ValueError("Unsupported/non-CPU reference")
        require_identity(manifest["identity"], identity)  # fail BEFORE accelerator model allocation
        numerical_path = Path(args.reference) / "numerics.npz"
        if digest(numerical_path) != manifest["numerics_sha256"]:
            raise ValueError("Reference numerics hash mismatch")
        with np.load(numerical_path, allow_pickle=False) as values:
            reference = {name: values[name] for name in values.files}
        expected_shape = (len(args.indices), checkpoint["classes"], args.size, args.size)
        if set(reference) != {"logits", "loss"} or reference["logits"].shape != expected_shape or reference["loss"].shape != ():
            raise ValueError("Reference output keys/shapes do not match model and batch")
    actual = evaluate(checkpoint, images, masks, device)
    env = environment(device)
    if args.mode == "reference":
        numerical_path = output / "numerics.npz"
        np.savez_compressed(numerical_path, **actual)
        manifest = {"schema": SCHEMA, "identity": identity, "environment": env,
                    "numerics_sha256": digest(numerical_path), "success": True,
                    "privacy": "logits/loss only; no image/target arrays; keep local and access-controlled"}
        write_new(output / "reference.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        return {"success": True, "mode": "reference", "environment": env, "loss": float(actual["loss"]),
                "logits_shape": list(actual["logits"].shape), "reference": str(output)}
    result = compare_arrays(reference, actual, args.rtol, args.atol)
    result.update(mode="compare", identity_verified=True, environment=env, reference_environment=manifest["environment"],
                  checkpoint_sha256=identity["checkpoint_sha256"], identity=identity,
                  limitation="Inference parity only; backward/optimizer parity and clinical accuracy NOT established")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "compare"))
    parser.add_argument("--checkpoint", required=True, help="trusted local smoke model.pt only")
    parser.add_argument("--data", default="archive")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference", help="reference directory required for compare")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--output", required=True, help="new output directory; never overwrite")
    args = parser.parse_args()
    if args.size < 32 or args.size % 32 or len(args.indices) > 8:
        parser.error("size>=32 divisible by32; at most8 sample indices")
    if args.mode == "compare" and not args.reference:
        parser.error("compare requires --reference")
    if not all(math.isfinite(v) and v >= 0 for v in (args.rtol, args.atol)):
        parser.error("finite nonnegative tolerances required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        report = execute(args, output)
    except Exception as exc:
        report = {"success": False, "mode": args.mode, "error": str(exc), "error_type": type(exc).__name__, "rtol": args.rtol, "atol": args.atol}
    write_new(output / "summary.json", json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "identity"}, indent=2))
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
