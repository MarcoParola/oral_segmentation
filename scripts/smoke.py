"""Real-data FP32 eager smoke using the original model classes, not a toy net.

python -m scripts.smoke --device cuda:0 --model fcn --output artifacts/fcn-cuda
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from src.datasets import BinarySegmentationDataset, MultiClassSegmentationDataset
from src.devices import resolve_device, synchronize
from src.models.factory import build_model


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def check_finite(tensor, name):
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError(f"Non-finite {name}")


def evaluate(model, batch, device, classes):
    model.eval()
    images, masks, _ = batch
    images, masks = images.to(device), masks.to(device)
    with torch.no_grad():
        logits = model(images)
        check_finite(logits, "validation logits")
        loss = model.loss(logits, masks if classes == 1 else masks.argmax(1))
        check_finite(loss, "validation loss")
        if classes == 1:
            prediction = (logits.sigmoid() > 0.5).float()
            target = masks
        else:
            prediction = (logits.argmax(1) > 0).float()
            target = (masks.argmax(1) > 0).float()
        intersection = (prediction * target).sum()
        total = prediction.sum() + target.sum()
        stats = {"loss": loss.item(), "foreground_dice": ((2 * intersection + 1e-5) / (total + 1e-5)).item(),
                 "foreground_fraction": prediction.mean().item(), "logits_shape": list(logits.shape)}
    return logits.detach().cpu(), stats


def run(args, output):
    start = time.perf_counter()
    device = resolve_device(args.device)  # fail BEFORE loading data/model if unavailable
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "npu":
        torch.npu.manual_seed_all(args.seed)
    classes = 1 if args.classes == 1 else args.classes + 1
    transform = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    dataset_class = BinarySegmentationDataset if args.classes == 1 else MultiClassSegmentationDataset
    extra = {} if args.classes == 1 else {"n_classes": args.classes}
    batches, sample_ids, source_hashes = {}, {}, {}
    for split in ("train", "val", "test"):
        path = Path(args.data) / f"{split}.json"
        dataset = dataset_class(path, transform=transform, **extra)
        if len(dataset) < args.batch_size:
            raise ValueError(f"Not enough {split} samples")
        batches[split] = next(iter(DataLoader(Subset(dataset, range(args.batch_size)), batch_size=args.batch_size, num_workers=0)))
        sample_ids[split] = [im["id"] for im in dataset.dataset["images"][:args.batch_size]]
        source_hashes[split] = digest(path)
        check_finite(batches[split][0], f"{split} images")
        check_finite(batches[split][1], f"{split} masks")
    model = build_model(args.model, classes, weights=None, encoder_name=args.encoder).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    # Track the whole input convolution (not BatchNorm running buffers).
    tracked_name, tracked_parameter = next((n,p) for n,p in model.named_parameters() if p.requires_grad and p.ndim == 4)
    before_parameter = tracked_parameter.detach().cpu().clone()
    _, before_eval = evaluate(model, batches["train"], device, classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    images, masks, _ = batches["train"]
    images, masks = images.to(device), masks.to(device)
    losses, grad_norms = [], []
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        check_finite(logits, "training logits")
        loss = model.loss(logits, masks if classes == 1 else masks.argmax(1))
        check_finite(loss, "training loss")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not gradients:
            raise RuntimeError("No gradients")
        for grad in gradients:
            check_finite(grad, "gradient")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, foreach=False)
        check_finite(norm, "gradient norm")
        if norm.item() == 0:
            raise RuntimeError("Zero gradient norm")
        optimizer.step()
        losses.append(loss.item())
        grad_norms.append(norm.item())
        print(json.dumps({"step": step+1, "loss": losses[-1], "gradient_norm_before_clip": grad_norms[-1]}), flush=True)
    delta = (tracked_parameter.detach().cpu() - before_parameter).abs().max().item()
    if delta <= 0:
        raise RuntimeError("Tracked model parameter did not change")
    _, after_eval = evaluate(model, batches["train"], device, classes)
    expected, val_stats = evaluate(model, batches["val"], device, classes)
    _, test_stats = evaluate(model, batches["test"], device, classes)
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    for key, value in state.items():
        check_finite(value, f"checkpoint {key}")
    checkpoint = output / "model.pt"
    torch.save({"state_dict": state, "model": args.model, "classes": classes, "encoder": args.encoder,
                "weights": None, "steps": args.steps, "seed": args.seed}, checkpoint)
    # Release optimizer/model before allocating the second full network (8GB laptop).
    del optimizer, model, state, logits, gradients, images, masks
    reloaded = build_model(args.model, classes, weights=None, encoder_name=args.encoder)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(saved["state_dict"], strict=True)
    del saved
    reloaded.to(device)
    actual, _ = evaluate(reloaded, batches["val"], device, classes)
    max_error = (expected - actual).abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    synchronize(device)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "pytorch-lightning", "numpy", "segmentation-models-pytorch")}
    if device.type == "npu":
        versions["torch-npu"] = importlib.metadata.version("torch-npu")
    return {"success": True, "purpose": "engineering smoke only, not accuracy or paper reproduction", "args": vars(args),
            "python": platform.python_version(), "platform": platform.platform(), "versions": versions,
            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "precision": "FP32 eager", "weights": None, "parameter_count": parameter_count,
            "sample_ids": sample_ids, "split_sha256": source_hashes,
            "train_losses": losses, "gradient_norms": grad_norms, "all_gradients_finite": True,
            "tracked_parameter": tracked_name, "parameter_max_abs_change": delta,
            "train_eval_before": before_eval, "train_eval_after": after_eval,
            "validation": val_stats, "test_inference": test_stats,
            "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
            "reload_max_abs_error": max_error, "reload_assert_close": True,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "elapsed_seconds": time.perf_counter() - start}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model", choices=("fcn", "deeplab", "unet"), default="fcn")
    p.add_argument("--encoder", default="resnet50")
    p.add_argument("--data", default="archive")
    p.add_argument("--classes", type=int, choices=(1, 3), default=1, help="1=binary union; 3=three foreground categories + background")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="artifacts/smoke")
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.size < 32 or args.size % 32 or args.threads < 1:
        p.error("steps>=1, batch-size>=2 (DeepLab BatchNorm), size>=32 divisible by 32, threads>=1 required")
    output = Path(args.output)
    # Never overwrite original weights or another smoke run.
    output.mkdir(parents=True, exist_ok=False)
    try:
        summary = run(args, output)
    except Exception as exc:
        (output / "summary.json").write_text(json.dumps({"success": False, "error": str(exc), "args": vars(args)}, indent=2), encoding="utf-8")
        raise
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
