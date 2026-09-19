"""Strictly reload a trusted local Lightning FCN checkpoint on real validation data."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from src.datasets import BinarySegmentationDataset
from src.models.fcn import FcnSegmentationNet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="Only load a checkpoint created locally/trusted: Lightning pickle is not an untrusted format")
    p.add_argument("--data", default="archive/val.json")
    p.add_argument("--output", default="artifacts/lightning-reload.json")
    args = p.parse_args()
    torch.set_num_threads(4)
    path = Path(args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = FcnSegmentationNet.load_from_checkpoint(str(path), map_location="cpu", weights=None, strict=True).eval()
    if model.num_classes != 1:
        raise ValueError("This check is for the binary FCN Lightning smoke checkpoint")
    dataset = BinarySegmentationDataset(args.data, transform=T.Compose([T.Resize((64,64)), T.ToTensor()]))
    images, masks, _ = next(iter(DataLoader(Subset(dataset, [0,1]), batch_size=2)))
    with torch.no_grad():
        logits = model(images)
        loss = model.loss(logits, masks)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)
    report = {"global_step": checkpoint["global_step"], "epoch": checkpoint["epoch"],
              "optimizer_states": len(checkpoint["optimizer_states"]), "lr_schedulers": len(checkpoint["lr_schedulers"]),
              "strict_reload": True, "validation_loss": loss.item(), "logits_finite": True,
              "shape": list(logits.shape), "checkpoint_bytes": path.stat().st_size}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
