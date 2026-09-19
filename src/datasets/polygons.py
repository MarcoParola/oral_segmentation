"""COCO polygon loading shared by the original binary/multiclass datasets."""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


def image_root(annotation_path):
    root = Path(annotation_path).resolve().parent / "oral1"
    # Kaggle archive contains oral1/oral1; legacy datasets use a single oral1.
    return root / "oral1" if (root / "oral1").is_dir() else root


def image_path(annotation_path, image):
    root = image_root(annotation_path)
    path = (root / image["file_name"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image filename escapes oral1")
    return path


def polygons(annotation):
    segments = annotation.get("segmentation", [])
    if not isinstance(segments, list):
        raise ValueError("Only COCO polygon lists are supported, not RLE")
    if not segments:
        raise ValueError("Annotation has no polygons")
    for points in segments:
        if not isinstance(points, list) or len(points) < 6 or len(points) % 2:
            raise ValueError("Polygon needs at least three XY pairs")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in points):
            raise ValueError("Polygon has non-finite/non-numeric coordinates")
        yield list(zip(points[::2], points[1::2]))


def paired_transform(image, mask, transform):
    """Apply deterministic geometry together; categorical masks ALWAYS use nearest.

    Legacy Compose(Resize, CenterCrop, ToTensor) remains accepted. Random or
    unknown transforms fail rather than silently applying different geometry.
    """
    ops = transform.transforms if isinstance(transform, T.Compose) else [transform]
    for op in ops:
        if op is None or isinstance(op, T.ToTensor):
            continue
        if isinstance(op, T.Resize):
            image = TF.resize(image, op.size, op.interpolation, op.max_size, antialias=True)
            mask = TF.resize(mask, op.size, InterpolationMode.NEAREST, op.max_size)
        elif isinstance(op, T.CenterCrop):
            image, mask = TF.center_crop(image, op.size), TF.center_crop(mask, op.size)
        else:
            raise ValueError(f"Unsupported unpaired transform {type(op).__name__}; use Resize/CenterCrop/ToTensor")
    return TF.to_tensor(image), torch.from_numpy(np.array(mask, dtype=np.int64))


class PolygonDataset(torch.utils.data.Dataset):
    def __init__(self, annonations, transform=None, n_classes=1):
        self.annonations = str(annonations)  # legacy API spelling
        self.transform = transform
        self.n_classes = n_classes
        with open(annonations, encoding="utf-8") as f:
            self.dataset = json.load(f)
        self.by_image = defaultdict(list)
        self.category_ids = {c["id"] for c in self.dataset["categories"]}
        if n_classes > 1 and self.category_ids != set(range(1, n_classes + 1)):
            raise ValueError("Multiclass category IDs must be contiguous 1..n_classes")
        for ann in self.dataset["annotations"]:
            if ann["category_id"] not in self.category_ids:
                raise ValueError("Annotation references unknown category")
            self.by_image[ann["image_id"]].append(ann)

    def __len__(self):
        return len(self.dataset["images"])

    def __getitem__(self, idx):
        record = self.dataset["images"][idx]
        with Image.open(image_path(self.annonations, record)) as im:
            image = im.convert("RGB")
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        annotations = self.by_image[record["id"]]
        for ann in annotations:
            value = 1 if self.n_classes == 1 else ann["category_id"]
            for polygon in polygons(ann):
                draw.polygon(polygon, fill=value)
        image, mask = paired_transform(image, mask, self.transform)
        if self.n_classes == 1:
            mask = mask.unsqueeze(0).float()
        else:
            mask = torch.nn.functional.one_hot(mask, self.n_classes + 1).permute(2, 0, 1).float()
        categories = {a["category_id"] for a in annotations}
        # Legacy third return: homogeneous image category; 0=empty, -1=mixed.
        category = next(iter(categories)) if len(categories) == 1 else (-1 if categories else 0)
        return image, mask, category
