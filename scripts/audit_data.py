"""Read-only archive audit; JSON report contains metadata/hashes, never images."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from PIL import Image
from src.datasets.polygons import image_root, image_path


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(root):
    root = Path(root)
    results, identities, manifest = {}, {}, {}
    cache = {}
    for split in ("train", "val", "test", "dataset"):
        path = root / f"{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        manifest[path.name] = sha256(path)
        imgs, anns = data["images"], data["annotations"]
        ids = {i["id"] for i in imgs}
        cats = {c["id"] for c in data["categories"]}
        counts = Counter(a["image_id"] for a in anns)
        issues = defaultdict(list)
        by_id = {i["id"]: i for i in imgs}
        names, hashes, pixels = set(), set(), set()
        hash_ids = defaultdict(list)
        readable_count = 0
        for im in imgs:
            p = image_path(path, im)
            if not p.is_relative_to(image_root(path)):
                issues["unsafe_paths"].append(im["id"])
                continue
            names.add(im["file_name"].casefold())
            if not p.exists():
                issues["missing_images"].append(im["id"])
                continue
            if p not in cache:
                try:
                    with Image.open(p) as image:
                        image.load()
                        rgb = image.convert("RGB")
                        decoded_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                        cache[p] = (sha256(p), decoded_hash, image.size)
                except Exception as e:
                    cache[p] = str(e)
            if isinstance(cache[p], str):
                issues["unreadable_images"].append(im["id"])
                continue
            h, ph, size = cache[p]
            readable_count += 1
            manifest[str(p.relative_to(root.resolve()))] = h
            hashes.add(h)
            hash_ids[h].append(im["id"])
            pixels.add(ph)
            if size != (im["width"], im["height"]):
                issues["dimension_mismatch"].append(im["id"])
        for ann in anns:
            aid = ann["id"]
            if ann["image_id"] not in ids:
                issues["orphan_annotations"].append(aid)
                continue
            if ann["category_id"] not in cats:
                issues["unknown_categories"].append(aid)
            segments = ann.get("segmentation")
            if not isinstance(segments, list) or not segments:
                issues["invalid_polygons"].append(aid)
                continue
            im = by_id[ann["image_id"]]
            for points in segments:
                if not isinstance(points, list) or len(points) < 6 or len(points) % 2 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in points):
                    issues["invalid_polygons"].append(aid)
                    continue
                xy = list(zip(points[::2], points[1::2]))
                area = abs(sum(x * xy[(i+1)%len(xy)][1] - y * xy[(i+1)%len(xy)][0] for i, (x,y) in enumerate(xy))) / 2
                if area == 0:
                    issues["zero_area_polygons"].append(aid)
                if any(x < 0 or y < 0 or x > im["width"] or y > im["height"] for x,y in xy):
                    issues["out_of_bounds_polygons"].append(aid)
        category_sets = defaultdict(set)
        for ann in anns:
            category_sets[ann["image_id"]].add(ann["category_id"])
        results[split] = {
            "images": len(imgs), "annotations": len(anns), "categories": [{"id": c["id"], "name": c["name"]} for c in data["categories"]],
            "annotations_per_category": dict(Counter(a["category_id"] for a in anns)),
            "images_without_annotations": sum(i not in counts for i in ids),
            "images_with_multiple_annotations": sum(c > 1 for c in counts.values()),
            "images_with_mixed_categories": sum(len(c) > 1 for c in category_sets.values()),
            "annotations_with_multiple_polygons": sum(isinstance(a.get("segmentation"), list) and len(a["segmentation"]) > 1 for a in anns),
            "duplicate_image_ids": len(imgs) - len(ids),
            "duplicate_annotation_ids": len(anns) - len({a["id"] for a in anns}),
            "duplicate_file_names": len(imgs) - len(names),
            "duplicate_file_hashes": readable_count - len(hashes),
            "duplicate_decoded_pixels": readable_count - len(pixels),
            "duplicate_file_groups_image_ids": [group for group in hash_ids.values() if len(group) > 1],
            "issues": dict(issues),
        }
        identities[split] = {"ids": ids, "names": names, "sha256": hashes, "decoded_rgb_sha256": pixels}
        results[split]["readable_images"] = readable_count
    overlaps = {}
    for a, b in combinations(("train", "val", "test"), 2):
        overlaps[f"{a}/{b}"] = {k: len(identities[a][k] & identities[b][k]) for k in identities[a]}
    union = set.union(*(identities[s]["ids"] for s in ("train", "val", "test")))
    disk_files = [p for p in image_root(root / "dataset.json").rglob("*") if p.is_file()]
    referenced = {image_path(root / "dataset.json", im) for im in json.loads((root / "dataset.json").read_text(encoding="utf-8"))["images"]}
    return {"image_directory": str(image_root(root / "dataset.json")),
            "image_files_on_disk": len(disk_files),
            "unreferenced_files_on_disk": sum(p.resolve() not in referenced for p in disk_files),
            "splits": results, "cross_split_overlap": overlaps,
            "split_union_equals_dataset_ids": union == identities["dataset"]["ids"],
            "patient_level_isolation": "NOT VERIFIED: no reliable patient ID mapping audited; exact hashes do not exclude same patient/near duplicates",
            "source_sha256": manifest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="archive")
    parser.add_argument("--output", default="artifacts/data-audit.json")
    args = parser.parse_args()
    report = audit(args.data)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "source_sha256"}, indent=2, ensure_ascii=False))
    blocking = {"missing_images", "unsafe_paths", "unreadable_images", "invalid_polygons", "orphan_annotations", "unknown_categories", "dimension_mismatch", "zero_area_polygons"}
    if any(blocking.intersection(s["issues"]) for s in report["splits"].values()):
        raise SystemExit("Audit found blocking issues; inspect report")


if __name__ == "__main__":
    main()
