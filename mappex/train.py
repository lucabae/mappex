"""Turn the generated dataset into a YOLO dataset and train a detector."""

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

VAL_RATIO = 0.2
SEED = 42


def collect_classes(data_dir):
    """Class names, discovered from the folder names of ``data/``."""
    folders = sorted(p for p in Path(data_dir).iterdir() if (p / "labels.json").exists())
    if not folders:
        raise FileNotFoundError(f"No class folders with labels.json under {data_dir}")
    return folders, sorted(p.name for p in folders)


def load_samples(folders, class_ids):
    """Pair every image with its YOLO-format annotations, grouped by class."""
    per_class = {}
    for folder in folders:
        with open(folder / "labels.json", encoding="utf-8") as handle:
            payload = json.load(handle)
        bucket = per_class.setdefault(folder.name, [])
        for sample in payload["samples"]:
            image = folder / sample["image"]
            if not image.exists():
                raise FileNotFoundError(f"Missing image: {image}")
            bucket.append(
                {"image": image, "annotations": _annotations(sample, class_ids)}
            )
    return per_class


def _annotations(sample, class_ids):
    """YOLO lines ``class cx cy w h`` from a sample's object labels."""
    lines = []
    for obj in sample.get("labels", []):
        class_id = class_ids.get(obj.get("class"))
        if class_id is None:
            continue
        x, y, w, h = (float(v) for v in obj["bbox_xywh_normalized"])
        x, y = max(0.0, min(1.0, x)), max(0.0, min(1.0, y))
        w, h = min(w, 1.0 - x), min(h, 1.0 - y)
        if w <= 0.0 or h <= 0.0:
            continue
        lines.append(f"{class_id} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
    return lines


def split(per_class, val_ratio=VAL_RATIO, seed=SEED):
    """Stratified train/validation split, so no class is left out of either."""
    rng = random.Random(seed)
    train, val = [], []
    for samples in per_class.values():
        rng.shuffle(samples)
        if len(samples) < 2:
            train.extend(samples)
            continue
        count = min(max(1, round(len(samples) * val_ratio)), len(samples) - 1)
        val.extend(samples[:count])
        train.extend(samples[count:])
    return train, val


def write_dataset(out_dir, train, val, class_ids):
    """Copy images and labels into the YOLO layout and write ``data.yaml``."""
    out_dir = Path(out_dir)
    for split_name, samples in (("train", train), ("val", val)):
        image_dir = out_dir / "images" / split_name
        label_dir = out_dir / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for index, sample in enumerate(samples):
            stem = f"{index:06d}_{sample['image'].stem}"
            shutil.copy2(sample["image"], image_dir / f"{stem}{sample['image'].suffix.lower()}")
            text = "\n".join(sample["annotations"])
            (label_dir / f"{stem}.txt").write_text(text + ("\n" if text else ""))

    config = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for name, index in class_ids.items()},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return yaml_path


def train(yaml_path, base_model="yolo26n.pt", epochs=4, imgsz=640, batch=16, device="mps"):
    """Fine-tune a YOLO checkpoint on the generated dataset."""
    from ultralytics import YOLO

    model = YOLO(base_model)
    return model.train(
        data=str(Path(yaml_path).resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs",
        name="room_detection",
        device=device,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("yolo_dataset"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--base-model", default="yolo26n.pt")
    args = parser.parse_args(argv)

    folders, names = collect_classes(args.data)
    class_ids = {name: index for index, name in enumerate(names)}
    per_class = load_samples(folders, class_ids)
    train_samples, val_samples = split(per_class)
    yaml_path = write_dataset(args.out, train_samples, val_samples, class_ids)

    print(f"{len(train_samples)} train / {len(val_samples)} val images")
    print(f"Dataset written to {yaml_path}")
    train(yaml_path, args.base_model, args.epochs, args.imgsz, args.batch, args.device)
    print("Training finished")


if __name__ == "__main__":
    main()
