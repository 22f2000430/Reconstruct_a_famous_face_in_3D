#!/usr/bin/env python3
"""Orient, inspect, and crop the assigned reference photograph."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "input" / "assigned_face.jpg"
DEFAULT_OUTPUT = ROOT / "reports" / "face_crop.jpg"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Reference image not found: {args.input}")

    with Image.open(args.input) as opened:
        image_format = opened.format
        opened.verify()
    with Image.open(args.input) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        original_size = image.size

    pixels = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    detector_path = (
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )
    detector = cv2.CascadeClassifier(str(detector_path))
    if detector.empty():
        raise RuntimeError(f"Could not load OpenCV face detector: {detector_path}")

    detections = detector.detectMultiScale(
        pixels,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(detections) == 0:
        raise RuntimeError(
            "No face was detected. Use a clear frontal reference image; "
            "the crop was not guessed."
        )

    x, y, width, height = max(detections, key=lambda box: box[2] * box[3])
    image_width, image_height = image.size
    horizontal_margin = round(width * 0.10)
    top_margin = round(height * 0.22)
    bottom_margin = round(height * 0.20)
    left = int(max(0, x - horizontal_margin))
    right = int(min(image_width, x + width + horizontal_margin))
    top = int(max(0, y - top_margin))
    bottom = int(min(image_height, y + height + bottom_margin))
    crop = image.crop((left, top, right, bottom))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(args.output, quality=95)
    report = {
        "input": str(args.input),
        "input_dimensions_after_orientation": [int(value) for value in original_size],
        "format": image_format,
        "face_box_xywh": [int(x), int(y), int(width), int(height)],
        "crop_box_ltrb": [left, top, right, bottom],
        "crop_dimensions": [int(value) for value in crop.size],
        "crop_path": str(args.output),
        "detector": "OpenCV Haar frontal-face cascade",
    }
    report_path = ROOT / "reports" / "image_inspection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
