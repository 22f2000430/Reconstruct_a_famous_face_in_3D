#!/usr/bin/env python3
"""Build a textured, parametric head surface from the inspected face crop."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROP = ROOT / "reports" / "face_crop.jpg"
DEFAULT_MESH = ROOT / "models" / "head_mesh.npz"
DEFAULT_ATLAS = ROOT / "models" / "face_atlas.jpg"


def create_head() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = 72
    columns = 96
    vertical = np.linspace(-0.985, 0.985, rows)
    angle = np.linspace(-np.pi / 2, 3 * np.pi / 2, columns + 1)
    yy, theta = np.meshgrid(vertical, angle, indexing="ij")

    jaw_taper = 0.84 + 0.16 * np.clip((yy + 0.1) / 0.9, 0, 1)
    width = 0.72 * np.sqrt(np.maximum(0, 1 - yy**2)) * jaw_taper
    depth = 0.48 * np.sqrt(np.maximum(0, 1 - yy**2))
    xx = width * np.sin(theta)
    zz = depth * np.cos(theta)

    front_weight = np.clip(np.cos(theta), 0, 1) ** 8
    nose_bridge = 0.20 * np.exp(-((xx / 0.10) ** 2) - (((yy - 0.02) / 0.34) ** 2))
    nose_tip = 0.14 * np.exp(-((xx / 0.15) ** 2) - (((yy + 0.22) / 0.12) ** 2))
    cheek_projection = 0.045 * np.exp(
        -(((np.abs(xx) - 0.34) / 0.16) ** 2) - (((yy + 0.10) / 0.20) ** 2)
    )
    eye_recess = 0.035 * np.exp(
        -(((np.abs(xx) - 0.22) / 0.13) ** 2) - (((yy - 0.18) / 0.09) ** 2)
    )
    zz += front_weight * (nose_bridge + nose_tip + cheek_projection - eye_recess)

    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    faces = []
    for row in range(rows - 1):
        for column in range(columns):
            first = row * (columns + 1) + column
            second = first + columns + 1
            faces.extend(
                ((first, first + 1, second), (first + 1, second + 1, second))
            )
    bottom_pole = len(vertices)
    top_pole = bottom_pole + 1
    vertices = np.vstack((vertices, (0, -1, 0), (0, 1, 0)))
    top_ring = (rows - 1) * (columns + 1)
    for column in range(columns):
        faces.append((bottom_pole, column + 1, column))
        faces.append((top_pole, top_ring + column, top_ring + column + 1))

    uv = np.empty((rows, columns + 1, 2), dtype=np.float32)
    for column, value in enumerate(angle):
        if -np.pi / 3 <= value <= np.pi / 3:
            u = 1 + 0.5 * (value + np.pi / 3) / (2 * np.pi / 3)
        elif -np.pi / 2 <= value < -np.pi / 3:
            u = 1.75 - 0.75 * (value + np.pi / 2) / (np.pi / 6)
        elif np.pi / 3 < value <= np.pi / 2:
            u = 1.5 + 0.25 * (value - np.pi / 3) / (np.pi / 6)
        else:
            u = 1.75
        uv[:, column, 0] = u
    uv[:, :, 1] = (1 - vertical[:, None]) / 2
    uv = uv.reshape(-1, 2)
    uv = np.vstack((uv, (1.75, 1.0), (1.75, 0.0)))
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.int32), uv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", type=Path, default=DEFAULT_CROP)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    args = parser.parse_args()

    if not args.crop.is_file():
        raise FileNotFoundError(
            f"Face crop not found: {args.crop}. Run scripts/inspect_image.py first."
        )

    with Image.open(args.crop) as opened:
        face = ImageOps.exif_transpose(opened).convert("RGB")
    face.thumbnail((512, 512), Image.Resampling.LANCZOS)
    pixels = np.asarray(face).copy()
    hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
    background_candidates = (
        (hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 145)
    ).astype(np.uint8)
    _, labels = cv2.connectedComponents(background_candidates, connectivity=8)
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    background = np.isin(labels, border_labels) & (background_candidates == 1)
    pixels[background] = (54, 36, 32)
    face = Image.fromarray(pixels)
    back = Image.new("RGB", face.size, (58, 38, 34))
    atlas = Image.new("RGB", (face.width * 2, face.height))
    atlas.paste(face, (0, 0))
    atlas.paste(back, (face.width, 0))

    vertices, faces, uv = create_head()
    args.mesh.parent.mkdir(parents=True, exist_ok=True)
    args.atlas.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.mesh, vertices=vertices, faces=faces, uv=uv)
    atlas.save(args.atlas, quality=88, optimize=True)
    print(
        f"Created parametric head surface: {len(vertices)} vertices, "
        f"{len(faces)} triangles\nTexture atlas: {args.atlas}"
    )
    print(
        "Note: the single photo supplies front texture only. Side and rear "
        "geometry are procedural approximations."
    )


if __name__ == "__main__":
    main()
