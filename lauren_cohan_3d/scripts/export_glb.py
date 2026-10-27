#!/usr/bin/env python3
"""Export the reconstructed textured head as a self-contained glTF binary."""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MESH = ROOT / "models" / "head_mesh.npz"
DEFAULT_ATLAS = ROOT / "models" / "face_atlas.jpg"
DEFAULT_OUTPUT = ROOT / "output" / "lauren_cohan.glb"


def colored_material(color: tuple[int, int, int, int]) -> trimesh.visual.material.PBRMaterial:
    return trimesh.visual.material.PBRMaterial(
        baseColorFactor=color,
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )


def ellipsoid(extents: tuple[float, float, float], subdivisions: int) -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    mesh.apply_scale(np.asarray(extents) / 2)
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (args.mesh, args.atlas):
        if not path.is_file():
            raise FileNotFoundError(f"Required reconstruction asset not found: {path}")

    with np.load(args.mesh) as data:
        vertices = data["vertices"]
        faces = data["faces"]
        uv = data["uv"]
    with Image.open(args.atlas) as opened:
        atlas = opened.convert("RGB")

    face_material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=atlas,
        metallicFactor=0.0,
        roughnessFactor=0.86,
    )
    head = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=face_material),
        process=True,
    )

    scene = trimesh.Scene()
    scene.add_geometry(head, node_name="textured_head", geom_name="textured_head")
    skin = colored_material((170, 125, 104, 255))
    neck = ellipsoid(extents=(0.42, 0.66, 0.38), subdivisions=3)
    neck.apply_translation((0, -1.05, -0.02))
    neck.visual = trimesh.visual.TextureVisuals(material=skin)
    scene.add_geometry(neck, node_name="neck", geom_name="neck")

    for side, label in ((-1, "left_ear"), (1, "right_ear")):
        ear = ellipsoid(extents=(0.18, 0.30, 0.15), subdivisions=2)
        ear.apply_translation((side * 0.68, -0.02, 0.07))
        ear.visual = trimesh.visual.TextureVisuals(material=skin)
        scene.add_geometry(ear, node_name=label, geom_name=label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(args.output, file_type="glb")
    size = args.output.stat().st_size
    print(f"Exported {args.output} ({size:,} bytes)")
    if size > 2_000_000:
        raise RuntimeError(
            f"GLB exceeds the 2,000,000-byte assignment limit: {size:,} bytes"
        )


if __name__ == "__main__":
    main()
