#!/usr/bin/env python3
"""Render front, profile, and three-quarter views for a lightweight inspection."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLB = ROOT / "output" / "lauren_cohan.glb"
DEFAULT_OUTPUT = ROOT / "reports" / "views"
SIZE = 480
BACKGROUND = np.array((238, 236, 232), dtype=np.uint8)


def _material_color(material: object) -> np.ndarray:
    factor = getattr(material, "baseColorFactor", None)
    if factor is None:
        return np.array((160, 125, 108), dtype=np.uint8)
    values = np.asarray(factor).reshape(-1)[:3]
    if values.max(initial=0) <= 1:
        values = values * 255
    return np.clip(values, 0, 255).astype(np.uint8)


def _render(scene: trimesh.Scene, yaw_degrees: float) -> Image.Image:
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        if isinstance(mesh, trimesh.Trimesh):
            meshes.append(mesh)

    yaw = np.deg2rad(yaw_degrees)
    camera = np.array((np.sin(yaw), 0.0, np.cos(yaw)))
    forward = -camera
    world_up = np.array((0.0, 1.0, 0.0))
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    target = np.array((0.0, -0.10, 0.0))
    background = np.broadcast_to(BACKGROUND, (SIZE, SIZE, 3)).copy()
    depth_buffer = np.full((SIZE, SIZE), -np.inf, dtype=np.float32)

    for mesh in meshes:
        points = mesh.vertices - target
        screen_x = (points @ right) * (SIZE / 2.8) + SIZE / 2
        screen_y = -(points @ up) * (SIZE / 2.8) + SIZE / 2
        depth = points @ camera
        uv = getattr(mesh.visual, "uv", None)
        material = getattr(mesh.visual, "material", None)
        texture = getattr(material, "baseColorTexture", None)
        texture_pixels = np.asarray(texture.convert("RGB")) if texture is not None else None
        fallback_color = _material_color(material) if material is not None else np.array((160, 125, 108))

        for indices in mesh.faces:
            tri_x = screen_x[indices]
            tri_y = screen_y[indices]
            min_x = max(0, int(np.floor(tri_x.min())))
            max_x = min(SIZE - 1, int(np.ceil(tri_x.max())))
            min_y = max(0, int(np.floor(tri_y.min())))
            max_y = min(SIZE - 1, int(np.ceil(tri_y.max())))
            if min_x > max_x or min_y > max_y:
                continue

            x0, x1, x2 = tri_x
            y0, y1, y2 = tri_y
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denominator) < 1e-8:
                continue
            grid_x, grid_y = np.meshgrid(
                np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5,
                np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5,
            )
            weight0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
            weight1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
            weight2 = 1 - weight0 - weight1
            inside = (
                (weight0 >= -1e-5)
                & (weight1 >= -1e-5)
                & (weight2 >= -1e-5)
            )
            tri_depth = (
                weight0 * depth[indices[0]]
                + weight1 * depth[indices[1]]
                + weight2 * depth[indices[2]]
            )
            current_depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
            visible = inside & (tri_depth > current_depth)
            if not np.any(visible):
                continue

            if texture_pixels is not None and uv is not None:
                tri_uv = uv[indices]
                interpolated_u = (
                    weight0 * tri_uv[0, 0]
                    + weight1 * tri_uv[1, 0]
                    + weight2 * tri_uv[2, 0]
                )
                interpolated_v = (
                    weight0 * tri_uv[0, 1]
                    + weight1 * tri_uv[1, 1]
                    + weight2 * tri_uv[2, 1]
                )
                wrapped_u = np.mod(interpolated_u, 1)
                map_x = (wrapped_u * (texture_pixels.shape[1] - 1)).astype(np.float32)
                map_y = np.clip(interpolated_v * (texture_pixels.shape[0] - 1), 0, texture_pixels.shape[0] - 1).astype(np.float32)
                color = cv2.remap(
                    texture_pixels,
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            else:
                color = np.broadcast_to(fallback_color, (*visible.shape, 3))

            normal = np.cross(
                mesh.vertices[indices[1]] - mesh.vertices[indices[0]],
                mesh.vertices[indices[2]] - mesh.vertices[indices[0]],
            )
            normal_length = np.linalg.norm(normal)
            if normal_length > 1e-8:
                normal /= normal_length
                light = 0.28 + 0.72 * max(float(np.dot(normal, camera)), 0.0)
            else:
                light = 0.7
            shaded = np.clip(color.astype(np.float32) * light, 0, 255).astype(np.uint8)

            view = background[min_y : max_y + 1, min_x : max_x + 1]
            view[visible] = shaded[visible]
            current_depth[visible] = tri_depth[visible]

    return Image.fromarray(background)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_GLB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"GLB not found: {args.input}")

    scene = trimesh.load(args.input, file_type="glb", force="scene")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    views = (
        ("front", 0),
        ("left_profile", -90),
        ("right_profile", 90),
        ("left_three_quarter", -45),
        ("right_three_quarter", 45),
    )
    images = []
    for name, yaw in views:
        image = _render(scene, yaw)
        image.save(args.output_dir / f"{name}.png")
        images.append((name.replace("_", " "), image))

    contact_sheet = Image.new("RGB", (len(images) * SIZE, SIZE + 36), "white")
    draw = ImageDraw.Draw(contact_sheet)
    for index, (label, image) in enumerate(images):
        x = index * SIZE
        contact_sheet.paste(image, (x, 36))
        draw.text((x + 12, 12), label, fill=(20, 20, 20))
    contact_sheet.save(args.output_dir.parent / "multi_view_inspection.png")
    print(f"Saved five inspection views in {args.output_dir}")
    print(f"Contact sheet: {args.output_dir.parent / 'multi_view_inspection.png'}")


if __name__ == "__main__":
    main()
