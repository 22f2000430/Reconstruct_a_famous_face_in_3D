#!/usr/bin/env python3
"""Validate GLB container structure and report meshes, materials, and textures."""

import argparse
import json
import struct
from io import BytesIO
from pathlib import Path

import trimesh
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLB = ROOT / "output" / "lauren_cohan.glb"
DEFAULT_REPORT = ROOT / "reports" / "validation.json"
KHRONOS_REPORT = ROOT / "reports" / "khronos_validation.json"
JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942


def parse_glb(path: Path) -> tuple[dict, list[tuple[int, int]], bytes]:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError("File is too short to contain a GLB header and JSON chunk")
    magic, version, declared_length = struct.unpack_from("<4sII", raw, 0)
    if magic != b"glTF":
        raise ValueError("File signature is not binary glTF (glTF)")
    if version != 2:
        raise ValueError(f"Expected glTF 2.0 container version 2, got {version}")
    if declared_length != len(raw):
        raise ValueError(
            f"Declared GLB length {declared_length} does not match {len(raw)} bytes"
        )

    chunks = []
    binary_chunk = b""
    offset = 12
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError("Truncated GLB chunk header")
        length, chunk_type = struct.unpack_from("<II", raw, offset)
        start = offset + 8
        end = start + length
        if end > len(raw):
            raise ValueError("GLB chunk extends beyond the declared file length")
        chunks.append((length, chunk_type))
        if offset == 12:
            if chunk_type != JSON_CHUNK:
                raise ValueError("The first GLB chunk must contain JSON")
            document = json.loads(raw[start:end].decode("utf-8").rstrip(" \t\r\n\0"))
        elif chunk_type == BIN_CHUNK:
            if binary_chunk:
                raise ValueError("GLB contains more than one binary chunk")
            binary_chunk = raw[start:end]
        offset = end
    if offset != len(raw):
        raise ValueError("GLB chunk boundaries do not end at the file boundary")
    if not chunks or chunks[0][1] != JSON_CHUNK:
        raise ValueError("GLB JSON chunk is missing")
    if document.get("asset", {}).get("version") != "2.0":
        raise ValueError("The glTF document does not declare asset version 2.0")
    buffers = document.get("buffers", [])
    if buffers and not any(chunk_type == BIN_CHUNK for _, chunk_type in chunks):
        raise ValueError("Buffer data is declared but the binary chunk is missing")
    if any("uri" in buffer for buffer in buffers):
        raise ValueError("GLB contains an external buffer dependency")
    if buffers and (
        len(buffers) != 1
        or buffers[0].get("byteLength", 0) > len(binary_chunk)
        or len(binary_chunk) - buffers[0].get("byteLength", 0) > 3
    ):
        raise ValueError("GLB buffer length does not match the binary chunk")
    for view in document.get("bufferViews", []):
        buffer_index = view.get("buffer", 0)
        if buffer_index >= len(buffers):
            raise ValueError("Buffer view references a missing buffer")
        start = view.get("byteOffset", 0)
        if start < 0 or view.get("byteLength", 0) < 0:
            raise ValueError("Buffer view has a negative offset or length")
        if start + view["byteLength"] > buffers[buffer_index]["byteLength"]:
            raise ValueError("Buffer view extends beyond its declared buffer")
    images = document.get("images", [])
    if any("uri" in image and not image["uri"].startswith("data:") for image in images):
        raise ValueError("GLB contains an external image dependency")
    for image in images:
        if "bufferView" in image and image["bufferView"] >= len(
            document.get("bufferViews", [])
        ):
            raise ValueError("Embedded image references a missing buffer view")
    return document, chunks, binary_chunk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_GLB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"GLB not found: {args.input}")
    document, chunks, binary_chunk = parse_glb(args.input)
    loaded = trimesh.load(args.input, file_type="glb", force="scene")
    geometry = list(loaded.geometry.values())
    mesh_count = sum(isinstance(item, trimesh.Trimesh) for item in geometry)
    if mesh_count == 0:
        raise ValueError("Parsed GLB contains no triangle meshes")

    image_info = []
    for image in document.get("images", []):
        info = {"mime_type": image.get("mimeType")}
        if "bufferView" in image:
            view = document["bufferViews"][image["bufferView"]]
            start = view.get("byteOffset", 0)
            end = start + view["byteLength"]
            with Image.open(BytesIO(binary_chunk[start:end])) as texture:
                info["width"], info["height"] = texture.size
        image_info.append(info)

    khronos_result = None
    if KHRONOS_REPORT.is_file():
        khronos_result = json.loads(KHRONOS_REPORT.read_text(encoding="utf-8"))

    result = {
        "file": str(args.input),
        "size_bytes": args.input.stat().st_size,
        "size_limit_bytes": 2_000_000,
        "within_size_limit": args.input.stat().st_size <= 2_000_000,
        "container": "glTF Binary",
        "container_version": 2,
        "asset_version": document["asset"]["version"],
        "chunk_types": [f"0x{kind:08x}" for _, kind in chunks],
        "mesh_count": len(document.get("meshes", [])),
        "material_count": len(document.get("materials", [])),
        "embedded_image_count": sum(
            "bufferView" in image for image in document.get("images", [])
        ),
        "textures": image_info,
        "external_dependencies": False,
        "parser": "trimesh",
        "parser_mesh_count": mesh_count,
        "khronos_validator": (
            {
                "report": str(KHRONOS_REPORT),
                "valid": khronos_result["valid"],
                "errors": khronos_result["numErrors"],
                "warnings": khronos_result["numWarnings"],
                "infos": khronos_result["numInfos"],
                "hints": khronos_result["numHints"],
            }
            if khronos_result
            else "not run"
        ),
    }
    if not result["within_size_limit"]:
        raise ValueError(
            f"GLB is larger than 2,000,000 bytes: {result['size_bytes']}"
        )
    if result["material_count"] == 0:
        raise ValueError("GLB contains no materials")
    if result["embedded_image_count"] == 0:
        raise ValueError("GLB contains no embedded image texture")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
