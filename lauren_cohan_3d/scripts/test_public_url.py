#!/usr/bin/env python3
"""Test that a public HTTPS URL serves a parseable GLB with CORS enabled."""

import argparse
import json
import urllib.error
import urllib.request

from validate_glb import parse_glb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Actual public HTTPS URL of the GLB")
    args = parser.parse_args()
    if not args.url.lower().startswith("https://"):
        raise ValueError("The hosting URL must use HTTPS")

    request = urllib.request.Request(
        args.url,
        headers={"Origin": "https://example.invalid", "User-Agent": "GLB-check/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            cors = response.headers.get("Access-Control-Allow-Origin")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Hosting returned HTTP {error.code}: {error.reason}") from error

    if status != 200:
        raise RuntimeError(f"Expected HTTP 200, got {status}")
    if cors not in ("*", "https://example.invalid"):
        raise RuntimeError(f"CORS is not enabled for browser fetches (header: {cors!r})")
    if content[:4] != b"glTF":
        raise RuntimeError("The URL did not return GLB binary data")

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="glb-url-check-") as directory:
        local_copy = Path(directory) / "download.glb"
        local_copy.write_bytes(content)
        document, chunks, _ = parse_glb(local_copy)

    print(
        json.dumps(
            {
                "url": args.url,
                "https": True,
                "http_status": status,
                "content_type": content_type,
                "cors_allow_origin": cors,
                "downloaded_bytes": len(content),
                "glb_version": document["asset"]["version"],
                "chunk_count": len(chunks),
                "valid_glb": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
