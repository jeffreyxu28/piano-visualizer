"""HTTP Range support for serving audio files.

Browsers rely on byte-range requests to seek within an <audio> element
without downloading the whole file first. We implement this manually so
seeking works reliably regardless of the installed Starlette version.
"""

from __future__ import annotations

from pathlib import Path

from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

CHUNK_SIZE = 1024 * 1024  # 1 MB


def _iter_file_range(path: Path, start: int, end: int):
    with open(path, "rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            read_size = min(CHUNK_SIZE, remaining)
            data = f.read(read_size)
            if not data:
                break
            remaining -= len(data)
            yield data


def ranged_file_response(request: Request, path: Path, media_type: str) -> Response:
    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    if range_header is None:
        return StreamingResponse(
            _iter_file_range(path, 0, file_size - 1),
            media_type=media_type,
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            },
        )

    try:
        units, _, range_spec = range_header.partition("=")
        if units.strip() != "bytes":
            raise ValueError("unsupported range unit")
        start_str, _, end_str = range_spec.partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        if start < 0 or start > end:
            raise ValueError("invalid range")
    except ValueError:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    content_length = end - start + 1
    return StreamingResponse(
        _iter_file_range(path, start, end),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
        },
    )
