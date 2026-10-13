"""Local page for correcting and verifying one photograph at a time.

Run from the repository root:

    .venv/bin/python -m src.review

Images are shown after the in-memory EXIF transpose. The JPEG files are not
rewritten. A row becomes verified only when that photograph's checkbox is
checked. This module does not call a model.
"""

from __future__ import annotations

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.annotate import _load, _save, apply_annotation
from src.dataset import ANGLES

ROOT = Path(".").resolve()
HOST = "127.0.0.1"
PORT = 8765


def _rows() -> list[dict[str, str]]:
    return _load(ROOT / "data" / "annotations.csv")


def _safe_image(product_id: str, filename: str) -> Path | None:
    if "/" in product_id or "\\" in product_id or product_id in {"", ".", ".."}:
        return None
    angle = filename.removesuffix(".jpg")
    if filename != f"{angle}.jpg" or angle not in ANGLES:
        return None
    path = (ROOT / "data" / "raw" / product_id / filename).resolve()
    raw = (ROOT / "data" / "raw").resolve()
    if raw not in path.parents or not path.is_file():
        return None
    return path


def _page(rows: list[dict[str, str]]) -> bytes:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["product_id"], []).append(row)
    verified = sum(row.get("review_status") == "verified" for row in rows)
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Verify annotations</title>",
        "<style>body{font-family:sans-serif;max-width:980px;margin:24px} img{max-width:420px;height:auto} textarea{width:100%;min-height:72px} .card{border:1px solid #ccc;padding:12px;margin:16px 0}</style>",
        "</head><body>",
        "<h1>Verify one photograph at a time</h1>",
        f"<p>{verified} of {len(rows)} photographs are verified. A draft stays a draft until its own checkbox is checked.</p>",
    ]
    for product_id, product_rows in grouped.items():
        parts.append(f"<h2>{html.escape(product_id)}</h2>")
        for row in product_rows:
            image_path = row["image_path"]
            parts.append("<div class='card'>")
            parts.append(
                f"<p><strong>{html.escape(row['angle'])}</strong> "
                f"status {html.escape(row.get('review_status') or 'blank')}</p>"
            )
            parts.append(f"<img src='/image/{html.escape(product_id)}/{html.escape(row['angle'])}.jpg' alt=''>")
            parts.append(f"<form method='post' action='/save'>")
            parts.append(f"<input type='hidden' name='image_path' value='{html.escape(image_path)}'>")
            parts.append("<p>Latin</p>")
            parts.append(f"<textarea name='latin'>{html.escape(row.get('text_true_latin') or '')}</textarea>")
            parts.append("<p>Urdu</p>")
            parts.append(f"<textarea name='urdu'>{html.escape(row.get('text_true_urdu') or '')}</textarea>")
            parts.append("<p>Script</p><select name='script'>")
            for script in ("latin", "urdu", "roman_urdu", "mixed"):
                selected = " selected" if row.get("script_present") == script else ""
                parts.append(f"<option value='{script}'{selected}>{script}</option>")
            parts.append("</select>")
            parts.append("<p>Brand visible</p><select name='brand_visible'>")
            for value in ("yes", "no", ""):
                label = value or "blank"
                selected = " selected" if row.get("brand_visible") == value else ""
                parts.append(f"<option value='{html.escape(value)}'{selected}>{label}</option>")
            parts.append("</select>")
            parts.append("<p>Occlusion note</p>")
            parts.append(f"<textarea name='occlusion'>{html.escape(row.get('occlusion_note') or '')}</textarea>")
            parts.append("<p>Uncertain text</p>")
            parts.append(f"<textarea name='ambiguous'>{html.escape(row.get('ambiguous_text') or '')}</textarea>")
            parts.append("<p>Notes</p>")
            parts.append(f"<textarea name='notes'>{html.escape(row.get('notes') or '')}</textarea>")
            parts.append(
                "<p><label><input type='checkbox' name='verify' value='yes'> "
                "I looked at this photograph and these fields are ground truth.</label></p>"
            )
            parts.append("<p><button type='submit'>Save this photograph</button></p></form></div>")
    parts.append("</body></html>")
    return "\n".join(parts).encode("utf-8")


class ReviewHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = _page(_rows())
            self._send(200, "text/html; charset=utf-8", body)
            return
        if parsed.path.startswith("/image/"):
            pieces = parsed.path.split("/")
            if len(pieces) != 4:
                self._send(404, "text/plain", b"not found")
                return
            path = _safe_image(pieces[2], pieces[3])
            if path is None:
                self._send(404, "text/plain", b"not found")
                return
            from src.images import load_rgb_image

            buffer = BytesIO()
            load_rgb_image(path).save(buffer, format="JPEG", quality=85)
            self._send(200, "image/jpeg", buffer.getvalue())
            return
        self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/save":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(length).decode("utf-8"))
        image_path = form.get("image_path", [""])[0]
        rows = _rows()
        if not any(row["image_path"] == image_path for row in rows):
            self._send(400, "text/plain", b"unknown photograph")
            return
        verified = form.get("verify", [""])[0] == "yes"
        apply_annotation(
            rows,
            image_path=image_path,
            latin=form.get("latin", [""])[0],
            urdu=form.get("urdu", [""])[0],
            script=form.get("script", [""])[0],
            brand_visible=form.get("brand_visible", [""])[0],
            occlusion=form.get("occlusion", [""])[0],
            ambiguous=form.get("ambiguous", [""])[0],
            notes=form.get("notes", [""])[0],
            review_status="verified" if verified else "needs_review",
        )
        _save(ROOT / "data" / "annotations.csv", rows)
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), ReviewHandler)
    print(f"Open http://{HOST}:{PORT}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
