"""Validate the Phase 1 dataset plan. This module does not score a model."""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from pathlib import Path

ANGLES = ("straight", "tilt", "occluded")
BRAND_GROUPS = ("international", "pakistan_origin", "exploratory")
ORIGINS = ("international", "local", "exploratory")
COMPARISON_ROLES = ("primary", "exploratory")
SCRIPTS = ("latin", "urdu", "roman_urdu", "mixed")
GROUP_FOR_ORIGIN = {
    "international": "international",
    "local": "pakistan_origin",
    "exploratory": "exploratory",
}
# One rule per group. A product line cannot satisfy both.
GROUP_RULE = {
    "pakistan_origin": "pakistan_market_line",
    "international": "sold_under_same_name_outside_pakistan",
    "exploratory": "excluded_from_primary",
}

ANNOTATION_COLUMNS = (
    "image_path",
    "product_id",
    "category",
    "variant",
    "brand_true",
    "brand_group",
    "origin",
    "script_present",
    "text_true_latin",
    "text_true_urdu",
    "angle",
    "occlusion_note",
    "brand_visible",
    "ambiguous_text",
    "notes",
    "review_status",
)
MANIFEST_COLUMNS = (
    "product_id",
    "category",
    "brand",
    "product_line",
    "variant",
    "package_type",
    "brand_group",
    "origin",
    "membership_basis",
    "group_rule",
    "availability_status",
    "shelf_status",
    "selection_basis",
    "source_note",
    "script_note",
    "comparison_role",
    "limitation",
    "photo_location",
)
SEEN_COLUMNS = (
    "tested_product",
    "filename",
    "on_manifest",
    "manifest_product_id",
    "used_to_choose_manifest",
    "overlap_note",
)
# Visible text is allowed to differ by photograph. Brand identity is not.
SHARED_PRODUCT_FIELDS = ("category", "variant", "brand_true", "brand_group", "origin", "script_present")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            if row is None:
                continue
            if all(not (value or "").strip() for value in row.values()):
                continue
            rows.append({key: (value or "").strip() for key, value in row.items()})
    return header, rows


def _require_header(errors: list[str], label: str, header: list[str], expected: tuple[str, ...]) -> None:
    if header != list(expected):
        errors.append(f"{label} header must be {', '.join(expected)}. Found {header or 'none'}.")


def validate_dataset(root: Path, *, require_complete: bool = False) -> dict[str, object]:
    """Check the planning files and any collected images under data/.

    A header-only annotation file and an empty data/raw tree is valid. That
    state means collection has not started. Phase 0 uploads are not evaluation
    images. ``require_complete`` turns every missing photograph into an error.
    """
    root = Path(root)
    data = root / "data"
    errors: list[str] = []
    annotation_gaps: list[str] = []
    manifest_header, manifest = _read_csv(data / "product_manifest.csv")
    annotation_header, annotations = _read_csv(data / "annotations.csv")
    seen_header, seen_rows = _read_csv(data / "phase0_seen_products.csv")
    _require_header(errors, "product_manifest.csv", manifest_header, MANIFEST_COLUMNS)
    _require_header(errors, "annotations.csv", annotation_header, ANNOTATION_COLUMNS)
    _require_header(errors, "phase0_seen_products.csv", seen_header, SEEN_COLUMNS)

    _validate_manifest(manifest, errors)
    _validate_collection_config(root, errors)
    _validate_brand_aliases(root, manifest, errors)
    _validate_seen_products(seen_rows, errors)
    image_files = _image_files(data / "raw")
    _validate_raw_directories(data / "raw", manifest, errors)
    _reject_phase0_paths(annotations, image_files, errors)
    _validate_annotations(root, annotations, image_files, manifest, errors, annotation_gaps)
    _validate_image_files(root, image_files, errors)
    waiting, partial = _photograph_gaps(root, manifest, image_files, errors, require_complete)

    if errors:
        status = "invalid"
    elif annotation_gaps:
        status = "annotations_incomplete"
    elif image_files or annotations:
        status = "collected"
    else:
        status = "collection_not_started"
    expected = [row.get("product_id", "") for row in manifest if row.get("product_id")]
    return {
        "ok": not errors and not annotation_gaps,
        "status": status,
        "errors": errors,
        "annotation_gaps": annotation_gaps,
        "waiting": waiting,
        "partial": partial,
        "present_count": len(image_files),
        "expected_count": len(expected) * len(ANGLES),
    }


def _validate_collection_config(root: Path, errors: list[str]) -> None:
    path = root / "configs" / "collection_config.yaml"
    if not path.is_file():
        errors.append("configs/collection_config.yaml is missing.")
        return
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    photo = loaded.get("photography") if isinstance(loaded, dict) else None
    comparison = loaded.get("comparison") if isinstance(loaded, dict) else None
    if not isinstance(photo, dict) or not isinstance(comparison, dict):
        errors.append("collection_config.yaml needs photography and comparison sections.")
        return
    if photo.get("location_may_differ_by_category") is not True:
        errors.append("Photography locations may differ by category and must be configured as such.")
    shared = photo.get("pair_must_share")
    if shared != ["lighting", "background", "phone", "camera"]:
        errors.append("Each pair must share lighting, background, phone, and camera.")
    if list(photo.get("angles") or []) != list(ANGLES):
        errors.append("The photograph angles must stay straight, tilt, and occluded.")
    if comparison.get("exploratory_excluded_from_primary") is not True:
        errors.append("Exploratory products must stay out of the primary comparison.")


def _validate_manifest(rows: list[dict[str, str]], errors: list[str]) -> None:
    seen: set[str] = set()
    by_category: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        product_id = row.get("product_id", "")
        category = row.get("category", "")
        group = row.get("brand_group", "")
        origin = row.get("origin", "")
        role = row.get("comparison_role", "")
        if not product_id or not category or not row.get("brand", ""):
            errors.append(f"Manifest row is missing product_id, category, or brand: {row}")
            continue
        if product_id in seen:
            errors.append(f"Duplicate manifest product_id: {product_id}")
        seen.add(product_id)
        if group not in BRAND_GROUPS:
            errors.append(f"{product_id} brand_group must be international, pakistan_origin, or exploratory.")
        if origin not in ORIGINS:
            errors.append(f"{product_id} origin must be international, local, or exploratory.")
        if role not in COMPARISON_ROLES:
            errors.append(f"{product_id} comparison_role must be primary or exploratory.")
        if group in BRAND_GROUPS and origin in ORIGINS and GROUP_FOR_ORIGIN[origin] != group:
            errors.append(
                f"{product_id} has brand_group {group} and origin {origin}. "
                "Those labels must name the same group."
            )
        if role == "primary" and origin == "exploratory":
            errors.append(f"{product_id} is marked primary and exploratory.")
        if role == "exploratory" and origin != "exploratory":
            errors.append(f"{product_id} is exploratory and must not carry an international or local label.")
        rule = row.get("group_rule", "")
        if group in GROUP_RULE and rule != GROUP_RULE[group]:
            errors.append(
                f"{product_id} group_rule must be {GROUP_RULE.get(group, 'a single rule')}. "
                "A product line is in one group only."
            )
        if origin == "exploratory":
            if product_id.endswith("_intl") or product_id.endswith("_local"):
                errors.append(f"{product_id} must not use an international or local id suffix.")
        else:
            expected_suffix = "_intl" if origin == "international" else "_local"
            if origin in ORIGINS and not product_id.endswith(expected_suffix):
                errors.append(f"{product_id} must end with {expected_suffix}.")
        if not product_id.startswith(f"{category}_"):
            errors.append(f"{product_id} must start with the category name {category}_.")
        by_category.setdefault(category, []).append(row)
    for category, category_rows in sorted(by_category.items()):
        roles = {row.get("comparison_role", "") for row in category_rows}
        groups = [row.get("brand_group", "") for row in category_rows]
        limitations = {row.get("limitation", "") for row in category_rows}
        locations = {row.get("photo_location", "") for row in category_rows if row.get("photo_location", "")}
        if len(limitations) != 1:
            errors.append(f"Category {category} must record the same limitation on both products.")
        if len(locations) > 1:
            errors.append(
                f"Category {category} uses more than one photo location. "
                "A pair must share one background and one light."
            )
        if roles == {"exploratory"}:
            if sorted(groups) != ["exploratory", "exploratory"]:
                errors.append(f"Exploratory category {category} must contain two exploratory products.")
            if not category_rows[0].get("limitation", ""):
                errors.append(f"Exploratory category {category} must record why it is outside the primary comparison.")
        elif roles == {"primary"}:
            if sorted(groups) != ["international", "pakistan_origin"]:
                errors.append(
                    f"Category {category} must have one international brand and one pakistan_origin brand."
                )
        else:
            errors.append(f"Category {category} mixes primary and exploratory products.")


def _alias_key(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = folded.replace("'", "").replace("’", "").replace(".", "")
    folded = re.sub(r"[\s\-]+", " ", folded)
    return folded.strip()


def _validate_brand_aliases(root: Path, manifest: list[dict[str, str]], errors: list[str]) -> None:
    """Accepted brand strings are fixed before inference and unique across products."""
    path = root / "configs" / "brand_aliases.yaml"
    if not path.is_file():
        errors.append("configs/brand_aliases.yaml is missing. Aliases must be fixed before inference.")
        return
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    products = loaded.get("products") if isinstance(loaded, dict) else None
    if not isinstance(products, dict):
        errors.append("configs/brand_aliases.yaml must contain a products mapping.")
        return
    manifest_ids = [row.get("product_id", "") for row in manifest if row.get("product_id")]
    if sorted(products) != sorted(manifest_ids):
        errors.append("Brand aliases must list each manifest product once and no other product.")
    owners: dict[str, str] = {}
    by_id = {row.get("product_id", ""): row for row in manifest}
    for product_id, entry in products.items():
        if not isinstance(entry, dict):
            errors.append(f"{product_id} alias entry must be a mapping.")
            continue
        aliases = entry.get("aliases")
        brand = str(entry.get("brand", "")).strip()
        manifest_brand = by_id.get(product_id, {}).get("brand", "")
        if manifest_brand and brand != manifest_brand:
            errors.append(f"{product_id} alias brand must match the manifest brand {manifest_brand}.")
        if not isinstance(aliases, list) or not aliases:
            errors.append(f"{product_id} needs a non-empty aliases list.")
            continue
        if brand and brand not in aliases:
            errors.append(f"{product_id} aliases must include the canonical brand {brand}.")
        parent_names = entry.get("parent_names") or []
        if not isinstance(parent_names, list):
            errors.append(f"{product_id} parent_names must be a list.")
            parent_names = []
        alias_keys = {_alias_key(alias) for alias in aliases if isinstance(alias, str)}
        for parent_name in parent_names:
            if not isinstance(parent_name, str) or not parent_name.strip():
                errors.append(f"{product_id} has an empty parent name.")
                continue
            if _alias_key(parent_name) in alias_keys:
                errors.append(
                    f"{product_id} parent name {parent_name} is also a product-brand alias."
                )
        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip():
                errors.append(f"{product_id} has an empty alias.")
                continue
            key = _alias_key(alias)
            previous = owners.get(key)
            if previous is not None and previous != product_id:
                errors.append(f"Alias {alias!r} is listed for both {previous} and {product_id}.")
            owners[key] = product_id


def _validate_seen_products(rows: list[dict[str, str]], errors: list[str]) -> None:
    for row in rows:
        if row.get("used_to_choose_manifest", "").lower() == "yes":
            errors.append(
                "A Phase 0 product was used to choose the manifest "
                f"({row.get('tested_product') or row.get('filename')}). "
                "Selection must not follow the smoke-test results."
            )


def _image_files(raw: Path) -> list[Path]:
    if not raw.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.name not in {".gitkeep", ".DS_Store"} and not path.name.startswith("._"):
            files.append(path)
    return files


def _reject_phase0_paths(
    annotations: list[dict[str, str]],
    image_files: list[Path],
    errors: list[str],
) -> None:
    for row in annotations:
        path = row.get("image_path", "").replace("\\", "/")
        if "results/" in path or "phase0" in path.lower():
            errors.append(f"Annotation points at a Phase 0 path: {path}")
    for path in image_files:
        if "phase0" in path.name.lower():
            errors.append(f"Phase 0 file is inside data/raw: {path.name}")


def _manifest_by_id(manifest: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("product_id", ""): row for row in manifest if row.get("product_id")}


def _relative_image(root: Path, path: Path, errors: list[str]) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        errors.append(f"Image is outside the project: {path}")
        return ""


def _validate_raw_directories(raw: Path, manifest: list[dict[str, str]], errors: list[str]) -> None:
    if not raw.is_dir():
        return
    known = set(_manifest_by_id(manifest))
    for path in sorted(raw.iterdir()):
        if path.is_dir() and path.name not in known:
            errors.append(f"Directory data/raw/{path.name} is not a manifest product_id.")


def _photograph_gaps(
    root: Path,
    manifest: list[dict[str, str]],
    image_files: list[Path],
    errors: list[str],
    require_complete: bool,
) -> tuple[list[str], list[str]]:
    present: set[str] = set()
    for path in image_files:
        relative = _relative_image(root, path, errors)
        if relative:
            present.add(relative)
    waiting: list[str] = []
    partial: list[str] = []
    for product_id in _manifest_by_id(manifest):
        expected = [f"data/raw/{product_id}/{angle}.jpg" for angle in ANGLES]
        missing = [path for path in expected if path not in present]
        if not missing:
            continue
        if len(missing) == len(expected):
            waiting.append(product_id)
            if require_complete:
                for path in missing:
                    errors.append(f"Missing photograph: {path}")
        else:
            partial.append(product_id)
            for path in missing:
                errors.append(f"Missing photograph: {path}")
    return waiting, partial


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_image_files(root: Path, image_files: list[Path], errors: list[str]) -> None:
    seen: dict[str, str] = {}
    for path in image_files:
        relative = _relative_image(root, path, errors)
        if not relative:
            continue
        if path.suffix.lower() != ".jpg" or path.stem not in ANGLES:
            continue
        try:
            from PIL import Image, UnidentifiedImageError

            with Image.open(path) as image:
                image_format = image.format
            with Image.open(path) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            errors.append(f"Unreadable image file: {relative} ({exc})")
            continue
        if image_format != "JPEG":
            errors.append(f"{relative} is {image_format or 'an unknown format'}, not a JPEG.")
            continue
        digest = _file_sha256(path)
        previous = seen.get(digest)
        if previous is not None:
            errors.append(f"Duplicate image hash {digest}: {previous} and {relative}")
        else:
            seen[digest] = relative


def _validate_annotations(
    root: Path,
    rows: list[dict[str, str]],
    image_files: list[Path],
    manifest: list[dict[str, str]],
    errors: list[str],
    annotation_gaps: list[str],
) -> None:
    paths: list[str] = []
    by_product: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        image_path = row.get("image_path", "").replace("\\", "/")
        product_id = row.get("product_id", "")
        angle = row.get("angle", "")
        if image_path in paths:
            errors.append(f"Duplicate annotation image_path: {image_path}")
        paths.append(image_path)
        if not image_path.startswith("data/raw/") or ".." in Path(image_path).parts:
            errors.append(f"image_path must stay under data/raw: {image_path}")
            continue
        parts = Path(image_path).parts
        if len(parts) != 4 or parts[0] != "data" or parts[1] != "raw":
            errors.append(f"image_path must be data/raw/<product_id>/<angle>.jpg: {image_path}")
            continue
        path_product, filename = parts[2], parts[3]
        if path_product != product_id:
            errors.append(f"{image_path} is filed under {path_product}, not {product_id}.")
        if angle not in ANGLES:
            errors.append(f"{image_path} angle must be straight, tilt, or occluded.")
        expected_name = f"{angle}.jpg"
        if filename != expected_name:
            errors.append(f"{image_path} filename must be {expected_name}.")
        if not row.get("brand_true", ""):
            errors.append(f"{image_path} brand_true is empty.")
        script = row.get("script_present", "")
        review = row.get("review_status", "")
        if review not in ("", "needs_review", "verified"):
            errors.append(f"{image_path} review_status must be needs_review, verified, or blank.")
        if not script:
            annotation_gaps.append(
                f"{image_path} still needs a human script label and visible-text check."
            )
        elif script not in SCRIPTS:
            errors.append(f"{image_path} script_present must be one of {', '.join(SCRIPTS)}.")
        elif review == "needs_review":
            annotation_gaps.append(
                f"{image_path} is a draft marked needs_review, not ground truth."
            )
        if row.get("brand_visible", "") not in ("", "yes", "no"):
            errors.append(f"{image_path} brand_visible must be yes, no, or blank.")
        if row.get("brand_group") not in BRAND_GROUPS or row.get("origin") not in ORIGINS:
            errors.append(f"{image_path} brand_group or origin is not a known label.")
        elif GROUP_FOR_ORIGIN[row["origin"]] != row["brand_group"]:
            errors.append(f"{image_path} brand_group and origin disagree.")
        planned = _manifest_by_id(manifest).get(product_id)
        if planned is None:
            errors.append(f"{product_id} is not a manifest product_id.")
        else:
            if row.get("category", "") != planned.get("category", ""):
                errors.append(
                    f"{image_path} category is {row.get('category', '')!r}; "
                    f"the manifest says {planned.get('category', '')!r}."
                )
            if row.get("brand_group", "") != planned.get("brand_group", ""):
                errors.append(
                    f"{image_path} brand_group does not match the manifest "
                    f"({planned.get('brand_group', '')})."
                )
            if row.get("origin", "") != planned.get("origin", ""):
                errors.append(
                    f"{image_path} origin does not match the manifest ({planned.get('origin', '')})."
                )
            if row.get("variant", "") != planned.get("variant", ""):
                errors.append(
                    f"{image_path} variant does not match the manifest ({planned.get('variant', '')!r})."
                )
        by_product.setdefault(product_id, []).append(row)

    for product_id, product_rows in sorted(by_product.items()):
        angles = [row.get("angle", "") for row in product_rows]
        if sorted(angles) != sorted(ANGLES):
            errors.append(
                f"{product_id} must have exactly one straight, one tilt, and one occluded row."
            )
        first = product_rows[0]
        for row in product_rows[1:]:
            for field in SHARED_PRODUCT_FIELDS:
                if row.get(field, "") != first.get(field, ""):
                    errors.append(
                        f"{product_id} has inconsistent {field} across its three photographs."
                    )
                    break

    annotated = {row.get("image_path", "").replace("\\", "/") for row in rows}
    present: set[str] = set()
    for path in image_files:
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            errors.append(f"Image is outside the project: {path}")
            continue
        present.add(relative)
        if path.suffix.lower() != ".jpg" or path.stem not in ANGLES:
            errors.append(f"Unexpected evaluation image name: {relative}")
    if rows or image_files:
        for image_path in sorted(path for path in annotated if path and path not in present):
            errors.append(f"Missing image file for annotation: {image_path}")
        for image_path in sorted(present - annotated):
            errors.append(f"Image has no annotation row: {image_path}")


def render_validation_report(result: dict[str, object]) -> str:
    """Readable collection report. This is not a model score."""
    errors = list(result.get("errors") or [])
    waiting = list(result.get("waiting") or [])
    partial = list(result.get("partial") or [])
    lines = [
        "# Dataset validation",
        "",
        f"Status: {result.get('status')}",
        f"OK: {'yes' if result.get('ok') else 'no'}",
        f"Photographs present: {result.get('present_count', 0)}",
        f"Photographs expected: {result.get('expected_count', 0)}",
        "",
        "An empty photograph set is collection not started. It is not a completed dataset.",
        "Exploratory products are photographed and left out of the primary comparison.",
        "Blank text fields are unverified. They are not model output and they are not confirmed ground truth.",
        "straight, tilt, and occluded are phone capture conditions, not a controlled 3D viewpoint experiment.",
        "",
        "## Errors",
        "",
    ]
    if errors:
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.append("None.")
    gaps = list(result.get("annotation_gaps") or [])
    lines.extend(["", "## Annotation fields still needing a person", ""])
    if gaps:
        lines.append(f"{len(gaps)} image rows are not verified yet.")
        lines.append("Leave text blank until you have read that photograph. Do not copy hidden text from another angle.")
    else:
        lines.append("None.")
    lines.extend(["", "## Partial products", ""])
    if partial:
        lines.extend(f"- {product_id}" for product_id in partial)
    else:
        lines.append("None.")
    lines.extend(["", "## Products with no photographs yet", ""])
    if waiting:
        lines.extend(f"- {product_id}: straight, tilt, occluded" for product_id in waiting)
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)
