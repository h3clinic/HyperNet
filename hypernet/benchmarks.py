"""Import independently captured planar-view sequences into pair manifests.

Supported layouts: Oxford affine ``img1.ppm`` / ``H1to2p``, its OpenCV PNG/XML
mirror, and HPatches ``1.ppm`` / ``H_1_2``. This is a scene-correspondence proxy, not an annotated
physical-object recognition benchmark. No network access is performed here.
"""
from __future__ import annotations
from pathlib import Path
import re
import json
import hashlib
import numpy as np
from PIL import Image
from .data import (SCHEMA_VERSION, load_rgb, file_sha256, pixel_sha256,
                   validate_manifest)


_IMAGE_SUFFIXES = {".ppm", ".pgm", ".png", ".jpg", ".jpeg"}
_OXFORD_CONDITIONS = {"bark": "scale_rotation", "boat": "scale_rotation",
                      "bikes": "blur", "trees": "blur", "graf": "viewpoint",
                      "wall": "viewpoint", "leuven": "illumination",
                      "ubc": "jpeg_compression"}


def _discover_sequences(source_dir):
    references = sorted(p for p in source_dir.rglob("*")
                        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
                        and p.stem in {"img1", "1"})
    sequences = []
    seen_folders = set()
    for reference in references:
        folder = reference.parent
        if folder in seen_folders:
            raise ValueError(f"Ambiguous multiple reference files in {folder}")
        seen_folders.add(folder)
        kind = "oxford_affine" if reference.stem == "img1" else "hpatches"
        pattern = re.compile(r"img([1-9]\d*)$" if kind == "oxford_affine" else r"([1-9]\d*)$")
        queries = []
        seen_indices = set()
        for image in sorted(folder.iterdir()):
            found = pattern.fullmatch(image.stem)
            if not image.is_file() or image.suffix.lower() not in _IMAGE_SUFFIXES or found is None:
                continue
            index = int(found.group(1))
            if index < 2:
                continue
            if index in seen_indices:
                raise ValueError(f"Ambiguous duplicate view index {index} in {folder}")
            seen_indices.add(index)
            homography = folder / (f"H1to{index}p" if kind == "oxford_affine" else f"H_1_{index}")
            if not homography.is_file() and homography.with_suffix(".xml").is_file():
                homography = homography.with_suffix(".xml")
            if not homography.is_file():
                raise ValueError(f"Missing ground-truth homography: {homography}")
            if homography.suffix.lower() == ".xml":
                try:
                    import cv2
                except ImportError as exc:
                    raise ImportError("OpenCV XML ground truth needs opencv-python-headless") from exc
                storage = cv2.FileStorage(str(homography), cv2.FILE_STORAGE_READ)
                try:
                    if not storage.isOpened():
                        raise ValueError(f"Cannot read OpenCV homography: {homography}")
                    matrix = storage.getFirstTopLevelNode().mat()
                finally:
                    storage.release()
                if matrix is None:
                    raise ValueError(f"Missing OpenCV matrix: {homography}")
                matrix = np.asarray(matrix, dtype=float)
            else:
                matrix = np.loadtxt(homography, dtype=float)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
                raise ValueError(f"Invalid homography: {homography}")
            queries.append({"index": index, "image": image, "homography_path": homography,
                            "homography": matrix})
        if not queries:
            raise ValueError(f"No query views with homographies in sequence {folder}")
        name = folder.relative_to(source_dir).as_posix()
        if name == ".":
            name = folder.name
        queries.sort(key=lambda q: q["index"])
        condition = _OXFORD_CONDITIONS.get(folder.name.lower(), "view_change")
        if kind == "hpatches":
            condition = "illumination" if folder.name.startswith("i_") else "viewpoint" if folder.name.startswith("v_") else "view_change"
        sequences.append({"name": name, "reference": reference, "queries": queries,
                          "kind": kind, "condition": condition})
    return sequences


def _resize_with_coordinates(array, size):
    height, width = array.shape[:2]
    scale = size / max(height, width)
    new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
    if min(new_width, new_height) < 8:
        raise ValueError("Requested size makes the shorter image edge less than 8 pixels")
    resized = Image.fromarray(array).resize((new_width, new_height), Image.Resampling.LANCZOS)
    sx, sy = new_width / width, new_height / height
    # Pixel-center convention of resampling: x' = sx * (x + .5) - .5.
    transform = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], float)
    return resized, transform


def _footprint_bbox(matrix, reference_shape, query_shape):
    """Clip the projected reference footprint to the query; return None if invalid.

    This is geometric footprint ground truth, not a human-annotated object box.
    """
    rh, rw = reference_shape[:2]
    qh, qw = query_shape[:2]
    corners = np.array([[0, 0, 1], [rw - 1, 0, 1], [rw - 1, rh - 1, 1], [0, rh - 1, 1]], float)
    projected = corners @ matrix.T
    denominators = projected[:, 2]
    if (not np.isfinite(projected).all() or np.any(np.abs(denominators) < 1e-9)
            or not (np.all(denominators > 0) or np.all(denominators < 0))):
        return None
    polygon = projected[:, :2] / denominators[:, None]
    edges = np.roll(polygon, -1, axis=0) - polygon
    following = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return None
    # Sutherland-Hodgman clipping to an axis-aligned image rectangle.
    for axis, bound, lower in [(0, 0., True), (0, float(qw), False),
                                (1, 0., True), (1, float(qh), False)]:
        if len(polygon) < 3:
            return None
        output = []
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            inside_a = a[axis] >= bound if lower else a[axis] <= bound
            inside_b = b[axis] >= bound if lower else b[axis] <= bound
            if inside_a:
                output.append(a)
            if inside_a != inside_b:
                denominator = b[axis] - a[axis]
                if abs(denominator) < 1e-12:
                    return None
                output.append(a + (bound - a[axis]) / denominator * (b - a))
        polygon = np.asarray(output, dtype=float).reshape(-1, 2)
    if len(polygon) < 3:
        return None
    area = abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1)) -
               np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))) / 2
    if not np.isfinite(area) or area < 1:
        return None
    return [float(v) for v in [*polygon.min(axis=0), *polygon.max(axis=0)]]


def build_sequence_dataset(source_dir, out_dir, size=384, seed=17):
    """Import all local sequences; split identities before pair construction.

    At least four sequences are required so each dev/test split contains at
    least two identities. Each positive uses view 1 as its sole reference and
    a separately captured view 2+ as query. Negatives reuse another sequence's
    resized query from the same split, at the same view index where available.
    """
    source_dir, out_dir = Path(source_dir).resolve(), Path(out_dir).resolve()
    if not source_dir.is_dir():
        raise ValueError("source_dir must be an existing sequence directory")
    if size < 32:
        raise ValueError("size must be at least 32")
    if source_dir == out_dir or out_dir.is_relative_to(source_dir):
        raise ValueError("Place imported data outside the source tree")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("Output directory must be empty")
    sequences = _discover_sequences(source_dir)
    if len(sequences) < 4:
        raise ValueError("At least four independent sequences are required (two per split)")
    rng = np.random.default_rng(seed)
    sequences = [sequences[i] for i in rng.permutation(len(sequences))]
    n_dev = max(2, min(len(sequences) - 2, round(len(sequences) * 0.4)))
    raw_owners = {}; pixel_owners = {}
    # Verify all input content before emitting files. Copied views across sequences
    # are rejected rather than assigned misleading independent identity labels.
    for number, sequence in enumerate(sequences):
        sequence["split"] = "dev" if number < n_dev else "test"
        sequence["views"] = []
        for index, path in [(1, sequence["reference"])] + [(q["index"], q["image"]) for q in sequence["queries"]]:
            array = load_rgb(path)
            if min(array.shape[:2]) < 8:
                raise ValueError(f"Image too small: {path}")
            raw_hash, pixels = file_sha256(path), pixel_sha256(array)
            for owners, digest in [(raw_owners, raw_hash), (pixel_owners, pixels)]:
                if digest in owners:
                    raise ValueError(f"Duplicate captured image in sequence input: {path}; earlier {owners[digest]}")
                owners[digest] = str(path.relative_to(source_dir))
            sequence["views"].append({"index": index, "path": path,
                                      "source_sha256": raw_hash, "source_pixel_sha256": pixels})
        reference_hash = sequence["views"][0]["source_sha256"]
        sequence["object_id"] = "sequence_" + hashlib.sha256((sequence["name"] + reference_hash).encode()).hexdigest()[:12]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "references").mkdir()
    (out_dir / "queries").mkdir()
    objects = []
    for sequence in sequences:
        oid = sequence["object_id"]
        for view in sequence["views"]:
            if file_sha256(view["path"]) != view["source_sha256"]:
                raise ValueError(f"Source image changed during import: {view['path']}")
            image, scale = _resize_with_coordinates(load_rgb(view["path"]), size)
            relative = f"references/{oid}.png" if view["index"] == 1 else f"queries/{oid}_{view['index']:02}.png"
            image.save(out_dir / relative)
            view.update({"relative": relative, "sha256": file_sha256(out_dir / relative),
                         "resize_matrix": scale, "shape": (image.height, image.width)})
        objects.append({"object_id": oid, "split": sequence["split"], "sequence": sequence["name"],
                        "dataset_layout": sequence["kind"], "condition": sequence["condition"],
                        "reference": sequence["views"][0]["relative"],
                        "source_id": sequence["name"],
                        "raw_views": [{"index": v["index"], "source_path": v["path"].relative_to(source_dir).as_posix(),
                                       "source_sha256": v["source_sha256"], "source_pixel_sha256": v["source_pixel_sha256"],
                                       "resized_path": v["relative"], "resized_sha256": v["sha256"],
                                       "resize_matrix": v["resize_matrix"].tolist()} for v in sequence["views"]]})
    pairs = []
    for sequence in sequences:
        reference = sequence["views"][0]
        others = [s for s in sequences if s["split"] == sequence["split"] and s is not sequence]
        for view_number, (query, source_h) in enumerate(zip(sequence["views"][1:], sequence["queries"])):
            h = query["resize_matrix"] @ source_h["homography"] @ np.linalg.inv(reference["resize_matrix"])
            if abs(h[2, 2]) > 1e-12:
                h /= h[2, 2]
            common = {"object_id": sequence["object_id"], "reference": reference["relative"],
                      "reference_sha256": reference["sha256"], "source_id": sequence["name"],
                      "reference_source_sha256": reference["source_sha256"], "split": sequence["split"],
                      "condition": sequence["condition"], "view_index": query["index"]}
            positive = {**common, "id": f"{sequence['object_id']}_{query['index']:02}_present",
                        "query": query["relative"], "query_sha256": query["sha256"],
                        "present": True, "query_object_id": sequence["object_id"],
                        "query_source_id": sequence["name"], "query_source_sha256": query["source_sha256"],
                        "H_ref_to_query": h.tolist(),
                        "source_homography_sha256": file_sha256(source_h["homography_path"])}
            bbox = _footprint_bbox(h, reference["shape"], query["shape"])
            if bbox is not None:
                positive["target_bbox"] = bbox
                positive["bbox_annotation"] = "projected_reference_footprint; not a semantic object box"
            pairs.append(positive)
            negative_sequence = others[view_number % len(others)]
            negative_query = next((v for v in negative_sequence["views"][1:] if v["index"] == query["index"]),
                                  negative_sequence["views"][1 + view_number % (len(negative_sequence["views"]) - 1)])
            pairs.append({**common, "id": f"{sequence['object_id']}_{query['index']:02}_absent",
                          "query": negative_query["relative"], "query_sha256": negative_query["sha256"],
                          "query_source_sha256": negative_query["source_sha256"],
                          "present": False, "query_object_id": negative_sequence["object_id"],
                          "query_source_id": negative_sequence["name"],
                          "query_condition": negative_sequence["condition"],
                          "negative_selection": "other sequence in same split; same view index where available"})
    manifest = {"schema_version": SCHEMA_VERSION, "protocol": "independent_views",
                "title": "Oxford / HPatches independent planar-scene correspondence proxy",
                "seed": int(seed), "max_image_edge": int(size), "objects": objects, "pairs": pairs,
                "notes": ["One independently captured reference view per sequence, with separately captured query views.",
                          "Sequences act as source identities; this is not an annotated physical-object recognition benchmark.",
                          "Entire scenes/planes are matched; background matching can be valid here but is not evidence of object understanding.",
                          "Development and test sequences, source hashes and pixel content are disjoint.",
                          "Negatives use other same-split sequences with identical resize processing; they are not verified hard same-category object negatives.",
                          "Per-condition labels describe the reference sequence; negative query_condition is also recorded and may differ.",
                          "Footprint boxes are derived from supplied planar homographies, not human object annotations.",
                          "Resize matrices use pixel-center coordinates; original benchmark H is used as supplied.",
                          "Raw-file and decoded-pixel hashes are retained for every input view."],
                "sources": ["https://www.robots.ox.ac.uk/~vgg/research/affine/",
                            "https://github.com/hpatches/hpatches-dataset",
                            "https://github.com/opencv/opencv_extra/tree/4.x/testdata/cv/detectors_descriptors_evaluation/images_datasets"]}
    manifest["validation"] = validate_manifest(manifest, out_dir)
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
