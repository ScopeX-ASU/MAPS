"""Raster binary mask to GDSII polygon conversion utilities."""

from pathlib import Path

import cv2
import gdstk
import numpy as np

__all__ = ["mask_to_gds", "mask_dir_to_gds"]


def mask_to_gds(
    img_path: str,
    out_gds: str,
    *,
    layer: int = 1,
    datatype: int = 0,
    pixel_size_um: float = 0.01,
    threshold: int | None = None,
    invert: bool = False,
    rotate_ccw_90: bool = True,
    morph_open: int = 0,
    morph_close: int = 0,
    simplify_eps_px: float = 1.0,
    min_area_px: float = 10.0,
    max_points: int = 8000,
    cell_name: str = "TOP",
    precision: float = 1e-9,
    unit: float = 1e-6,
) -> int:
    """Convert a raster mask image to a GDSII file with hole-preserving contours.

    Args:
        img_path: Input grayscale mask image path.
        out_gds: Output ``.gds`` file path.
        layer: Target GDS layer number.
        datatype: Target GDS datatype number.
        pixel_size_um: Physical size of one pixel in micrometers.
        threshold: Binarization threshold in ``[0, 255]``; ``None`` selects Otsu.
        invert: Invert foreground/background before contour extraction.
        rotate_ccw_90: Rotate image 90 degrees counter-clockwise before extraction.
        morph_open: Elliptical opening kernel size in pixels (``0`` disables).
        morph_close: Elliptical closing kernel size in pixels (``0`` disables).
        simplify_eps_px: Douglas-Peucker contour simplification tolerance in pixels.
        min_area_px: Minimum contour area in pixels retained in the output.
        max_points: Maximum vertex count per contour before adaptive simplification.
        cell_name: Top-level GDS cell name.
        precision: GDS database precision in meters.
        unit: GDS database unit in meters.

    Returns:
        Number of GDS polygons written to ``cell_name``.
    """
    img_path = str(img_path)
    out_gds = str(out_gds)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    if rotate_ccw_90:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if threshold is None:
        _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(img, int(threshold), 255, cv2.THRESH_BINARY)

    if invert:
        bw = 255 - bw

    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        raise ValueError("No contours found. Check threshold/invert settings.")

    hierarchy = hierarchy[0]

    lib = gdstk.Library(unit=unit, precision=precision)
    cell = lib.new_cell(cell_name)

    h, _ = bw.shape

    def to_phys(points_px: np.ndarray) -> np.ndarray:
        pts = points_px.reshape(-1, 2).astype(np.float64)
        x = pts[:, 0] * pixel_size_um
        y = (h - 1 - pts[:, 1]) * pixel_size_um
        return np.stack([x, y], axis=1)

    def approx_contour(cnt, eps_px):
        if eps_px <= 0:
            return cnt
        return cv2.approxPolyDP(cnt, epsilon=eps_px, closed=True)

    def adaptive_simplify(cnt, base_eps):
        eps = float(base_eps)
        c = cnt
        for _ in range(8):
            c2 = approx_contour(c, eps)
            if len(c2) <= max_points:
                return c2
            eps *= 1.5
            c = c2
        return c

    def area_ok(cnt):
        a = abs(cv2.contourArea(cnt))
        return a >= float(min_area_px)

    def points_to_tuples(pts: np.ndarray) -> list[tuple[float, float]]:
        if pts.size == 0:
            return []
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Invalid points shape: {pts.shape}")
        pts = pts.astype(np.float64, copy=False)
        if not np.isfinite(pts).all():
            pts = pts[np.isfinite(pts).all(axis=1)]
        return [(float(x), float(y)) for x, y in pts]

    polys_added = 0
    boolean_precision = float(precision) / float(unit)

    for i, cnt in enumerate(contours):
        parent = hierarchy[i][3]
        if parent != -1:
            continue

        if not area_ok(cnt):
            continue

        outer = adaptive_simplify(cnt, simplify_eps_px)
        outer_xy = to_phys(outer)
        outer_pts = points_to_tuples(outer_xy)
        if len(outer_pts) < 3:
            continue

        holes_xy = []
        child = hierarchy[i][2]
        while child != -1:
            hc = contours[child]
            if area_ok(hc):
                hc_s = adaptive_simplify(hc, simplify_eps_px)
                hole_pts = points_to_tuples(to_phys(hc_s))
                if len(hole_pts) >= 3:
                    holes_xy.append(hole_pts)
            child = hierarchy[child][0]

        if holes_xy:
            result = gdstk.boolean(
                gdstk.Polygon(outer_pts),
                holes_xy,
                "not",
                precision=boolean_precision,
                layer=layer,
                datatype=datatype,
            )
            if result:
                for poly in result:
                    cell.add(poly)
                polys_added += len(result)
        else:
            poly = gdstk.Polygon(outer_pts, layer=layer, datatype=datatype)
            cell.add(poly)
            polys_added += 1

    if polys_added == 0:
        raise ValueError(
            "All polygons filtered out. Try lowering min_area_px or adjusting threshold."
        )

    Path(out_gds).parent.mkdir(parents=True, exist_ok=True)
    lib.write_gds(out_gds)
    return polys_added


def mask_dir_to_gds(
    img_dir: str,
    out_dir: str | None = None,
    *,
    layer: int = 1,
    datatype: int = 0,
    pixel_size_um: float = 0.01,
    threshold: int | None = None,
    invert: bool = False,
    rotate_ccw_90: bool = True,
    morph_open: int = 0,
    morph_close: int = 0,
    simplify_eps_px: float = 1.0,
    min_area_px: float = 10.0,
    max_points: int = 8000,
    cell_name: str = "TOP",
    precision: float = 1e-9,
    unit: float = 1e-6,
) -> dict[str, int]:
    """Batch convert all mask images in ``img_dir`` to same-stem ``.gds`` files."""
    img_dir_path = Path(img_dir)
    out_dir_path = img_dir_path if out_dir is None else Path(out_dir)
    if not img_dir_path.exists():
        raise FileNotFoundError(f"Input directory not found: {img_dir_path}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    img_paths = sorted(
        p for p in img_dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {img_dir_path}")

    out_dir_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, int] = {}
    for img_path in img_paths:
        out_gds = out_dir_path / f"{img_path.stem}.gds"
        n = mask_to_gds(
            img_path=str(img_path),
            out_gds=str(out_gds),
            layer=layer,
            datatype=datatype,
            pixel_size_um=pixel_size_um,
            threshold=threshold,
            invert=invert,
            rotate_ccw_90=rotate_ccw_90,
            morph_open=morph_open,
            morph_close=morph_close,
            simplify_eps_px=simplify_eps_px,
            min_area_px=min_area_px,
            max_points=max_points,
            cell_name=cell_name,
            precision=precision,
            unit=unit,
        )
        results[str(out_gds)] = n
    return results
