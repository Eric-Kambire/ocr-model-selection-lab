"""Détecteur hybride de document et correcteur de perspective.

Ce moteur est partagé par les laboratoires Gradio et par le code applicatif.
Il ne dépend pas de l'interface : il reçoit des pixels, génère plusieurs
quadrilatères candidats, les classe puis redresse le candidat retenu.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .json_utils import dumps_json

try:
    import fitz  # PyMuPDF
except ImportError:  # PDF support becomes optional for image-only use.
    fitz = None


@dataclass
class DetectorConfig:
    expected_aspect_ratio: float = 1.586  # ISO/IEC 7810 ID-1: 85.60 / 53.98
    min_area_ratio: float = 0.035
    max_area_ratio: float = 0.92
    max_working_side: int = 1800
    edge_tolerance_ratio: float = 0.004
    density_window_ratio: float = 0.035
    max_candidates_per_detector: int = 30
    final_margin_ratio: float = 0.012
    frame_ignore_ratio: float = 0.008


@dataclass
class Candidate:
    quad: np.ndarray
    source: str
    score: float = 0.0
    metrics: dict[str, float] | None = None

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "score": round(float(self.score), 6),
            "quad_working": np.round(self.quad, 2).tolist(),
            "metrics": {
                key: round(float(value), 6)
                for key, value in (self.metrics or {}).items()
            },
        }


def load_input(path: str | Path, pdf_page: int = 0, pdf_dpi: int = 220) -> np.ndarray:
    """Load an image or render one PDF page as BGR.

    Deliberately does NOT use EXIF orientation. Orientation is handled later from
    the detected card geometry, not from metadata.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for PDF input: pip install pymupdf")
        doc = fitz.open(path)
        if not 0 <= pdf_page < len(doc):
            raise IndexError(f"PDF page {pdf_page} is outside 0..{len(doc)-1}")
        page = doc[pdf_page]
        zoom = pdf_dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read input: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"Unable to encode image for {path}")
    encoded.tofile(str(path))


def resize_for_work(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image.copy(), 1.0
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    # After angular sorting, rotate so top-left comes first.
    start = np.argmin(pts[:, 0] + pts[:, 1])
    pts = np.roll(pts, -start, axis=0)
    # Ensure TL, TR, BR, BL rather than TL, BL, BR, TR.
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[1]
    cross_z = float(v1[0] * v2[1] - v1[1] * v2[0])
    if cross_z < 0:
        pts[[1, 3]] = pts[[3, 1]]
    return pts.astype(np.float32)


def polygon_area(quad: np.ndarray) -> float:
    return abs(float(cv2.contourArea(order_quad(quad))))


def quad_from_contour(contour: np.ndarray) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    for epsilon_ratio in (0.015, 0.025, 0.04, 0.06):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_quad(approx.reshape(4, 2))

    # Fallback: rotated rectangle. Useful when a border has small breaks/noise.
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    if min(rect[1]) < 8:
        return None
    return order_quad(box)


def auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper - lower < 30:
        lower = max(0, int(median) - 45)
        upper = min(255, int(median) + 45)
    return cv2.Canny(gray, lower, upper, L2gradient=True)


def detect_dark_frame_bands(
    image: np.ndarray,
) -> tuple[int, int, int, int, np.ndarray]:
    """Détecte uniquement les bandes sombres continues attachées au cadre.

    Une ligne ou colonne est considérée comme une bande de scanner lorsque
    82 % au moins de ses pixels ont une luminance inférieure à 55. La recherche
    part du bord et s'arrête au premier rang non conforme ; un objet sombre
    isolé dans un coin ne suffit donc pas à être supprimé.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    dark = gray < 55
    row_ratio = dark.mean(axis=1)
    col_ratio = dark.mean(axis=0)

    def leading(values: np.ndarray, max_count: int) -> int:
        count = 0
        for value in values[:max_count]:
            if float(value) < 0.82:
                break
            count += 1
        return count

    def trailing(values: np.ndarray, max_count: int) -> int:
        count = 0
        for value in values[::-1][:max_count]:
            if float(value) < 0.82:
                break
            count += 1
        return count

    top = leading(row_ratio, max(1, int(round(height * 0.08))))
    bottom = trailing(row_ratio, max(1, int(round(height * 0.08))))
    left = leading(col_ratio, max(1, int(round(width * 0.08))))
    right = trailing(col_ratio, max(1, int(round(width * 0.08))))

    mask = np.zeros((height, width), dtype=np.uint8)
    if top:
        mask[:top, :] = 255
    if bottom:
        mask[height - bottom :, :] = 255
    if left:
        mask[:, :left] = 255
    if right:
        mask[:, width - right :] = 255
    return top, bottom, left, right, mask


def replace_detected_frame(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Remplace le cadre détecté dans une copie par la couleur de fond voisine."""
    top, bottom, left, right, mask = detect_dark_frame_bands(image)
    cleaned = image.copy()
    height, width = image.shape[:2]

    y0 = min(height - 1, top + 2)
    y1 = max(0, height - bottom - 3)
    x0 = min(width - 1, left + 2)
    x1 = max(0, width - right - 3)
    samples: list[np.ndarray] = []
    if y0 < height:
        samples.append(image[y0 : min(height, y0 + 5), max(0, x0) : min(width, x1 + 1)])
    if y1 >= 0:
        samples.append(image[max(0, y1 - 4) : y1 + 1, max(0, x0) : min(width, x1 + 1)])
    if x0 < width:
        samples.append(image[max(0, y0) : min(height, y1 + 1), x0 : min(width, x0 + 5)])
    if x1 >= 0:
        samples.append(image[max(0, y0) : min(height, y1 + 1), max(0, x1 - 4) : x1 + 1])
    valid = [sample.reshape(-1, 3) for sample in samples if sample.size]
    fill = (
        np.median(np.concatenate(valid, axis=0), axis=0).astype(np.uint8)
        if valid
        else np.array([255, 255, 255], dtype=np.uint8)
    )
    cleaned[mask > 0] = fill
    return cleaned, mask, {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def preprocess(image: np.ndarray, config: DetectorConfig) -> dict[str, Any]:
    analysis_image, frame_artifact_mask, frame_bands = replace_detected_frame(image)
    gray = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.bilateralFilter(clahe, 7, 40, 40)

    sobel_x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient_u8 = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    canny = auto_canny(smooth)
    _, strong_gradient = cv2.threshold(
        gradient_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    edge_union = cv2.bitwise_or(canny, strong_gradient)

    short_side = min(image.shape[:2])
    close_size = max(3, int(round(short_side * 0.008)))
    close_size += 1 - close_size % 2
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
    connected_edges = cv2.morphologyEx(edge_union, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Local edge density and local variance identify a textured document on a
    # relatively uniform support, even when the outer border is incomplete.
    density_window = max(15, int(round(short_side * config.density_window_ratio)))
    density_window += 1 - density_window % 2
    edge_float = (edge_union > 0).astype(np.float32)
    edge_density = cv2.boxFilter(
        edge_float,
        ddepth=-1,
        ksize=(density_window, density_window),
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )

    gray_float = smooth.astype(np.float32) / 255.0
    mean = cv2.boxFilter(gray_float, -1, (density_window, density_window), normalize=True)
    mean_sq = cv2.boxFilter(gray_float * gray_float, -1, (density_window, density_window), normalize=True)
    local_variance = np.maximum(mean_sq - mean * mean, 0.0)
    variance_norm = cv2.normalize(local_variance, None, 0, 1, cv2.NORM_MINMAX)

    # Density map: supports both printed text/photo texture and strong borders.
    texture_score = 0.62 * edge_density + 0.38 * variance_norm
    positive = texture_score[texture_score > 0]
    threshold = float(np.percentile(positive, 58)) if positive.size else 0.05
    threshold = max(0.035, min(threshold, 0.22))
    density_mask = (texture_score >= threshold).astype(np.uint8) * 255

    density_close = max(9, int(round(short_side * 0.035)))
    density_close += 1 - density_close % 2
    density_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (density_close, density_close)
    )
    density_mask = cv2.morphologyEx(density_mask, cv2.MORPH_CLOSE, density_kernel, iterations=2)
    density_mask = cv2.morphologyEx(
        density_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    # Foreground-vs-background map.
    #
    # The background colour is estimated robustly from the outer strips of the
    # image. This detector is especially useful for screenshots/scans where the
    # ID card is a compact object on a large white or nearly uniform support.
    # It does not assume A4 and it does not require an intact outer border.
    height, width = image.shape[:2]
    strip = max(4, int(round(min(height, width) * 0.025)))
    lab = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2LAB).astype(np.float32)
    border_pixels = np.concatenate(
        [
            lab[:strip].reshape(-1, 3),
            lab[-strip:].reshape(-1, 3),
            lab[:, :strip].reshape(-1, 3),
            lab[:, -strip:].reshape(-1, 3),
        ],
        axis=0,
    )
    background_lab = np.median(border_pixels, axis=0)
    colour_distance = np.linalg.norm(lab - background_lab[None, None, :], axis=2)
    border_distance = np.linalg.norm(border_pixels - background_lab[None, :], axis=1)
    # A heterogeneous phone background naturally gives a larger threshold;
    # a clean white scan keeps it low enough to retain text/photo content.
    foreground_threshold = float(np.percentile(border_distance, 75) + 8.0)
    foreground_threshold = float(np.clip(foreground_threshold, 10.0, 32.0))
    foreground_mask = (colour_distance >= foreground_threshold).astype(np.uint8) * 255

    # Keep a separate texture-based map for diagnostics/future fallback. It is
    # intentionally NOT merged here: merging it may connect a distant button,
    # watermark or screenshot frame to the card through sparse edge chains.
    texture_binary = (texture_score >= max(0.055, threshold * 0.82)).astype(np.uint8) * 255

    # Remove the literal image frame, which is frequently a black screenshot
    # outline and should never be interpreted as a document side.
    clear = max(2, int(round(short_side * config.frame_ignore_ratio * 0.45)))
    foreground_mask[:clear, :] = 0
    foreground_mask[-clear:, :] = 0
    foreground_mask[:, :clear] = 0
    foreground_mask[:, -clear:] = 0

    # Join neighbouring text/photo regions inside one document without joining
    # objects separated by a large blank zone (e.g. an unrelated “Copy” button).
    fg_kx = max(7, int(round(short_side * 0.045)))
    fg_ky = max(5, int(round(short_side * 0.018)))
    fg_kx += 1 - fg_kx % 2
    fg_ky += 1 - fg_ky % 2
    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (fg_kx, fg_ky)),
        iterations=2,
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    return {
        "analysis_image": analysis_image,
        "frame_artifact_mask": frame_artifact_mask,
        "frame_bands": frame_bands,
        "gray": gray,
        "smooth": smooth,
        "gradient": gradient_u8,
        "canny": canny,
        "edge_union": edge_union,
        "connected_edges": connected_edges,
        "edge_density": edge_density,
        "local_variance": variance_norm,
        "texture_score": texture_score,
        "density_mask": density_mask,
        "foreground_mask": foreground_mask,
        "texture_foreground_mask": texture_binary,
        "colour_distance": cv2.normalize(
            colour_distance, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8),
    }


def valid_candidate_area(quad: np.ndarray, image_area: float, config: DetectorConfig) -> bool:
    ratio = polygon_area(quad) / image_area
    return config.min_area_ratio <= ratio <= config.max_area_ratio


def candidates_from_binary(
    binary: np.ndarray,
    source: str,
    config: DetectorConfig,
) -> list[Candidate]:
    height, width = binary.shape[:2]
    image_area = float(height * width)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    candidates: list[Candidate] = []

    for contour in contours[:250]:
        if cv2.contourArea(contour) < config.min_area_ratio * image_area * 0.35:
            break
        hull = cv2.convexHull(contour)
        quad = quad_from_contour(hull)
        if quad is None or not valid_candidate_area(quad, image_area, config):
            continue
        candidates.append(Candidate(quad=quad, source=source))
        if len(candidates) >= config.max_candidates_per_detector:
            break
    return candidates


def contour_detector(maps: dict[str, np.ndarray], config: DetectorConfig) -> list[Candidate]:
    results: list[Candidate] = []
    results.extend(candidates_from_binary(maps["connected_edges"], "contours", config))

    # Black-hat catches dark printed/document boundaries on brighter supports.
    gray = maps["smooth"]
    short_side = min(gray.shape)
    kernel_size = max(15, int(short_side * 0.03))
    kernel_size += 1 - kernel_size % 2
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
    )
    _, blackhat_bin = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blackhat_bin = cv2.morphologyEx(
        blackhat_bin,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, max(3, kernel_size // 4))),
        iterations=2,
    )
    maps["blackhat"] = blackhat_bin
    results.extend(candidates_from_binary(blackhat_bin, "blackhat", config))
    return results


def line_detector(maps: dict[str, np.ndarray], config: DetectorConfig) -> list[Candidate]:
    """Line-based detector.

    Instead of demanding four perfectly continuous lines, it detects fragments,
    redraws them into a support mask, closes small gaps and extracts quadrilaterals.
    """
    edges = maps["canny"]
    height, width = edges.shape
    min_len = max(30, int(min(height, width) * 0.12))
    max_gap = max(8, int(min(height, width) * 0.025))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=max(35, int(min(height, width) * 0.045)),
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    line_mask = np.zeros_like(edges)
    if lines is not None:
        segments: list[tuple[float, tuple[int, int, int, int]]] = []
        for item in lines[:, 0]:
            x1, y1, x2, y2 = map(int, item)
            frame_margin = max(2, int(round(min(height, width) * config.frame_ignore_ratio)))
            on_frame = (
                (x1 <= frame_margin and x2 <= frame_margin)
                or (x1 >= width - 1 - frame_margin and x2 >= width - 1 - frame_margin)
                or (y1 <= frame_margin and y2 <= frame_margin)
                or (y1 >= height - 1 - frame_margin and y2 >= height - 1 - frame_margin)
            )
            if on_frame:
                continue
            length = math.hypot(x2 - x1, y2 - y1)
            segments.append((length, (x1, y1, x2, y2)))
        segments.sort(reverse=True, key=lambda value: value[0])
        for _, (x1, y1, x2, y2) in segments[:160]:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=3, lineType=cv2.LINE_AA)

    gap = max(7, int(min(height, width) * 0.018))
    line_mask = cv2.morphologyEx(
        line_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap)),
        iterations=2,
    )
    line_mask = cv2.dilate(
        line_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=1,
    )
    maps["line_mask"] = line_mask
    return candidates_from_binary(line_mask, "lines", config)



def _orientation_distance(angle_a: float, angle_b: float) -> float:
    delta = abs(angle_a - angle_b) % math.pi
    return min(delta, math.pi - delta)


def _line_equation(segment: tuple[float, float, float, float]) -> np.ndarray | None:
    x1, y1, x2, y2 = segment
    a = y1 - y2
    b = x2 - x1
    norm = math.hypot(a, b)
    if norm < 1e-6:
        return None
    a /= norm
    b /= norm
    c = -(a * x1 + b * y1)
    return np.array([a, b, c], dtype=np.float64)


def _intersect_lines(line_a: np.ndarray, line_b: np.ndarray) -> np.ndarray | None:
    matrix = np.array([[line_a[0], line_a[1]], [line_b[0], line_b[1]]], dtype=np.float64)
    determinant = float(np.linalg.det(matrix))
    if abs(determinant) < 1e-5:
        return None
    rhs = -np.array([line_a[2], line_b[2]], dtype=np.float64)
    point = np.linalg.solve(matrix, rhs)
    return point.astype(np.float32)


def line_quad_detector(maps: dict[str, np.ndarray], config: DetectorConfig) -> list[Candidate]:
    """Build quadrilaterals from two families of long line fragments.

    This detector is designed for interrupted borders. It does not require a
    closed contour: two opposite lines from one orientation family and two from
    the approximately perpendicular family are intersected to form a quad.
    """
    gray = maps["smooth"]
    height, width = gray.shape
    short_side = min(height, width)
    min_length = max(35.0, short_side * 0.10)

    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)[0]
    if detected is None:
        return []

    segments: list[dict] = []
    for raw in detected[:, 0]:
        x1, y1, x2, y2 = map(float, raw)
        frame_margin = max(2.0, short_side * config.frame_ignore_ratio)
        on_frame = (
            (x1 <= frame_margin and x2 <= frame_margin)
            or (x1 >= width - 1 - frame_margin and x2 >= width - 1 - frame_margin)
            or (y1 <= frame_margin and y2 <= frame_margin)
            or (y1 >= height - 1 - frame_margin and y2 >= height - 1 - frame_margin)
        )
        if on_frame:
            continue
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_length:
            continue
        angle = math.atan2(y2 - y1, x2 - x1) % math.pi
        segments.append({
            "coords": (x1, y1, x2, y2),
            "length": length,
            "angle": angle,
            "mid": np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32),
        })

    if len(segments) < 4:
        return []

    # Weighted orientation histogram. Printed text often reinforces the same
    # horizontal family as the long edges, which is useful here.
    bins = 180
    histogram = np.zeros(bins, dtype=np.float64)
    for segment in segments:
        index = int(round(segment["angle"] / math.pi * bins)) % bins
        histogram[index] += segment["length"]
    dominant_index = int(np.argmax(histogram))
    theta_a = dominant_index / bins * math.pi
    theta_b = (theta_a + math.pi / 2.0) % math.pi
    tolerance = math.radians(24.0)

    family_a = [s for s in segments if _orientation_distance(s["angle"], theta_a) <= tolerance]
    family_b = [s for s in segments if _orientation_distance(s["angle"], theta_b) <= tolerance]
    if len(family_a) < 2 or len(family_b) < 2:
        return []

    def boundary_lines(family: list[dict], family_theta: float) -> tuple[list[dict], list[dict]]:
        # Project segment midpoints on the family normal, then retain long lines
        # near both spatial extremes. Internal text lines are rarely extreme.
        normal = np.array([-math.sin(family_theta), math.cos(family_theta)], dtype=np.float32)
        for segment in family:
            segment["rho"] = float(np.dot(segment["mid"], normal))
        family = sorted(family, key=lambda item: item["rho"])
        take = min(10, max(3, len(family) // 3))
        low = sorted(family[:take], key=lambda item: item["length"], reverse=True)[:6]
        high = sorted(family[-take:], key=lambda item: item["length"], reverse=True)[:6]
        return low, high

    a_low, a_high = boundary_lines(family_a, theta_a)
    b_low, b_high = boundary_lines(family_b, theta_b)

    image_area = float(height * width)
    results: list[Candidate] = []
    seen = 0
    for first_a in a_low:
        for second_a in a_high:
            if abs(first_a["rho"] - second_a["rho"]) < short_side * 0.08:
                continue
            line_a1 = _line_equation(first_a["coords"])
            line_a2 = _line_equation(second_a["coords"])
            if line_a1 is None or line_a2 is None:
                continue
            for first_b in b_low:
                for second_b in b_high:
                    if abs(first_b["rho"] - second_b["rho"]) < short_side * 0.08:
                        continue
                    line_b1 = _line_equation(first_b["coords"])
                    line_b2 = _line_equation(second_b["coords"])
                    if line_b1 is None or line_b2 is None:
                        continue
                    points = [
                        _intersect_lines(line_a1, line_b1),
                        _intersect_lines(line_a1, line_b2),
                        _intersect_lines(line_a2, line_b2),
                        _intersect_lines(line_a2, line_b1),
                    ]
                    if any(point is None for point in points):
                        continue
                    quad = order_quad(np.array(points, dtype=np.float32))
                    # Allow intersections just outside the frame, but reject
                    # numerically unstable, remote intersections.
                    if (
                        np.any(quad[:, 0] < -0.08 * width)
                        or np.any(quad[:, 0] > 1.08 * width)
                        or np.any(quad[:, 1] < -0.08 * height)
                        or np.any(quad[:, 1] > 1.08 * height)
                    ):
                        continue
                    if not cv2.isContourConvex(np.round(quad).astype(np.int32)):
                        continue
                    if not valid_candidate_area(quad, image_area, config):
                        continue
                    results.append(Candidate(quad=quad, source="line_quad"))
                    seen += 1
                    if seen >= 220:
                        break
                if seen >= 220:
                    break
            if seen >= 220:
                break
        if seen >= 220:
            break

    # Cheap preselection before the full score: boundary continuity + area prior.
    tolerance_px = max(2, int(round(short_side * config.edge_tolerance_ratio)))
    for candidate in results:
        continuity = edge_continuity_score(candidate.quad, maps["edge_union"], tolerance_px)
        ratio = polygon_area(candidate.quad) / image_area
        candidate.score = 0.75 * continuity + 0.25 * area_score(ratio)
    results.sort(key=lambda item: item.score, reverse=True)
    return results[: config.max_candidates_per_detector]

def density_detector(maps: dict[str, np.ndarray], config: DetectorConfig) -> list[Candidate]:
    return candidates_from_binary(maps["density_mask"], "density", config)


def foreground_detector(maps: dict[str, np.ndarray], config: DetectorConfig) -> list[Candidate]:
    """Generate compact document candidates from foreground components.

    Unlike the density detector, this uses only external connected components
    and deliberately avoids a large closing kernel. It is therefore resistant
    to two unrelated objects separated by blank space.
    """
    binary = maps["foreground_mask"]
    height, width = binary.shape
    image_area = float(height * width)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    candidates: list[Candidate] = []

    for contour in contours[:120]:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < config.min_area_ratio * image_area * 0.18:
            continue

        hull = cv2.convexHull(contour)
        rect = cv2.minAreaRect(hull)
        if min(rect[1]) < 8:
            continue
        quad = order_quad(cv2.boxPoints(rect))
        if valid_candidate_area(quad, image_area, config):
            candidates.append(Candidate(quad=quad, source="foreground"))

        # Axis-aligned alternative is valuable for flatbed scans/screenshots.
        x, y, w, h = cv2.boundingRect(hull)
        box = np.array(
            [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
            dtype=np.float32,
        )
        if valid_candidate_area(box, image_area, config):
            candidates.append(Candidate(quad=box, source="foreground_box"))

        if len(candidates) >= 2 * config.max_candidates_per_detector:
            break
    return candidates


def edge_continuity_score(
    quad: np.ndarray,
    edge_map: np.ndarray,
    tolerance_px: int,
) -> float:
    """Fraction of perimeter samples supported by a nearby edge pixel.

    Distance transform makes the score tolerant to a broken/shifted border.
    A sample counts as supported when an edge exists within tolerance_px.
    """
    distance = cv2.distanceTransform((edge_map == 0).astype(np.uint8), cv2.DIST_L2, 3)
    quad = order_quad(quad)
    supported = 0
    total = 0
    for index in range(4):
        start = quad[index]
        end = quad[(index + 1) % 4]
        length = max(2, int(np.linalg.norm(end - start)))
        samples = max(30, min(350, length))
        for t in np.linspace(0.0, 1.0, samples):
            point = start * (1.0 - t) + end * t
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
            if 0 <= x < edge_map.shape[1] and 0 <= y < edge_map.shape[0]:
                supported += int(distance[y, x] <= tolerance_px)
                total += 1
    return supported / total if total else 0.0


def boundary_gradient_score(quad: np.ndarray, gradient_u8: np.ndarray, tolerance_px: int) -> float:
    mask = np.zeros_like(gradient_u8)
    cv2.polylines(mask, [np.round(order_quad(quad)).astype(np.int32)], True, 255, 1, cv2.LINE_AA)
    kernel_size = max(3, 2 * tolerance_px + 1)
    band = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    values = gradient_u8[band > 0]
    if values.size == 0:
        return 0.0
    # 90 is already a useful boundary gradient; clip stronger values.
    return float(np.clip(np.mean(values) / 90.0, 0.0, 1.0))


def quad_mask(shape: Sequence[int], quad: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(order_quad(quad)).astype(np.int32), 255)
    return mask


def density_scores(quad: np.ndarray, edge_map: np.ndarray) -> tuple[float, float, float]:
    inside_mask = quad_mask(edge_map.shape, quad)
    area = max(1, int(np.count_nonzero(inside_mask)))
    inside_density = float(np.count_nonzero((edge_map > 0) & (inside_mask > 0)) / area)

    radius = max(5, int(math.sqrt(area) * 0.035))
    outer = cv2.dilate(
        inside_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    )
    ring = (outer > 0) & (inside_mask == 0)
    ring_area = max(1, int(np.count_nonzero(ring)))
    outside_density = float(np.count_nonzero((edge_map > 0) & ring) / ring_area)

    # Useful interval for a document containing text/photo, without rewarding
    # extremely noisy regions indefinitely.
    if inside_density < 0.01:
        density_score = inside_density / 0.01
    elif inside_density <= 0.20:
        density_score = 1.0
    else:
        density_score = max(0.0, 1.0 - (inside_density - 0.20) / 0.35)

    contrast_ratio = inside_density / (outside_density + 0.012)
    contrast_score = float(np.clip((contrast_ratio - 0.65) / 1.5, 0.0, 1.0))
    combined = 0.65 * density_score + 0.35 * contrast_score
    return float(combined), inside_density, outside_density


def angle_score(quad: np.ndarray) -> float:
    pts = order_quad(quad)
    errors: list[float] = []
    for i in range(4):
        prev_vec = pts[(i - 1) % 4] - pts[i]
        next_vec = pts[(i + 1) % 4] - pts[i]
        denominator = np.linalg.norm(prev_vec) * np.linalg.norm(next_vec) + 1e-6
        cosine = float(np.clip(np.dot(prev_vec, next_vec) / denominator, -1.0, 1.0))
        angle = math.degrees(math.acos(cosine))
        errors.append(abs(angle - 90.0))
    mean_error = float(np.mean(errors))
    return float(np.clip(1.0 - mean_error / 38.0, 0.0, 1.0))


def aspect_score(quad: np.ndarray, expected_ratio: float) -> tuple[float, float]:
    pts = order_quad(quad)
    width = 0.5 * (np.linalg.norm(pts[1] - pts[0]) + np.linalg.norm(pts[2] - pts[3]))
    height = 0.5 * (np.linalg.norm(pts[3] - pts[0]) + np.linalg.norm(pts[2] - pts[1]))
    ratio = max(width, height) / max(1e-6, min(width, height))
    log_error = abs(math.log(max(ratio, 1e-6) / expected_ratio))
    score = math.exp(-((log_error / 0.26) ** 2))
    return float(score), float(ratio)


def rectangularity_score(quad: np.ndarray) -> float:
    pts = order_quad(quad)
    area = polygon_area(pts)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    rect_area = max(1.0, float(rect[1][0] * rect[1][1]))
    return float(np.clip(area / rect_area, 0.0, 1.0))


def area_score(area_ratio: float) -> float:
    # Broad prior: reject tiny regions and avoid selecting the entire frame.
    rise = np.clip((area_ratio - 0.025) / 0.09, 0.0, 1.0)
    fall = np.clip((0.94 - area_ratio) / 0.18, 0.0, 1.0)
    return float(min(rise, fall))


def border_penalty(quad: np.ndarray, shape: Sequence[int]) -> float:
    height, width = shape[:2]
    pts = order_quad(quad)
    margin = 0.007 * min(height, width)
    touches = np.sum(
        (pts[:, 0] <= margin)
        | (pts[:, 0] >= width - 1 - margin)
        | (pts[:, 1] <= margin)
        | (pts[:, 1] >= height - 1 - margin)
    )
    return float(touches / 4.0)


def frame_side_penalty(quad: np.ndarray, shape: Sequence[int], margin_ratio: float) -> float:
    """Penalize quadrilateral sides that coincide with the literal image frame."""
    height, width = shape[:2]
    pts = order_quad(quad)
    margin = max(2.0, margin_ratio * min(height, width))
    side_hits = 0
    for index in range(4):
        start = pts[index]
        end = pts[(index + 1) % 4]
        on_frame = (
            (start[0] <= margin and end[0] <= margin)
            or (start[0] >= width - 1 - margin and end[0] >= width - 1 - margin)
            or (start[1] <= margin and end[1] <= margin)
            or (start[1] >= height - 1 - margin and end[1] >= height - 1 - margin)
        )
        side_hits += int(on_frame)
    return float(side_hits / 4.0)


def foreground_distribution_scores(
    quad: np.ndarray,
    foreground_mask: np.ndarray,
) -> tuple[float, float, float]:
    """Measure how compactly foreground evidence fills a candidate.

    A real ID card normally distributes text, photo and security patterns over
    much of its surface. A wrong quad spanning a card plus a distant UI button
    contains large blank bands and therefore receives a lower grid occupancy.
    """
    pts = order_quad(quad)
    output_w, output_h = 240, 150
    destination = np.array(
        [[0, 0], [output_w - 1, 0], [output_w - 1, output_h - 1], [0, output_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts.astype(np.float32), destination)
    warped = cv2.warpPerspective(
        foreground_mask,
        matrix,
        (output_w, output_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    binary = warped > 0
    fill_ratio = float(np.mean(binary))

    rows, cols = 5, 8
    active = 0
    cell_densities: list[float] = []
    for row in range(rows):
        y0 = row * output_h // rows
        y1 = (row + 1) * output_h // rows
        for col in range(cols):
            x0 = col * output_w // cols
            x1 = (col + 1) * output_w // cols
            density = float(np.mean(binary[y0:y1, x0:x1]))
            cell_densities.append(density)
            active += int(density >= 0.055)
    occupancy = active / float(rows * cols)

    # Broad, forgiving score. Sparse white cards are accepted, while very empty
    # candidates are penalized. High fill is not punished because a coloured
    # card or dark background can legitimately occupy most pixels.
    fill_score = float(np.clip((fill_ratio - 0.035) / 0.22, 0.0, 1.0))
    occupancy_score = float(np.clip((occupancy - 0.18) / 0.58, 0.0, 1.0))
    combined = 0.45 * fill_score + 0.55 * occupancy_score
    return float(combined), fill_ratio, float(occupancy)


def foreground_leakage_penalty(
    quad: np.ndarray,
    foreground_mask: np.ndarray,
    expansion_ratio: float = 0.18,
) -> tuple[float, float]:
    """Pénalise un quadrilatère qui laisse du document juste à l'extérieur.

    Le candidat est agrandi localement. Les pixels de premier plan présents
    dans la couronne indiquent qu'une ligne interne a probablement été choisie
    comme bord. Le bruit éloigné n'intervient pas car il reste hors couronne.
    """
    points = order_quad(quad)
    center = points.mean(axis=0)
    expanded = center + (points - center) * (1.0 + 2.0 * expansion_ratio)
    height, width = foreground_mask.shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)

    inside = quad_mask(foreground_mask.shape, points) > 0
    outer = quad_mask(foreground_mask.shape, expanded) > 0
    ring = outer & (~inside)
    inside_foreground = int(np.count_nonzero((foreground_mask > 0) & inside))
    ring_foreground = int(np.count_nonzero((foreground_mask > 0) & ring))
    leakage_ratio = float(ring_foreground / max(1, inside_foreground))
    penalty = float(np.clip((leakage_ratio - 0.025) / 0.22, 0.0, 1.0))
    return penalty, leakage_ratio


def score_candidate(
    candidate: Candidate,
    maps: dict[str, np.ndarray],
    config: DetectorConfig,
) -> Candidate:
    height, width = maps["gray"].shape
    image_area = float(height * width)
    quad = order_quad(candidate.quad)
    area_ratio = polygon_area(quad) / image_area
    tolerance_px = max(2, int(round(min(height, width) * config.edge_tolerance_ratio)))

    aspect, measured_ratio = aspect_score(quad, config.expected_aspect_ratio)
    continuity = edge_continuity_score(quad, maps["edge_union"], tolerance_px)
    gradient = boundary_gradient_score(quad, maps["gradient"], tolerance_px)
    density, inside_density, outside_density = density_scores(quad, maps["edge_union"])
    angles = angle_score(quad)
    rectangularity = rectangularity_score(quad)
    area = area_score(area_ratio)
    border = border_penalty(quad, maps["gray"].shape)
    frame_side = frame_side_penalty(
        quad,
        maps["gray"].shape,
        config.frame_ignore_ratio,
    )
    foreground, foreground_fill, foreground_occupancy = foreground_distribution_scores(
        quad,
        maps["foreground_mask"],
    )
    leakage_penalty, leakage_ratio = foreground_leakage_penalty(
        quad,
        maps["foreground_mask"],
    )

    source_bonus = {
        "contours": 0.025,
        "lines": 0.02,
        "line_quad": 0.025,
        "density": 0.0,
        "blackhat": 0.0,
        "foreground": 0.045,
        "foreground_box": 0.04,
    }.get(
        candidate.source, 0.0
    )

    score = (
        0.17 * continuity
        + 0.14 * gradient
        + 0.18 * aspect
        + 0.10 * density
        + 0.14 * foreground
        + 0.10 * angles
        + 0.08 * rectangularity
        + 0.09 * area
        + source_bonus
        - 0.08 * border
        - 0.34 * frame_side
        - 0.18 * leakage_penalty
    )

    candidate.quad = quad
    candidate.score = float(np.clip(score, 0.0, 1.0))
    candidate.metrics = {
        "edge_continuity": continuity,
        "boundary_gradient": gradient,
        "aspect_score": aspect,
        "measured_aspect_ratio": measured_ratio,
        "density_score": density,
        "inside_edge_density": inside_density,
        "outside_edge_density": outside_density,
        "angle_score": angles,
        "rectangularity": rectangularity,
        "area_ratio": area_ratio,
        "area_score": area,
        "border_penalty": border,
        "frame_side_penalty": frame_side,
        "foreground_score": foreground,
        "foreground_fill_ratio": foreground_fill,
        "foreground_grid_occupancy": foreground_occupancy,
        "foreground_leakage_ratio": leakage_ratio,
        "foreground_leakage_penalty": leakage_penalty,
    }
    return candidate


def convex_iou(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    a = order_quad(quad_a).astype(np.float32)
    b = order_quad(quad_b).astype(np.float32)
    area_a = polygon_area(a)
    area_b = polygon_area(b)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - float(intersection)
    return float(intersection / union) if union > 0 else 0.0


def deduplicate_candidates(candidates: Iterable[Candidate], iou_threshold: float = 0.78) -> list[Candidate]:
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if all(convex_iou(candidate.quad, other.quad) < iou_threshold for other in kept):
            kept.append(candidate)
    return kept


def expand_quad(quad: np.ndarray, margin_ratio: float, shape: Sequence[int]) -> np.ndarray:
    pts = order_quad(quad)
    center = pts.mean(axis=0)
    expanded = center + (pts - center) * (1.0 + 2.0 * margin_ratio)
    height, width = shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return order_quad(expanded)


def warp_card(
    image: np.ndarray,
    quad: np.ndarray,
    expected_ratio: float,
    margin_ratio: float = 0.0,
) -> np.ndarray:
    pts = expand_quad(quad, margin_ratio, image.shape) if margin_ratio else order_quad(quad)
    tl, tr, br, bl = pts
    measured_width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    measured_height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    # Always produce landscape output for an ID-1 card. No metadata is required.
    long_side = int(round(max(measured_width, measured_height)))
    short_side = max(1, int(round(long_side / expected_ratio)))
    output_width, output_height = long_side, short_side

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    # If detected sides indicate portrait orientation, rotate source ordering.
    horizontal = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    vertical = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    source = pts.copy()
    if horizontal < vertical:
        source = np.array([bl, tl, tr, br], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(source.astype(np.float32), destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def draw_debug(image: np.ndarray, candidates: Sequence[Candidate], limit: int = 12) -> np.ndarray:
    canvas = image.copy()
    for rank, candidate in enumerate(candidates[:limit], start=1):
        quad = np.round(candidate.quad).astype(np.int32)
        thickness = 4 if rank == 1 else 2
        color = (0, 255, 0) if rank == 1 else (0, max(70, 230 - rank * 12), 255)
        cv2.polylines(canvas, [quad], True, color, thickness, cv2.LINE_AA)
        x, y = quad[0]
        label = f"#{rank} {candidate.source} {candidate.score:.3f}"
        cv2.putText(
            canvas,
            label,
            (int(x), max(20, int(y) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def detect_card(
    image: np.ndarray,
    config: DetectorConfig | None = None,
) -> tuple[Candidate | None, list[Candidate], dict[str, np.ndarray], float]:
    config = config or DetectorConfig()
    working, scale = resize_for_work(image, config.max_working_side)
    maps = preprocess(working, config)

    raw_candidates: list[Candidate] = []
    raw_candidates.extend(contour_detector(maps, config))
    raw_candidates.extend(line_detector(maps, config))
    raw_candidates.extend(line_quad_detector(maps, config))
    raw_candidates.extend(density_detector(maps, config))
    raw_candidates.extend(foreground_detector(maps, config))

    scored = [score_candidate(candidate, maps, config) for candidate in raw_candidates]
    candidates = deduplicate_candidates(scored)
    best = candidates[0] if candidates else None
    return best, candidates, maps, scale


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    config: DetectorConfig,
    pdf_page: int = 0,
    pdf_dpi: int = 220,
) -> dict:
    image = load_input(input_path, pdf_page=pdf_page, pdf_dpi=pdf_dpi)
    best, candidates, maps, scale = detect_card(image, config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    working, _ = resize_for_work(image, config.max_working_side)
    debug = draw_debug(working, candidates)
    save_image(output_dir / "debug_candidates.jpg", debug)

    debug_maps = {
        "00_frame_artifact_mask.png": maps["frame_artifact_mask"],
        "00_analysis_image.png": maps["analysis_image"],
        "01_gray.png": maps["gray"],
        "02_gradient.png": maps["gradient"],
        "03_edges.png": maps["edge_union"],
        "04_connected_edges.png": maps["connected_edges"],
        "05_density_mask.png": maps["density_mask"],
        "06_line_mask.png": maps.get("line_mask", np.zeros_like(maps["gray"])),
        "07_blackhat.png": maps.get("blackhat", np.zeros_like(maps["gray"])),
        "08_foreground_mask.png": maps.get("foreground_mask", np.zeros_like(maps["gray"])),
        "09_colour_distance.png": maps.get("colour_distance", np.zeros_like(maps["gray"])),
    }
    for filename, debug_map in debug_maps.items():
        save_image(output_dir / filename, debug_map)

    result: dict = {
        "input": str(input_path),
        "config": asdict(config),
        "working_scale": scale,
        "status": "NO_CARD_FOUND",
        "candidate_count": len(candidates),
        "detected_frame_bands_px": maps.get("frame_bands", {}),
        "candidates": [candidate.to_json() for candidate in candidates[:20]],
    }

    if best is not None:
        original_quad = best.quad / scale
        crop_strict = warp_card(image, original_quad, config.expected_aspect_ratio, margin_ratio=0.0)
        crop_margin = warp_card(
            image,
            original_quad,
            config.expected_aspect_ratio,
            margin_ratio=config.final_margin_ratio,
        )
        save_image(output_dir / "crop_strict.png", crop_strict)
        save_image(output_dir / "crop_margin.png", crop_margin)

        confidence = best.score
        status = "SUCCESS" if confidence >= 0.55 else "LOW_CONFIDENCE"
        result.update(
            {
                "status": status,
                "confidence": round(float(confidence), 6),
                "source": best.source,
                "quad_original": np.round(original_quad, 2).tolist(),
                "metrics": best.metrics,
                "crop_strict": str(output_dir / "crop_strict.png"),
                "crop_margin": str(output_dir / "crop_margin.png"),
            }
        )

    (output_dir / "result.json").write_text(
        dumps_json(result),
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid smart crop: contours + Hough lines + gradient/density."
    )
    parser.add_argument("input", help="Input image or PDF")
    parser.add_argument("--output", default="smart_crop_output", help="Output directory")
    parser.add_argument("--page", type=int, default=0, help="PDF page, zero-based")
    parser.add_argument("--dpi", type=int, default=220, help="PDF rendering DPI")
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.586,
        help="Expected long-side / short-side ratio",
    )
    parser.add_argument("--min-area", type=float, default=0.035, help="Minimum image area ratio")
    parser.add_argument("--max-area", type=float, default=0.92, help="Maximum image area ratio")
    parser.add_argument("--max-side", type=int, default=1800, help="Maximum working image side")
    parser.add_argument(
        "--edge-tolerance",
        type=float,
        default=0.004,
        help="Tolerance relative to short image side for interrupted borders",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DetectorConfig(
        expected_aspect_ratio=args.ratio,
        min_area_ratio=args.min_area,
        max_area_ratio=args.max_area,
        max_working_side=args.max_side,
        edge_tolerance_ratio=args.edge_tolerance,
    )
    result = run_pipeline(
        args.input,
        args.output,
        config,
        pdf_page=args.page,
        pdf_dpi=args.dpi,
    )
    print(dumps_json(result))


if __name__ == "__main__":
    main()
