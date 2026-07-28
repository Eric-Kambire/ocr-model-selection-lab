from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    # Chargement comme module du dépôt.
    from .smart_crop import load_input, save_image
except ImportError:
    # Lancement direct depuis le dossier autonome.
    from smart_crop import load_input, save_image


@dataclass
class LabExport:
    zip_path: str
    png_path: str
    pdf_path: str
    json_path: str
    mask_path: str | None


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def editor_composite(value: Any) -> np.ndarray | None:
    """Return an RGB uint8 image from a Gradio ImageEditor value."""
    if value is None:
        return None
    if isinstance(value, dict):
        image = value.get("composite")
        if image is None:
            image = value.get("background")
    else:
        image = value
    if image is None:
        return None
    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"))
    elif isinstance(image, (str, Path)):
        array = np.array(Image.open(image).convert("RGB"))
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif array.shape[2] == 4:
            # Composite RGBA on white rather than silently discarding alpha.
            rgba = array.astype(np.float32)
            alpha = rgba[:, :, 3:4] / 255.0
            array = rgba[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def blank_state() -> dict[str, Any]:
    return {
        "original": None,
        "current": None,
        "corners": [],
        "operations": [],
        "source_name": None,
        "seed": 1234,
    }


def load_for_lab(path: str | Path, pdf_page: int = 0, pdf_dpi: int = 300) -> tuple[np.ndarray, dict]:
    bgr = load_input(path, pdf_page=pdf_page, pdf_dpi=pdf_dpi)
    rgb = bgr_to_rgb(bgr)
    state = blank_state()
    state.update(
        {
            "original": rgb.copy(),
            "current": rgb.copy(),
            "source_name": Path(path).name,
            "operations": [
                {
                    "type": "load",
                    "source": Path(path).name,
                    "pdf_page": int(pdf_page),
                    "pdf_dpi": int(pdf_dpi),
                }
            ],
        }
    )
    return rgb, state


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = (value or "#000000").strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (0, 0, 0)


def _alpha_blend(base: np.ndarray, overlay: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha_mask.astype(np.float32), 0.0, 1.0)[..., None]
    output = base.astype(np.float32) * (1.0 - alpha) + overlay.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def _transform_points(points: list[list[float]], matrix: np.ndarray) -> list[list[float]]:
    if not points:
        return []
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, matrix.astype(np.float32)).reshape(-1, 2)
    return transformed.astype(float).tolist()


def draw_annotations(image: np.ndarray, corners: list[list[float]]) -> np.ndarray:
    canvas = image.copy()
    if not corners:
        return canvas
    pts = np.round(np.asarray(corners, dtype=np.float32)).astype(np.int32)
    for index, (x, y) in enumerate(pts):
        cv2.circle(canvas, (int(x), int(y)), 7, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x), int(y)), 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index + 1),
            (int(x) + 10, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    if len(pts) == 4:
        cv2.polylines(canvas, [pts], True, (0, 200, 0), 3, cv2.LINE_AA)
    elif len(pts) > 1:
        cv2.polylines(canvas, [pts], False, (0, 200, 0), 2, cv2.LINE_AA)
    return canvas


def apply_local_effect(
    image: np.ndarray,
    effect: str,
    x_percent: float,
    y_percent: float,
    size: int,
    thickness: int,
    opacity: float,
    angle_deg: float,
    count: int,
    blur: int,
    color_hex: str,
    text: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Apply a deterministic local degradation to an RGB image."""
    if image is None:
        raise ValueError("Aucune image chargée dans le laboratoire.")
    output = image.copy()
    height, width = output.shape[:2]
    cx = int(np.clip(float(x_percent), 0, 100) / 100.0 * max(width - 1, 1))
    cy = int(np.clip(float(y_percent), 0, 100) / 100.0 * max(height - 1, 1))
    size = max(1, int(size))
    thickness = max(1, int(thickness))
    opacity = float(np.clip(opacity, 0.0, 1.0))
    count = max(1, int(count))
    blur = max(0, int(blur))
    if blur % 2 == 0 and blur > 0:
        blur += 1
    color = _hex_to_rgb(color_hex)
    rng = np.random.default_rng(int(seed))

    overlay = output.copy()
    mask = np.zeros((height, width), dtype=np.float32)
    theta = math.radians(float(angle_deg))
    dx = math.cos(theta)
    dy = math.sin(theta)

    if effect == "Point":
        cv2.circle(overlay, (cx, cy), size, color, -1, cv2.LINE_AA)
        cv2.circle(mask, (cx, cy), size, 1.0, -1, cv2.LINE_AA)

    elif effect == "Tache irrégulière":
        for _ in range(count):
            px = int(cx + rng.normal(0, size * 0.45))
            py = int(cy + rng.normal(0, size * 0.45))
            radius = max(1, int(size * rng.uniform(0.25, 0.75)))
            cv2.circle(overlay, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(mask, (px, py), radius, 1.0, -1, cv2.LINE_AA)

    elif effect in {"Tiret", "Ligne"}:
        length = size * (3 if effect == "Tiret" else 8)
        p1 = (int(cx - dx * length / 2), int(cy - dy * length / 2))
        p2 = (int(cx + dx * length / 2), int(cy + dy * length / 2))
        cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
        cv2.line(mask, p1, p2, 1.0, thickness, cv2.LINE_AA)

    elif effect in {"Zone blanche", "Occultation noire"}:
        fill = (255, 255, 255) if effect == "Zone blanche" else (0, 0, 0)
        half_w = size * 2
        half_h = max(size, thickness * 2)
        p1 = (max(0, cx - half_w), max(0, cy - half_h))
        p2 = (min(width - 1, cx + half_w), min(height - 1, cy + half_h))
        cv2.rectangle(overlay, p1, p2, fill, -1)
        cv2.rectangle(mask, p1, p2, 1.0, -1)

    elif effect in {"Ombre localisée", "Reflet lumineux"}:
        fill = (0, 0, 0) if effect == "Ombre localisée" else (255, 255, 255)
        axes = (max(2, size * 3), max(2, size * 2))
        cv2.ellipse(overlay, (cx, cy), axes, angle_deg, 0, 360, fill, -1, cv2.LINE_AA)
        cv2.ellipse(mask, (cx, cy), axes, angle_deg, 0, 360, 1.0, -1, cv2.LINE_AA)
        local_blur = blur if blur > 0 else max(9, size | 1)
        if local_blur % 2 == 0:
            local_blur += 1
        mask = cv2.GaussianBlur(mask, (local_blur, local_blur), 0)

    elif effect == "Poussière":
        radius_max = max(1, thickness)
        spread = max(size * 3, 3)
        for _ in range(count):
            px = int(np.clip(cx + rng.normal(0, spread), 0, width - 1))
            py = int(np.clip(cy + rng.normal(0, spread), 0, height - 1))
            radius = int(rng.integers(1, radius_max + 1))
            cv2.circle(overlay, (px, py), radius, color, -1, cv2.LINE_AA)
            cv2.circle(mask, (px, py), radius, 1.0, -1, cv2.LINE_AA)

    elif effect == "Bord noir du scanner":
        distances = {
            "left": cx,
            "right": width - 1 - cx,
            "top": cy,
            "bottom": height - 1 - cy,
        }
        side = min(distances, key=distances.get)
        border = max(1, size)
        if side == "left":
            overlay[:, :border] = (0, 0, 0)
            mask[:, :border] = 1.0
        elif side == "right":
            overlay[:, width - border :] = (0, 0, 0)
            mask[:, width - border :] = 1.0
        elif side == "top":
            overlay[:border, :] = (0, 0, 0)
            mask[:border, :] = 1.0
        else:
            overlay[height - border :, :] = (0, 0, 0)
            mask[height - border :, :] = 1.0

    elif effect == "Texte parasite":
        message = text.strip() or "COPY"
        scale = max(0.4, size / 18.0)
        cv2.putText(
            overlay,
            message,
            (max(0, cx - size * 2), min(height - 1, cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        # Text mask obtained by drawing the same glyph in white.
        cv2.putText(
            mask,
            message,
            (max(0, cx - size * 2), min(height - 1, cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            1.0,
            thickness,
            cv2.LINE_AA,
        )

    else:
        raise ValueError(f"Effet local non reconnu : {effect}")

    if blur > 0 and effect not in {"Ombre localisée", "Reflet lumineux"}:
        mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    output = _alpha_blend(output, overlay, mask * opacity)
    metadata = {
        "type": "local_effect",
        "effect": effect,
        "center_percent": [float(x_percent), float(y_percent)],
        "center_px": [cx, cy],
        "size": size,
        "thickness": thickness,
        "opacity": opacity,
        "angle_deg": float(angle_deg),
        "count": count,
        "blur": blur,
        "color": color_hex,
        "text": text,
        "seed": int(seed),
    }
    return output, metadata


def _apply_matrix(image: np.ndarray, matrix: np.ndarray, border_value=(255, 255, 255)) -> np.ndarray:
    height, width = image.shape[:2]
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def apply_global_degradations(
    image: np.ndarray,
    corners: list[list[float]],
    rotation_deg: float,
    perspective: float,
    translate_x: float,
    translate_y: float,
    brightness: float,
    contrast: float,
    shadow: float,
    shadow_angle: float,
    glare: float,
    glare_x: float,
    glare_y: float,
    focus_blur: int,
    motion_blur: int,
    motion_angle: float,
    gaussian_noise: float,
    jpeg_quality: int,
    downscale: float,
    nonuniform_background: float,
    seed: int,
) -> tuple[np.ndarray, list[list[float]], dict]:
    if image is None:
        raise ValueError("Aucune image chargée dans le laboratoire.")
    output = image.copy()
    height, width = output.shape[:2]
    rng = np.random.default_rng(int(seed))
    current_corners = [list(map(float, point)) for point in (corners or [])]
    total_matrix = np.eye(3, dtype=np.float32)

    # Perspective: independently perturb the four image-frame corners.
    perspective = float(np.clip(perspective, 0.0, 0.25))
    if perspective > 0:
        src = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        max_shift = perspective * min(height, width)
        offsets = rng.uniform(-max_shift, max_shift, size=(4, 2)).astype(np.float32)
        dst = src + offsets
        matrix = cv2.getPerspectiveTransform(src, dst)
        output = _apply_matrix(output, matrix)
        total_matrix = matrix @ total_matrix

    # Rotation around the image centre; no EXIF is used.
    if abs(float(rotation_deg)) > 1e-6:
        affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(rotation_deg), 1.0)
        matrix = np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float32)
        output = _apply_matrix(output, matrix)
        total_matrix = matrix @ total_matrix

    # Translation deliberately allows partial card clipping.
    tx = float(translate_x) / 100.0 * width
    ty = float(translate_y) / 100.0 * height
    if abs(tx) > 1e-6 or abs(ty) > 1e-6:
        matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        output = _apply_matrix(output, matrix)
        total_matrix = matrix @ total_matrix

    if current_corners:
        current_corners = _transform_points(current_corners, total_matrix)

    # Contrast and brightness: I' = alpha * I + beta.
    alpha = float(max(0.05, contrast))
    beta = float(brightness)
    output = np.clip(output.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Smooth global illumination gradient / shadow.
    if float(shadow) > 0:
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        theta = math.radians(float(shadow_angle))
        projection = xx * math.cos(theta) + yy * math.sin(theta)
        projection -= projection.min()
        projection /= max(float(projection.max()), 1e-6)
        light = 1.0 - float(np.clip(shadow, 0, 1)) * projection
        output = np.clip(output.astype(np.float32) * light[..., None], 0, 255).astype(np.uint8)

    # Non-uniform support/lighting texture.
    if float(nonuniform_background) > 0:
        small_h = max(2, height // 32)
        small_w = max(2, width // 32)
        field = rng.normal(0, 1, (small_h, small_w)).astype(np.float32)
        field = cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC)
        field = cv2.GaussianBlur(field, (0, 0), sigmaX=max(width, height) / 18.0)
        field /= max(float(np.max(np.abs(field))), 1e-6)
        output = np.clip(
            output.astype(np.float32) + field[..., None] * float(nonuniform_background) * 45.0,
            0,
            255,
        ).astype(np.uint8)

    # Global glare as a smooth 2-D Gaussian lobe.
    if float(glare) > 0:
        gx = np.clip(float(glare_x), 0, 100) / 100.0 * width
        gy = np.clip(float(glare_y), 0, 100) / 100.0 * height
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        sigma_x = max(width * 0.18, 1.0)
        sigma_y = max(height * 0.12, 1.0)
        gaussian = np.exp(-(((xx - gx) ** 2) / (2 * sigma_x**2) + ((yy - gy) ** 2) / (2 * sigma_y**2)))
        alpha_glare = np.clip(gaussian * float(glare), 0.0, 0.95)[..., None]
        output = np.clip(output * (1.0 - alpha_glare) + 255.0 * alpha_glare, 0, 255).astype(np.uint8)

    focus_blur = max(0, int(focus_blur))
    if focus_blur > 0:
        if focus_blur % 2 == 0:
            focus_blur += 1
        output = cv2.GaussianBlur(output, (focus_blur, focus_blur), 0)

    motion_blur = max(0, int(motion_blur))
    if motion_blur > 1:
        kernel = np.zeros((motion_blur, motion_blur), dtype=np.float32)
        cv2.line(
            kernel,
            (0, motion_blur // 2),
            (motion_blur - 1, motion_blur // 2),
            1.0,
            1,
        )
        rotation = cv2.getRotationMatrix2D(
            ((motion_blur - 1) / 2.0, (motion_blur - 1) / 2.0),
            float(motion_angle),
            1.0,
        )
        kernel = cv2.warpAffine(kernel, rotation, (motion_blur, motion_blur))
        kernel_sum = float(kernel.sum())
        if kernel_sum > 0:
            kernel /= kernel_sum
            output = cv2.filter2D(output, -1, kernel)

    if float(gaussian_noise) > 0:
        noise = rng.normal(0, float(gaussian_noise), output.shape).astype(np.float32)
        output = np.clip(output.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    downscale = float(np.clip(downscale, 0.10, 1.0))
    if downscale < 0.999:
        small = cv2.resize(
            output,
            (max(1, int(width * downscale)), max(1, int(height * downscale))),
            interpolation=cv2.INTER_AREA,
        )
        output = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)

    jpeg_quality = int(np.clip(jpeg_quality, 10, 100))
    if jpeg_quality < 100:
        bgr = rgb_to_bgr(output)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            output = bgr_to_rgb(decoded)

    metadata = {
        "type": "global_degradations",
        "rotation_deg": float(rotation_deg),
        "perspective_strength": perspective,
        "translation_percent": [float(translate_x), float(translate_y)],
        "brightness": beta,
        "contrast": alpha,
        "shadow": float(shadow),
        "shadow_angle": float(shadow_angle),
        "glare": float(glare),
        "glare_center_percent": [float(glare_x), float(glare_y)],
        "focus_blur_kernel": focus_blur,
        "motion_blur_kernel": motion_blur,
        "motion_angle": float(motion_angle),
        "gaussian_noise_sigma": float(gaussian_noise),
        "jpeg_quality": jpeg_quality,
        "downscale_factor": downscale,
        "nonuniform_background": float(nonuniform_background),
        "seed": int(seed),
        "geometric_matrix": total_matrix.astype(float).tolist(),
    }
    return output, current_corners, metadata


def create_mask(shape: tuple[int, int], corners: list[list[float]]) -> np.ndarray | None:
    if len(corners) != 4:
        return None
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(np.asarray(corners, dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    return mask


def export_lab(state: dict[str, Any]) -> LabExport:
    if not state or state.get("current") is None or state.get("original") is None:
        raise ValueError("Aucune image à exporter.")
    output_dir = Path(tempfile.mkdtemp(prefix="smart_crop_lab_"))
    original_rgb = np.asarray(state["original"], dtype=np.uint8)
    current_rgb = np.asarray(state["current"], dtype=np.uint8)
    original_path = output_dir / "original.png"
    degraded_path = output_dir / "degraded.png"
    pdf_path = output_dir / "degraded.pdf"
    json_path = output_dir / "metadata.json"

    save_image(original_path, rgb_to_bgr(original_rgb))
    save_image(degraded_path, rgb_to_bgr(current_rgb))
    Image.fromarray(current_rgb).convert("RGB").save(pdf_path, "PDF", resolution=300.0)

    corners = state.get("corners") or []
    mask = create_mask(current_rgb.shape[:2], corners)
    mask_path: Path | None = None
    if mask is not None:
        mask_path = output_dir / "card_mask.png"
        save_image(mask_path, mask)

    metadata = {
        "version": "3.0",
        "source": state.get("source_name"),
        "image_size": {"width": int(current_rgb.shape[1]), "height": int(current_rgb.shape[0])},
        "card_corners": [[round(float(x), 3), round(float(y), 3)] for x, y in corners],
        "mask_available": mask is not None,
        "operations": state.get("operations", []),
    }
    json_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = Path(shutil.make_archive(str(output_dir), "zip", root_dir=output_dir))
    return LabExport(
        zip_path=str(zip_path),
        png_path=str(degraded_path),
        pdf_path=str(pdf_path),
        json_path=str(json_path),
        mask_path=str(mask_path) if mask_path else None,
    )
