"""
Synthetic imaging samples and scoring helpers.

These utilities build rotated Gaussian tweezer-grid images with known ground
truth and score extraction results against it. They are shared by the blob
parameter optimizers in `aod_atommovr.imaging.optimization` and by the
imaging test suite, so they live in the package proper rather than in a test
module.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from aod_atommovr.imaging.generation import (
    compute_scaled_image_shape,
    generate_gaussian_image,
)
from aod_atommovr.imaging.geometry import rotate_points_ccw


def setup_blob_params(
    params: Optional[dict],
    image_shape: Optional[Tuple[int, int]] = None,
    grid_size: Optional[int] = None,
) -> cv2.SimpleBlobDetector_Params:
    if params is None:
        blob_params = cv2.SimpleBlobDetector_Params()
        blob_params.filterByColor = True
        blob_params.blobColor = 255
        blob_params.minThreshold = 70
        blob_params.maxThreshold = 255
        blob_params.thresholdStep = 20
        blob_params.minDistBetweenBlobs = 10
        blob_params.minArea = 5
        blob_params.maxArea = 1000
        blob_params.filterByArea = True
        blob_params.filterByCircularity = False
        blob_params.filterByConvexity = False
        blob_params.filterByInertia = False
    else:
        blob_params = cv2.SimpleBlobDetector_Params()
        for key, val in params.items():
            setattr(blob_params, key, val)

    if image_shape is not None and grid_size:
        min_dim = float(min(int(image_shape[0]), int(image_shape[1])))
        spacing = min_dim / max(grid_size + 1, 1)
        blob_params.minDistBetweenBlobs = float(max(2.0, 0.5 * spacing))
        # Loosen thresholds for small synthetic images so blobs are detected reliably
        if min_dim <= 512:
            blob_params.minThreshold = 20
            blob_params.thresholdStep = 10
            blob_params.minArea = 2
            blob_params.maxArea = max(50, int((0.6 * spacing) ** 2))

    return blob_params


def generate_rot_img_in_memory(
    image_shape: Tuple[int, int],
    grid_size: int,
    true_angle: float,
    *,
    save_images: bool = False,
    suffix: Optional[str] = None,
    directory: str = "output",
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray, Optional[str]]:
    """
    Generate a rotated Gaussian grid image and optionally save it.

    Returns
    -------
    points : List[Tuple[int, int]]
        List of (x, y) coordinates of the Gaussian centers.
    true_binary : np.ndarray
        Binary grid indicating presence (1) or absence (0) of Gaussians.
    rot_image : np.ndarray
        Rotated image array.
    rot_image_path : Optional[str]
        Path to the saved rotated image if requested.
    """
    image_shape = compute_scaled_image_shape(image_shape, grid_size)
    grid_shape = (grid_size, grid_size)
    points: List[Tuple[int, int]] = []
    start_x = image_shape[0] // grid_shape[0]
    start_y = image_shape[1] // grid_shape[1]
    spacing_x = image_shape[0] // (grid_shape[0] + 1)
    spacing_y = image_shape[1] // (grid_shape[1] + 1)
    true_binary = np.zeros(grid_shape, dtype=int)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if np.random.rand() > 0.4:
                points.append((start_x + i * spacing_x, start_y + j * spacing_y))
                true_binary[i, j] = 1

    sigmas = np.random.normal(1.5, 0.01, len(points))
    brightness_factors = np.random.uniform(0.8, 1.0, len(points))

    gaussian_img = generate_gaussian_image(
        points, sigmas, brightness_factors, image_shape, angle=0.0
    )
    rot_image = generate_gaussian_image(
        points, sigmas, brightness_factors, image_shape, angle=true_angle
    )

    rot_path = None
    if save_images:
        if suffix is None:
            raise ValueError("suffix must be provided when save_images=True")
        os.makedirs(directory, exist_ok=True)

        import imageio.v2 as imageio

        def _to_uint8(img: np.ndarray) -> np.ndarray:
            arr = np.asarray(img)
            if arr.dtype == np.uint8:
                return arr
            arr = arr.astype(np.float32, copy=False)
            min_val = float(np.min(arr)) if arr.size else 0.0
            max_val = float(np.max(arr)) if arr.size else 0.0
            if max_val <= min_val:
                return np.zeros(arr.shape, dtype=np.uint8)
            arr = (arr - min_val) / (max_val - min_val)
            return np.clip(arr * 255.0, 0, 255).astype(np.uint8)

        imageio.imwrite(f"{directory}/{suffix}_image.png", _to_uint8(gaussian_img))
        rot_path = f"{directory}/{suffix}_rot_image.png"
        imageio.imwrite(rot_path, _to_uint8(rot_image))
        print(f"Saved images to {directory}/{suffix}_image.png and {rot_path}")
    return points, true_binary, rot_image, rot_path


def generate_rot_img(
    image_shape: Tuple[int, int],
    grid_size: int,
    true_angle: float,
    suffix: Optional[str],
    directory: str = "output",
) -> Tuple[Tuple[int, int], List[Tuple[int, int]], np.ndarray]:
    """
    Generate a rotated Gaussian grid image and save it.
    """
    if suffix is None:
        raise ValueError("suffix must be provided for generate_rot_img")
    points, true_binary, _rot_image, _ = generate_rot_img_in_memory(
        image_shape,
        grid_size,
        true_angle,
        save_images=True,
        suffix=suffix,
        directory=directory,
    )
    return points, true_binary


def sample_sparse_grid_points(
    grid_size: int,
    image_shape: Tuple[int, int],
    load_probability: float = 0.6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Generate sparse grid coordinates matching generate_rot_img spacing."""
    if rng is None:
        rng = np.random.default_rng()
    image_shape = compute_scaled_image_shape(image_shape, grid_size)
    grid_shape = (grid_size, grid_size)
    img_h, img_w = image_shape[:2]
    start_row = float(img_h // grid_shape[0])
    start_col = float(img_w // grid_shape[1])
    row_spacing = float(img_h // (grid_shape[0] + 1))
    col_spacing = float(img_w // (grid_shape[1] + 1))
    points: List[Tuple[float, float]] = []
    binary = np.zeros(grid_shape, dtype=int)
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            if rng.random() < load_probability:
                points.append(
                    (start_row + i * row_spacing, start_col + j * col_spacing)
                )
                binary[i, j] = 1
    return np.asarray(points, dtype=float), binary, row_spacing, col_spacing


def rotate_points_about_center(
    points: np.ndarray, angle_deg: float, image_shape: Tuple[int, int]
) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(-1, 2)
    return rotate_points_ccw(points, image_shape, angle_deg)


def _wrap_angle_deg(angle: float) -> float:
    return (angle + 90.0) % 180.0 - 90.0


def angle_error_deg(estimate: float, truth: float) -> float:
    return _wrap_angle_deg(estimate - truth)


def compute_assignment_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=int)
    truth = np.asarray(truth, dtype=int)
    truth_ones = int(truth.sum())
    pred_ones = int(pred.sum())
    true_pos = int(np.logical_and(pred == 1, truth == 1).sum())
    recall = true_pos / truth_ones if truth_ones else 1.0
    precision = true_pos / pred_ones if pred_ones else 1.0
    return {
        "recall": recall,
        "precision": precision,
        "exact_match": bool(np.array_equal(pred, truth)),
        "true_positives": true_pos,
        "true_ones": truth_ones,
        "predicted_ones": pred_ones,
    }
