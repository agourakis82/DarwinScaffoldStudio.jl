#!/usr/bin/env python3
"""
Test SAM for pore segmentation on PoreScript dataset.

Compares:
1. Otsu thresholding
2. SAM (Segment Anything Model)

Usage: python scripts/test_sam_segmentation.py
"""

import warnings
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "validation" / "porescript"
PIXEL_SIZE_UM = 3.5


def load_image(path):
    """Load image as grayscale numpy array."""
    img = Image.open(path).convert("L")
    return np.array(img) / 255.0


def otsu_threshold(image):
    """Simple Otsu implementation."""
    from skimage.filters import threshold_otsu

    thresh = threshold_otsu(image)
    return image < thresh  # dark = pore


def compute_pore_sizes(mask, pixel_size):
    """Compute equivalent circular diameter for each pore."""
    from scipy import ndimage

    labeled, n_features = ndimage.label(mask)

    diameters = []
    for i in range(1, n_features + 1):
        area = np.sum(labeled == i)
        if area > 10:  # Filter noise
            d = 2.0 * np.sqrt(area / np.pi) * pixel_size
            diameters.append(d)

    return diameters


def segment_with_sam(image_path, min_area=100, max_area=50000):
    """Segment using SAM (Segment Anything Model) with pore-specific filtering."""
    import torch
    from transformers import pipeline

    # Use SAM pipeline
    device = 0 if torch.cuda.is_available() else -1
    print(f"  Using device: {'CUDA' if device == 0 else 'CPU'}")

    # Load SAM
    generator = pipeline(
        "mask-generation", model="facebook/sam-vit-base", device=device
    )

    # Load image as RGB and grayscale
    img = Image.open(image_path).convert("RGB")
    gray = np.array(Image.open(image_path).convert("L")) / 255.0

    # Generate masks with more points for finer detail
    outputs = generator(img, points_per_side=64)

    # Output format: {'masks': [np.array, ...], 'scores': tensor}
    masks = outputs["masks"]
    scores = (
        outputs["scores"].cpu().numpy()
        if hasattr(outputs["scores"], "cpu")
        else outputs["scores"]
    )

    # Combine masks for dark regions (pores) with size filtering
    combined_mask = np.zeros(gray.shape, dtype=bool)
    pore_count = 0

    for i, mask in enumerate(masks):
        area = np.sum(mask)
        # Filter by size (pores should be small to medium, not giant regions)
        if min_area < area < max_area:
            mean_intensity = np.mean(gray[mask])
            # Dark regions are pores (intensity < 0.4)
            if mean_intensity < 0.4:
                combined_mask |= mask
                pore_count += 1

    print(f"  SAM filtered: {pore_count} pore-like regions")
    return combined_mask


def segment_with_sam_auto(image_path):
    """Use SAM with automatic mask generation for pore-like objects."""
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    # This requires: pip install segment-anything
    # and downloading the model checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model (use vit_b for speed)
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )

    # Load image
    img = np.array(Image.open(image_path).convert("RGB"))

    # Generate masks
    masks = mask_generator.generate(img)

    # Combine masks for dark regions (pores)
    gray = np.array(Image.open(image_path).convert("L")) / 255.0
    combined = np.zeros(gray.shape, dtype=bool)

    for mask_data in masks:
        mask = mask_data["segmentation"]
        # Only include if the masked region is dark (pore)
        if np.mean(gray[mask]) < 0.5:
            combined |= mask

    return combined


def main():
    print("=" * 70)
    print("SAM vs OTSU COMPARISON FOR PORE SEGMENTATION")
    print("=" * 70)

    # Check dependencies
    try:
        from scipy import ndimage
        from skimage.filters import threshold_otsu

        print("\n[OK] scikit-image and scipy available")
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Install with: pip install scikit-image scipy")
        return

    # Check SAM availability
    sam_available = False
    try:
        import torch
        from transformers import pipeline

        print(f"[OK] transformers available (CUDA: {torch.cuda.is_available()})")
        sam_available = True
    except ImportError as e:
        print(f"[WARN] SAM not available: {e}")

    # Ground truth (from previous analysis)
    ground_truth = {
        "S1_27x": 170.2,  # mean pore size in μm
        "S2_27x": 175.7,
        "S3_27x": 176.9,
    }

    print("\n" + "-" * 70)
    print("Processing samples...")
    print("-" * 70)

    results = {"otsu": [], "sam": []}

    for sample, gt_mean in ground_truth.items():
        image_path = DATA_DIR / f"{sample}.tif"

        if not image_path.exists():
            print(f"\n[SKIP] {sample}: Image not found")
            continue

        print(f"\n>>> {sample} (GT: {gt_mean:.1f} μm)")

        # Load image
        image = load_image(image_path)

        # Method 1: Otsu
        mask_otsu = otsu_threshold(image)
        diams_otsu = compute_pore_sizes(mask_otsu, PIXEL_SIZE_UM)
        mean_otsu = np.mean(diams_otsu) if diams_otsu else 0
        error_otsu = (mean_otsu - gt_mean) / gt_mean * 100
        print(
            f"  Otsu:  {mean_otsu:.1f} μm ({len(diams_otsu)} pores) | Error: {error_otsu:+.1f}%"
        )
        results["otsu"].append(abs(error_otsu))

        # Method 2: SAM
        if sam_available:
            try:
                mask_sam = segment_with_sam(image_path)
                diams_sam = compute_pore_sizes(mask_sam, PIXEL_SIZE_UM)
                mean_sam = np.mean(diams_sam) if diams_sam else 0
                error_sam = (mean_sam - gt_mean) / gt_mean * 100
                print(
                    f"  SAM:   {mean_sam:.1f} μm ({len(diams_sam)} pores) | Error: {error_sam:+.1f}%"
                )
                results["sam"].append(abs(error_sam))
            except Exception as e:
                print(f"  SAM:   FAILED - {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results["otsu"]:
        print(f"\nOtsu Mean APE: {np.mean(results['otsu']):.1f}%")

    if results["sam"]:
        print(f"SAM Mean APE:  {np.mean(results['sam']):.1f}%")
        improvement = np.mean(results["otsu"]) - np.mean(results["sam"])
        print(f"\nImprovement:   {improvement:+.1f} percentage points")

        if improvement > 0:
            print("\n✓ SAM REDUCES ERROR")
        else:
            print("\n✗ SAM does not improve (may need fine-tuning)")


if __name__ == "__main__":
    main()
