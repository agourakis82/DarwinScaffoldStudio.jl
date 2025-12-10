#!/usr/bin/env python3
"""Analyze SEM images from Kaique thesis to extract pore characteristics"""

import glob
import os

import numpy as np
from PIL import Image



def analyze_image(filepath):
    """Basic analysis of image to identify if it's likely SEM"""
    img = Image.open(filepath)
    arr = np.array(img)

    # Get basic stats
    width, height = img.size

    # Check if grayscale-ish (SEM images are usually grayscale)
    if len(arr.shape) == 3:
        # Check color variance - SEM is usually low color variance
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        color_var = np.std([r.mean(), g.mean(), b.mean()])
        is_grayscale_like = color_var < 20

        # Convert to grayscale for analysis
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = arr
        is_grayscale_like = True

    # Contrast analysis
    contrast = gray.std()

    # Look for typical SEM info bar (bottom portion usually darker/different)
    bottom_10pct = gray[int(0.9 * height) :, :]
    has_info_bar = bottom_10pct.mean() < gray[: int(0.9 * height), :].mean() - 20

    return {
        "file": os.path.basename(filepath),
        "size": f"{width}x{height}",
        "is_grayscale_like": is_grayscale_like,
        "contrast": contrast,
        "has_info_bar": has_info_bar,
        "likely_sem": is_grayscale_like
        and contrast > 30
        and (width > 800 or height > 800),
    }


# Analyze all images
images = sorted(glob.glob("kaique_images/*.png"))
print("=" * 70)
print("  ANÁLISE DAS IMAGENS EXTRAÍDAS DA TESE DO KAIQUE")
print("=" * 70)

sem_candidates = []
for img_path in images:
    result = analyze_image(img_path)
    status = "✓ SEM" if result["likely_sem"] else "  ---"
    print(
        f"{status}  {result['file']:25s}  {result['size']:12s}  contrast={result['contrast']:.1f}"
    )
    if result["likely_sem"]:
        sem_candidates.append(img_path)

print(f"\nImagens candidatas a SEM: {len(sem_candidates)}")
for img in sem_candidates:
    print(f"  - {img}")
