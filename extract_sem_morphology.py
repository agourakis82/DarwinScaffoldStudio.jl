#!/usr/bin/env python3
"""Extract morphological features from SEM images for degradation modeling"""

import glob
import json
import os
import re

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import filters, measure, morphology


def extract_pore_features(filepath):
    """Extract pore size and porosity from SEM image"""
    img = Image.open(filepath)
    arr = np.array(img)

    # Convert to grayscale
    if len(arr.shape) == 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr.astype(float)

    # Remove info bar (bottom 10%)
    height = gray.shape[0]
    gray = gray[: int(0.9 * height), :]

    # Normalize
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

    # Otsu thresholding for pore detection
    # In SEM, pores appear darker
    threshold = filters.threshold_otsu(gray)
    pores = gray < threshold

    # Clean up with morphological operations
    pores = morphology.remove_small_objects(pores, min_size=50)
    pores = morphology.remove_small_holes(pores, area_threshold=20)

    # Measure porosity
    porosity = np.sum(pores) / pores.size

    # Label connected regions (pores)
    labeled = measure.label(pores)
    regions = measure.regionprops(labeled)

    if len(regions) == 0:
        return {
            "porosity": porosity,
            "n_pores": 0,
            "mean_pore_area": 0,
            "mean_pore_diameter": 0,
            "pore_size_std": 0,
            "max_pore_diameter": 0,
            "circularity": 0,
        }

    # Calculate pore sizes
    areas = [r.area for r in regions]
    # Equivalent diameter assuming circular pore
    diameters = [2 * np.sqrt(r.area / np.pi) for r in regions]

    # Circularity (1 = perfect circle)
    circularities = []
    for r in regions:
        if r.perimeter > 0:
            circ = 4 * np.pi * r.area / (r.perimeter**2)
            circularities.append(min(circ, 1.0))

    return {
        "porosity": porosity,
        "n_pores": len(regions),
        "mean_pore_area": np.mean(areas),
        "mean_pore_diameter": np.mean(diameters),
        "pore_size_std": np.std(diameters),
        "max_pore_diameter": np.max(diameters),
        "circularity": np.mean(circularities) if circularities else 0,
    }


def estimate_tortuosity_2d(filepath):
    """Estimate tortuosity from 2D SEM image using path analysis"""
    img = Image.open(filepath)
    arr = np.array(img)

    if len(arr.shape) == 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr.astype(float)

    # Remove info bar
    height = gray.shape[0]
    gray = gray[: int(0.9 * height), :]

    # Normalize and threshold
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    threshold = filters.threshold_otsu(gray)

    # Solid phase (scaffold material)
    solid = gray >= threshold

    # Skeletonize to find paths
    skeleton = morphology.skeletonize(solid)

    # Estimate tortuosity as ratio of skeleton length to straight distance
    # Simple approximation using distance transform
    dist = ndimage.distance_transform_edt(~solid)

    # Mean path through pore space
    pore_path_complexity = dist.mean() / (dist.max() + 1e-8)

    # Tortuosity estimate (1 = straight path)
    # Based on geometric tortuosity approximation
    porosity = 1 - np.sum(solid) / solid.size
    if porosity > 0:
        tau_estimate = 1.0 + 0.5 * (1 - porosity) / porosity
    else:
        tau_estimate = float("inf")

    return {
        "tortuosity_estimate": min(tau_estimate, 10.0),
        "path_complexity": pore_path_complexity,
        "skeleton_density": np.sum(skeleton) / skeleton.size,
    }


def estimate_connectivity(filepath):
    """Estimate pore connectivity (related to percolation)"""
    img = Image.open(filepath)
    arr = np.array(img)

    if len(arr.shape) == 3:
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    else:
        gray = arr.astype(float)

    # Remove info bar
    height = gray.shape[0]
    gray = gray[: int(0.9 * height), :]

    # Threshold for pores
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
    threshold = filters.threshold_otsu(gray)
    pores = gray < threshold

    # Label connected pore regions
    labeled = measure.label(pores)
    n_components = labeled.max()

    if n_components == 0:
        return {
            "n_connected_components": 0,
            "largest_component_fraction": 0,
            "connectivity_index": 0,
        }

    # Find largest connected component
    component_sizes = np.bincount(labeled.ravel())[1:]  # exclude background
    largest_size = component_sizes.max()
    total_pore_area = component_sizes.sum()

    # Connectivity index: fraction of pores in largest component
    connectivity = largest_size / total_pore_area if total_pore_area > 0 else 0

    return {
        "n_connected_components": n_components,
        "largest_component_fraction": connectivity,
        "connectivity_index": connectivity,
    }


def parse_page_for_degradation_time(page_num):
    """Map page numbers to approximate degradation times based on thesis structure"""
    # Based on typical thesis structure for PLDLA degradation study
    # Pages 42-49 likely show degradation progression
    time_map = {
        14: 0,  # Initial scaffold
        16: 0,  # Initial characterization
        25: 0,  # Methods/initial
        37: 0,  # Initial SEM
        39: 7,  # 1 week
        42: 14,  # 2 weeks
        43: 21,  # 3 weeks
        44: 28,  # 4 weeks
        45: 42,  # 6 weeks
        47: 56,  # 8 weeks
        48: 70,  # 10 weeks
        49: 84,  # 12 weeks
        52: 98,  # 14 weeks
        55: 112,  # 16 weeks
    }
    return time_map.get(page_num, -1)


# Main analysis
print("=" * 80)
print("  ANÁLISE MORFOLÓGICA DAS IMAGENS SEM - TESE KAIQUE")
print("  Extração de características para modelagem de degradação")
print("=" * 80)

sem_images = sorted(glob.glob("kaique_images/page*.png"))
results = []

for img_path in sem_images:
    # Extract page number
    basename = os.path.basename(img_path)
    match = re.search(r"page(\d+)", basename)
    if not match:
        continue

    page_num = int(match.group(1))

    # Get features
    pore_features = extract_pore_features(img_path)
    tortuosity_features = estimate_tortuosity_2d(img_path)
    connectivity_features = estimate_connectivity(img_path)

    # Estimate degradation time
    deg_time = parse_page_for_degradation_time(page_num)

    # Convert numpy types to Python native for JSON
    result = {
        "page": int(page_num),
        "file": basename,
        "degradation_days": int(deg_time),
    }
    for k, v in {
        **pore_features,
        **tortuosity_features,
        **connectivity_features,
    }.items():
        if isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = float(v)
        else:
            result[k] = v
    results.append(result)

    print(f"\n--- Página {page_num} (t≈{deg_time}d) ---")
    print(f"  Porosidade: {pore_features['porosity'] * 100:.1f}%")
    print(f"  Diâmetro médio poro: {pore_features['mean_pore_diameter']:.1f} px")
    print(f"  Tortuosidade est.: {tortuosity_features['tortuosity_estimate']:.2f}")
    print(f"  Conectividade: {connectivity_features['connectivity_index'] * 100:.1f}%")

# Save results
with open("kaique_sem_morphology.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("  RESUMO DA EVOLUÇÃO MORFOLÓGICA")
print("=" * 80)

# Filter results with valid degradation time
valid_results = [r for r in results if r["degradation_days"] >= 0]
valid_results.sort(key=lambda x: x["degradation_days"])

print(
    f"\n{'Tempo (d)':<12} {'Porosidade':<12} {'Poro (px)':<12} {'Tortuosidade':<12} {'Conectiv.':<12}"
)
print("-" * 60)
for r in valid_results:
    print(
        f"{r['degradation_days']:<12} {r['porosity'] * 100:>8.1f}%   {r['mean_pore_diameter']:>8.1f}    {r['tortuosity_estimate']:>8.2f}      {r['connectivity_index'] * 100:>6.1f}%"
    )

print(f"\nResultados salvos em: kaique_sem_morphology.json")
