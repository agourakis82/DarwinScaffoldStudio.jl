"""
Fabrication Constants

Shared default parameters for scaffold fabrication methods.
References literature values for freeze-casting, 3D-bioprinting, and salt-leaching.
"""

module FabricationConstants

export FREEZE_CASTING_TEMP_C, FREEZE_CASTING_RATE, FREEZE_CASTING_SOLUTE_CONC
export BIOPRINTING_NOZZLE_UM, BIOPRINTING_LAYER_UM, BIOPRINTING_SPEED
export SALT_LEACHING_SIZE_UM, SALT_LEACHING_FRACTION, SALT_LEACHING_TIME_H
export HIGH_POROSITY_THRESHOLD, LARGE_PORE_THRESHOLD_UM, SMALL_PORE_THRESHOLD_UM
export PORE_SIZE_MIN_UM, PORE_SIZE_MAX_UM, POROSITY_MIN, POROSITY_MAX
export INTERCONNECTIVITY_MIN, TORTUOSITY_MAX
export MIN_POROSITY_DIVISOR, MIN_PORE_SIZE_DIVISOR, MIN_INTERCONNECTIVITY_DIVISOR, MIN_TORTUOSITY_DIVISOR

# =============================================================================
# Literature-Based Thresholds
# =============================================================================

# Pore size thresholds (Murphy 2010: 100-200 um optimal for bone tissue engineering)
const PORE_SIZE_MIN_UM = 100.0
const PORE_SIZE_MAX_UM = 200.0

# Porosity thresholds (Karageorgiou 2005: 90-95% optimal for bone scaffolds)
const POROSITY_MIN = 0.90
const POROSITY_MAX = 0.95

# Interconnectivity threshold (Karageorgiou 2005: >=90% for nutrient transport)
const INTERCONNECTIVITY_MIN = 0.90

# Tortuosity threshold (lower is better for straight diffusion paths)
const TORTUOSITY_MAX = 1.2

# =============================================================================
# Fabrication Method Selection Thresholds
# =============================================================================

const HIGH_POROSITY_THRESHOLD = 0.93
const LARGE_PORE_THRESHOLD_UM = 150.0
const SMALL_PORE_THRESHOLD_UM = 120.0

# =============================================================================
# Freeze-Casting Defaults (Ma & Zhang 2001)
# =============================================================================

const FREEZE_CASTING_TEMP_C = -20.0
const FREEZE_CASTING_RATE = 1.0
const FREEZE_CASTING_SOLUTE_CONC = 0.1

# =============================================================================
# 3D-Bioprinting Defaults (Murphy & Atala 2014)
# =============================================================================

const BIOPRINTING_NOZZLE_UM = 100.0
const BIOPRINTING_LAYER_UM = 50.0
const BIOPRINTING_SPEED = 10.0

# =============================================================================
# Salt-Leaching Defaults (Mikos et al. 1994)
# =============================================================================

const SALT_LEACHING_SIZE_UM = 150.0
const SALT_LEACHING_FRACTION = 0.92
const SALT_LEACHING_TIME_H = 24.0

# =============================================================================
# Numerical Stability Constants
# =============================================================================

const MIN_POROSITY_DIVISOR = 0.01
const MIN_PORE_SIZE_DIVISOR = 1.0
const MIN_INTERCONNECTIVITY_DIVISOR = 0.01
const MIN_TORTUOSITY_DIVISOR = 1.0

end # module
