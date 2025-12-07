"""
    PhysicalPropertiesLibrary

Comprehensive physical properties database for biomaterials.

Contains:
- Mechanical properties (elastic modulus, tensile/compressive strength, toughness)
- Thermal properties (melting point, glass transition, thermal conductivity)
- Electrical properties (conductivity, dielectric constant)
- Optical properties (refractive index, transparency)
- Surface properties (contact angle, surface energy, roughness)
- Rheological properties (viscosity, shear modulus)
- Structural properties (density, porosity, crystallinity)

Data sources:
- ASM Materials Database
- MatWeb Material Property Data
- CES EduPack
- Published Q1 literature

Author: Dr. Demetrios Agourakis
"""
module PhysicalPropertiesLibrary

export PhysicalProperties, MechanicalProperties, ThermalProperties
export ElectricalProperties, OpticalProperties, SurfaceProperties
export RheologicalProperties, StructuralProperties
export PHYSICAL_DB, get_physical_properties
export calculate_scaffold_modulus, gibson_ashby_modulus
export compare_materials, rank_by_property
export get_tissue_matching_materials

# =============================================================================
# Property Structures
# =============================================================================

"""Mechanical properties of materials."""
struct MechanicalProperties
    elastic_modulus_mpa::Float64       # Young's modulus (E)
    shear_modulus_mpa::Float64         # Shear modulus (G)
    bulk_modulus_mpa::Float64          # Bulk modulus (K)
    poisson_ratio::Float64             # Poisson's ratio (ν)
    tensile_strength_mpa::Float64      # Ultimate tensile strength (UTS)
    yield_strength_mpa::Float64        # Yield strength
    compressive_strength_mpa::Float64  # Compressive strength
    flexural_strength_mpa::Float64     # Bending strength
    flexural_modulus_mpa::Float64      # Bending modulus
    elongation_percent::Float64        # Elongation at break
    hardness_shore::String             # Shore hardness (e.g., "Shore A 80")
    toughness_kj_m2::Float64          # Fracture toughness
    fatigue_strength_mpa::Float64      # Fatigue limit
    creep_resistance::String           # Qualitative rating
    wear_resistance::String            # Qualitative rating
end

function MechanicalProperties(;
    elastic_modulus_mpa::Float64=NaN,
    shear_modulus_mpa::Float64=NaN,
    bulk_modulus_mpa::Float64=NaN,
    poisson_ratio::Float64=NaN,
    tensile_strength_mpa::Float64=NaN,
    yield_strength_mpa::Float64=NaN,
    compressive_strength_mpa::Float64=NaN,
    flexural_strength_mpa::Float64=NaN,
    flexural_modulus_mpa::Float64=NaN,
    elongation_percent::Float64=NaN,
    hardness_shore::String="",
    toughness_kj_m2::Float64=NaN,
    fatigue_strength_mpa::Float64=NaN,
    creep_resistance::String="",
    wear_resistance::String="")

    MechanicalProperties(elastic_modulus_mpa, shear_modulus_mpa, bulk_modulus_mpa,
        poisson_ratio, tensile_strength_mpa, yield_strength_mpa, compressive_strength_mpa,
        flexural_strength_mpa, flexural_modulus_mpa, elongation_percent, hardness_shore,
        toughness_kj_m2, fatigue_strength_mpa, creep_resistance, wear_resistance)
end

"""Thermal properties of materials."""
struct ThermalProperties
    melting_point_c::Float64           # Melting temperature
    glass_transition_c::Float64        # Glass transition temperature (Tg)
    crystallization_temp_c::Float64    # Crystallization temperature
    decomposition_temp_c::Float64      # Thermal decomposition
    max_service_temp_c::Float64        # Maximum continuous use
    min_service_temp_c::Float64        # Minimum service temperature
    thermal_conductivity_w_mk::Float64 # Thermal conductivity
    specific_heat_j_gk::Float64        # Specific heat capacity
    thermal_expansion_ppm_k::Float64   # Coefficient of thermal expansion (CTE)
    thermal_diffusivity_mm2_s::Float64 # Thermal diffusivity
    latent_heat_j_g::Float64          # Latent heat of fusion
    heat_distortion_temp_c::Float64    # HDT under load
    vicat_softening_c::Float64         # Vicat softening point
    flammability::String               # UL94 rating
end

function ThermalProperties(;
    melting_point_c::Float64=NaN,
    glass_transition_c::Float64=NaN,
    crystallization_temp_c::Float64=NaN,
    decomposition_temp_c::Float64=NaN,
    max_service_temp_c::Float64=NaN,
    min_service_temp_c::Float64=NaN,
    thermal_conductivity_w_mk::Float64=NaN,
    specific_heat_j_gk::Float64=NaN,
    thermal_expansion_ppm_k::Float64=NaN,
    thermal_diffusivity_mm2_s::Float64=NaN,
    latent_heat_j_g::Float64=NaN,
    heat_distortion_temp_c::Float64=NaN,
    vicat_softening_c::Float64=NaN,
    flammability::String="")

    ThermalProperties(melting_point_c, glass_transition_c, crystallization_temp_c,
        decomposition_temp_c, max_service_temp_c, min_service_temp_c,
        thermal_conductivity_w_mk, specific_heat_j_gk, thermal_expansion_ppm_k,
        thermal_diffusivity_mm2_s, latent_heat_j_g, heat_distortion_temp_c,
        vicat_softening_c, flammability)
end

"""Electrical properties of materials."""
struct ElectricalProperties
    electrical_resistivity_ohm_m::Float64  # Resistivity
    electrical_conductivity_s_m::Float64   # Conductivity
    dielectric_constant::Float64           # Relative permittivity (εr)
    dielectric_strength_kv_mm::Float64     # Breakdown voltage
    dissipation_factor::Float64            # Loss tangent (tan δ)
    volume_resistivity_ohm_cm::Float64     # Volume resistivity
    surface_resistivity_ohm::Float64       # Surface resistivity
    arc_resistance_s::Float64              # Arc resistance
    piezoelectric_coeff::Float64           # d33 coefficient (pC/N)
    is_conductive::Bool                    # Conductive (>10^-6 S/m)
    is_piezoelectric::Bool                 # Has piezoelectric response
    is_ferroelectric::Bool                 # Has ferroelectric response
end

function ElectricalProperties(;
    electrical_resistivity_ohm_m::Float64=NaN,
    electrical_conductivity_s_m::Float64=NaN,
    dielectric_constant::Float64=NaN,
    dielectric_strength_kv_mm::Float64=NaN,
    dissipation_factor::Float64=NaN,
    volume_resistivity_ohm_cm::Float64=NaN,
    surface_resistivity_ohm::Float64=NaN,
    arc_resistance_s::Float64=NaN,
    piezoelectric_coeff::Float64=NaN,
    is_conductive::Bool=false,
    is_piezoelectric::Bool=false,
    is_ferroelectric::Bool=false)

    ElectricalProperties(electrical_resistivity_ohm_m, electrical_conductivity_s_m,
        dielectric_constant, dielectric_strength_kv_mm, dissipation_factor,
        volume_resistivity_ohm_cm, surface_resistivity_ohm, arc_resistance_s,
        piezoelectric_coeff, is_conductive, is_piezoelectric, is_ferroelectric)
end

"""Optical properties of materials."""
struct OpticalProperties
    refractive_index::Float64              # n at 589 nm
    refractive_index_range::Tuple{Float64,Float64}  # n range
    abbe_number::Float64                   # Dispersion
    transparency_percent::Float64          # Visible light transmission
    haze_percent::Float64                  # Light scattering
    color::String                          # Natural color
    uv_stability::String                   # UV resistance rating
    birefringence::Float64                 # Optical anisotropy
    fluorescence::Bool                     # Exhibits fluorescence
    optical_clarity::String                # Qualitative
end

function OpticalProperties(;
    refractive_index::Float64=NaN,
    refractive_index_range::Tuple{Float64,Float64}=(NaN, NaN),
    abbe_number::Float64=NaN,
    transparency_percent::Float64=NaN,
    haze_percent::Float64=NaN,
    color::String="",
    uv_stability::String="",
    birefringence::Float64=NaN,
    fluorescence::Bool=false,
    optical_clarity::String="")

    OpticalProperties(refractive_index, refractive_index_range, abbe_number,
        transparency_percent, haze_percent, color, uv_stability, birefringence,
        fluorescence, optical_clarity)
end

"""Surface properties of materials."""
struct SurfaceProperties
    water_contact_angle_deg::Float64       # Wettability
    surface_energy_mj_m2::Float64          # Surface free energy
    surface_roughness_ra_um::Float64       # Average roughness
    surface_roughness_rz_um::Float64       # Mean peak-valley height
    zeta_potential_mv::Float64             # Surface charge
    protein_adsorption::String             # Low/Medium/High
    cell_adhesion::String                  # Rating
    hydrophilicity::Symbol                 # :hydrophilic, :hydrophobic, :amphiphilic
    functional_groups_available::Vector{String}  # Surface chemistry
end

function SurfaceProperties(;
    water_contact_angle_deg::Float64=NaN,
    surface_energy_mj_m2::Float64=NaN,
    surface_roughness_ra_um::Float64=NaN,
    surface_roughness_rz_um::Float64=NaN,
    zeta_potential_mv::Float64=NaN,
    protein_adsorption::String="",
    cell_adhesion::String="",
    hydrophilicity::Symbol=:neutral,
    functional_groups_available::Vector{String}=String[])

    SurfaceProperties(water_contact_angle_deg, surface_energy_mj_m2,
        surface_roughness_ra_um, surface_roughness_rz_um, zeta_potential_mv,
        protein_adsorption, cell_adhesion, hydrophilicity, functional_groups_available)
end

"""Rheological properties (for polymers and hydrogels)."""
struct RheologicalProperties
    viscosity_pa_s::Float64                # Dynamic viscosity
    intrinsic_viscosity_dl_g::Float64      # [η]
    melt_flow_index_g_10min::Float64       # MFI
    storage_modulus_pa::Float64            # G' (elastic)
    loss_modulus_pa::Float64               # G'' (viscous)
    complex_viscosity_pa_s::Float64        # η*
    tan_delta::Float64                     # G''/G'
    yield_stress_pa::Float64               # Flow onset
    thixotropy::Bool                       # Time-dependent thinning
    shear_thinning::Bool                   # Pseudoplastic
    gelation_time_min::Float64             # Time to gel
    gel_point_temp_c::Float64              # Gelation temperature
    swelling_ratio::Float64                # Hydrogel swelling
end

function RheologicalProperties(;
    viscosity_pa_s::Float64=NaN,
    intrinsic_viscosity_dl_g::Float64=NaN,
    melt_flow_index_g_10min::Float64=NaN,
    storage_modulus_pa::Float64=NaN,
    loss_modulus_pa::Float64=NaN,
    complex_viscosity_pa_s::Float64=NaN,
    tan_delta::Float64=NaN,
    yield_stress_pa::Float64=NaN,
    thixotropy::Bool=false,
    shear_thinning::Bool=false,
    gelation_time_min::Float64=NaN,
    gel_point_temp_c::Float64=NaN,
    swelling_ratio::Float64=NaN)

    RheologicalProperties(viscosity_pa_s, intrinsic_viscosity_dl_g, melt_flow_index_g_10min,
        storage_modulus_pa, loss_modulus_pa, complex_viscosity_pa_s, tan_delta,
        yield_stress_pa, thixotropy, shear_thinning, gelation_time_min,
        gel_point_temp_c, swelling_ratio)
end

"""Structural properties of materials."""
struct StructuralProperties
    density_g_cm3::Float64                 # Bulk density
    specific_gravity::Float64              # Relative to water
    molecular_weight_da::Float64           # Polymer MW
    molecular_weight_distribution::Float64 # PDI (Mw/Mn)
    crystallinity_percent::Float64         # Degree of crystallinity
    crystal_structure::String              # Crystal system
    porosity_percent::Float64              # Void fraction
    pore_size_um::Float64                  # Mean pore diameter
    surface_area_m2_g::Float64            # BET surface area
    water_absorption_percent::Float64      # 24h immersion
    moisture_content_percent::Float64      # Equilibrium moisture
    shrinkage_percent::Float64             # Processing shrinkage
end

function StructuralProperties(;
    density_g_cm3::Float64=NaN,
    specific_gravity::Float64=NaN,
    molecular_weight_da::Float64=NaN,
    molecular_weight_distribution::Float64=NaN,
    crystallinity_percent::Float64=NaN,
    crystal_structure::String="",
    porosity_percent::Float64=NaN,
    pore_size_um::Float64=NaN,
    surface_area_m2_g::Float64=NaN,
    water_absorption_percent::Float64=NaN,
    moisture_content_percent::Float64=NaN,
    shrinkage_percent::Float64=NaN)

    StructuralProperties(density_g_cm3, specific_gravity, molecular_weight_da,
        molecular_weight_distribution, crystallinity_percent, crystal_structure,
        porosity_percent, pore_size_um, surface_area_m2_g, water_absorption_percent,
        moisture_content_percent, shrinkage_percent)
end

"""Complete physical properties for a material."""
struct PhysicalProperties
    id::String
    name::String
    category::Symbol  # :polymer, :ceramic, :metal, :composite, :hydrogel
    mechanical::MechanicalProperties
    thermal::ThermalProperties
    electrical::ElectricalProperties
    optical::OpticalProperties
    surface::SurfaceProperties
    rheological::RheologicalProperties
    structural::StructuralProperties
end

# =============================================================================
# Physical Properties Database
# =============================================================================

const PHYSICAL_DB = Dict{String,PhysicalProperties}(
    # =========================================================================
    # SYNTHETIC BIODEGRADABLE POLYMERS
    # =========================================================================

    "PCL" => PhysicalProperties(
        "CHEBI:53310", "Polycaprolactone", :polymer,
        MechanicalProperties(
            elastic_modulus_mpa=400.0,
            shear_modulus_mpa=145.0,
            poisson_ratio=0.38,
            tensile_strength_mpa=25.0,
            yield_strength_mpa=14.0,
            compressive_strength_mpa=20.0,
            flexural_modulus_mpa=350.0,
            elongation_percent=700.0,
            hardness_shore="Shore D 55",
            toughness_kj_m2=35.0,
            creep_resistance="Low",
            wear_resistance="Medium"
        ),
        ThermalProperties(
            melting_point_c=60.0,
            glass_transition_c=-60.0,
            decomposition_temp_c=350.0,
            max_service_temp_c=50.0,
            thermal_conductivity_w_mk=0.21,
            specific_heat_j_gk=1.9,
            thermal_expansion_ppm_k=140.0,
            heat_distortion_temp_c=55.0,
            flammability="HB"
        ),
        ElectricalProperties(
            electrical_resistivity_ohm_m=1e14,
            dielectric_constant=2.8,
            dielectric_strength_kv_mm=20.0,
            is_conductive=false
        ),
        OpticalProperties(
            refractive_index=1.476,
            transparency_percent=80.0,
            color="White to off-white",
            uv_stability="Fair"
        ),
        SurfaceProperties(
            water_contact_angle_deg=75.0,
            surface_energy_mj_m2=42.0,
            hydrophilicity=:hydrophobic,
            protein_adsorption="Medium",
            cell_adhesion="Good"
        ),
        RheologicalProperties(
            viscosity_pa_s=100.0,
            melt_flow_index_g_10min=2.5,
            shear_thinning=true
        ),
        StructuralProperties(
            density_g_cm3=1.145,
            molecular_weight_da=80000.0,
            molecular_weight_distribution=2.0,
            crystallinity_percent=50.0,
            water_absorption_percent=0.5
        )
    ),

    "PLA" => PhysicalProperties(
        "CHEBI:53309", "Polylactic Acid", :polymer,
        MechanicalProperties(
            elastic_modulus_mpa=3500.0,
            shear_modulus_mpa=1300.0,
            poisson_ratio=0.35,
            tensile_strength_mpa=60.0,
            yield_strength_mpa=48.0,
            compressive_strength_mpa=80.0,
            flexural_strength_mpa=80.0,
            flexural_modulus_mpa=3800.0,
            elongation_percent=6.0,
            hardness_shore="Shore D 83",
            toughness_kj_m2=2.5,
            fatigue_strength_mpa=20.0,
            creep_resistance="Medium",
            wear_resistance="Medium"
        ),
        ThermalProperties(
            melting_point_c=175.0,
            glass_transition_c=60.0,
            crystallization_temp_c=100.0,
            decomposition_temp_c=300.0,
            max_service_temp_c=55.0,
            thermal_conductivity_w_mk=0.13,
            specific_heat_j_gk=1.8,
            thermal_expansion_ppm_k=74.0,
            heat_distortion_temp_c=55.0,
            vicat_softening_c=59.0,
            flammability="HB"
        ),
        ElectricalProperties(
            electrical_resistivity_ohm_m=1e16,
            dielectric_constant=3.0,
            dielectric_strength_kv_mm=24.0,
            is_conductive=false
        ),
        OpticalProperties(
            refractive_index=1.46,
            transparency_percent=90.0,
            color="Clear to white",
            uv_stability="Poor",
            optical_clarity="High"
        ),
        SurfaceProperties(
            water_contact_angle_deg=70.0,
            surface_energy_mj_m2=38.0,
            hydrophilicity=:hydrophobic,
            protein_adsorption="Low",
            cell_adhesion="Moderate"
        ),
        RheologicalProperties(
            viscosity_pa_s=500.0,
            melt_flow_index_g_10min=8.0,
            shear_thinning=true
        ),
        StructuralProperties(
            density_g_cm3=1.24,
            molecular_weight_da=100000.0,
            molecular_weight_distribution=1.8,
            crystallinity_percent=35.0,
            water_absorption_percent=0.5,
            shrinkage_percent=0.4
        )
    ),

    "PLGA" => PhysicalProperties(
        "CHEBI:53426", "Poly(lactic-co-glycolic acid)", :polymer,
        MechanicalProperties(
            elastic_modulus_mpa=2000.0,
            tensile_strength_mpa=45.0,
            compressive_strength_mpa=50.0,
            elongation_percent=4.0,
            hardness_shore="Shore D 75"
        ),
        ThermalProperties(
            melting_point_c=200.0,  # Depends on ratio
            glass_transition_c=50.0,  # 50:50 ratio
            decomposition_temp_c=280.0
        ),
        ElectricalProperties(is_conductive=false),
        OpticalProperties(color="White"),
        SurfaceProperties(
            water_contact_angle_deg=65.0,
            hydrophilicity=:hydrophobic
        ),
        RheologicalProperties(),
        StructuralProperties(
            density_g_cm3=1.30,
            molecular_weight_da=50000.0,
            crystallinity_percent=20.0
        )
    ),

    "PGA" => PhysicalProperties(
        "CHEBI:53312", "Polyglycolic Acid", :polymer,
        MechanicalProperties(
            elastic_modulus_mpa=7000.0,
            tensile_strength_mpa=70.0,
            elongation_percent=20.0,
            hardness_shore="Shore D 88"
        ),
        ThermalProperties(
            melting_point_c=225.0,
            glass_transition_c=35.0
        ),
        ElectricalProperties(is_conductive=false),
        OpticalProperties(color="White"),
        SurfaceProperties(
            water_contact_angle_deg=55.0,
            hydrophilicity=:hydrophilic
        ),
        RheologicalProperties(),
        StructuralProperties(
            density_g_cm3=1.53,
            crystallinity_percent=50.0
        )
    ),

    # =========================================================================
    # NATURAL POLYMERS
    # =========================================================================

    "Collagen" => PhysicalProperties(
        "CHEBI:3815", "Type I Collagen", :polymer,
        MechanicalProperties(
            elastic_modulus_mpa=5.0,  # Hydrated
            tensile_strength_mpa=1.0,
            elongation_percent=15.0
        ),
        ThermalProperties(
            decomposition_temp_c=60.0,  # Denaturation
            max_service_temp_c=37.0
        ),
        ElectricalProperties(is_conductive=false, is_piezoelectric=true),
        OpticalProperties(color="White", transparency_percent=40.0),
        SurfaceProperties(
            water_contact_angle_deg=45.0,
            hydrophilicity=:hydrophilic,
            protein_adsorption="High",
            cell_adhesion="Excellent",
            functional_groups_available=["amine", "carboxyl", "hydroxyl"]
        ),
        RheologicalProperties(
            gelation_time_min=30.0,
            gel_point_temp_c=37.0
        ),
        StructuralProperties(
            density_g_cm3=1.3,
            water_absorption_percent=200.0
        )
    ),

    "GelMA" => PhysicalProperties(
        "GelMA", "Methacrylated Gelatin", :hydrogel,
        MechanicalProperties(
            elastic_modulus_mpa=0.05,  # 50 kPa typical
            compressive_strength_mpa=0.1
        ),
        ThermalProperties(
            max_service_temp_c=37.0
        ),
        ElectricalProperties(is_conductive=false),
        OpticalProperties(transparency_percent=85.0, color="Light yellow"),
        SurfaceProperties(
            hydrophilicity=:hydrophilic,
            cell_adhesion="Excellent",
            functional_groups_available=["methacrylate", "amine", "RGD"]
        ),
        RheologicalProperties(
            storage_modulus_pa=5000.0,
            loss_modulus_pa=500.0,
            gelation_time_min=2.0,  # UV curing
            gel_point_temp_c=25.0,
            swelling_ratio=10.0
        ),
        StructuralProperties(
            density_g_cm3=1.05,
            water_absorption_percent=1000.0
        )
    ),

    "Alginate" => PhysicalProperties(
        "CHEBI:52747", "Sodium Alginate", :hydrogel,
        MechanicalProperties(
            elastic_modulus_mpa=0.02,  # 20 kPa
            compressive_strength_mpa=0.05
        ),
        ThermalProperties(
            decomposition_temp_c=200.0
        ),
        ElectricalProperties(is_conductive=false),
        OpticalProperties(transparency_percent=95.0, color="Colorless"),
        SurfaceProperties(
            hydrophilicity=:hydrophilic,
            cell_adhesion="Poor",  # No cell adhesion peptides
            functional_groups_available=["carboxyl", "hydroxyl"]
        ),
        RheologicalProperties(
            viscosity_pa_s=0.5,
            gelation_time_min=1.0,  # Ca2+ crosslinking
            shear_thinning=true,
            swelling_ratio=20.0
        ),
        StructuralProperties(
            density_g_cm3=1.6,  # Dry
            water_absorption_percent=500.0
        )
    ),

    "Chitosan" => PhysicalProperties(
        "CHEBI:16737", "Chitosan", :polymer,
        MechanicalProperties(
            elastic_modulus_mpa=100.0,  # Film
            tensile_strength_mpa=30.0,
            elongation_percent=10.0
        ),
        ThermalProperties(
            decomposition_temp_c=200.0,
            glass_transition_c=140.0
        ),
        ElectricalProperties(is_conductive=false),
        OpticalProperties(color="Light yellow"),
        SurfaceProperties(
            water_contact_angle_deg=60.0,
            hydrophilicity=:hydrophilic,
            zeta_potential_mv=40.0,  # Positive at low pH
            cell_adhesion="Good",
            functional_groups_available=["amine", "hydroxyl", "acetyl"]
        ),
        RheologicalProperties(
            viscosity_pa_s=1.0,
            shear_thinning=true
        ),
        StructuralProperties(
            density_g_cm3=1.4,
            crystallinity_percent=20.0,
            water_absorption_percent=100.0
        )
    ),

    "Hyaluronic_Acid" => PhysicalProperties(
        "CHEBI:18154", "Hyaluronic Acid", :hydrogel,
        MechanicalProperties(
            elastic_modulus_mpa=0.001  # Very soft, 1 kPa
        ),
        ThermalProperties(
            decomposition_temp_c=200.0
        ),
        ElectricalProperties(is_conductive=false),
        OpticalProperties(transparency_percent=99.0, color="Colorless"),
        SurfaceProperties(
            hydrophilicity=:hydrophilic,
            cell_adhesion="Medium",
            functional_groups_available=["carboxyl", "hydroxyl", "acetamido"]
        ),
        RheologicalProperties(
            viscosity_pa_s=1000.0,  # High MW
            shear_thinning=true,
            thixotropy=true,
            swelling_ratio=100.0
        ),
        StructuralProperties(
            molecular_weight_da=1000000.0,  # 1 MDa typical
            water_absorption_percent=1000.0
        )
    ),

    # =========================================================================
    # CERAMICS
    # =========================================================================

    "Hydroxyapatite" => PhysicalProperties(
        "CHEBI:52251", "Hydroxyapatite", :ceramic,
        MechanicalProperties(
            elastic_modulus_mpa=80000.0,
            shear_modulus_mpa=35000.0,
            poisson_ratio=0.28,
            tensile_strength_mpa=40.0,
            compressive_strength_mpa=500.0,
            flexural_strength_mpa=100.0,
            toughness_kj_m2=1.0,
            hardness_shore="Vickers 600"
        ),
        ThermalProperties(
            melting_point_c=1670.0,
            decomposition_temp_c=1200.0,
            thermal_conductivity_w_mk=1.25,
            thermal_expansion_ppm_k=11.0
        ),
        ElectricalProperties(
            is_conductive=false,
            is_piezoelectric=true,
            piezoelectric_coeff=0.1
        ),
        OpticalProperties(
            refractive_index=1.65,
            color="White"
        ),
        SurfaceProperties(
            water_contact_angle_deg=10.0,
            hydrophilicity=:hydrophilic,
            protein_adsorption="High",
            cell_adhesion="Excellent"
        ),
        RheologicalProperties(),
        StructuralProperties(
            density_g_cm3=3.16,
            crystal_structure="Hexagonal",
            surface_area_m2_g=50.0
        )
    ),

    "TCP" => PhysicalProperties(
        "CHEBI:53480", "Tricalcium Phosphate", :ceramic,
        MechanicalProperties(
            elastic_modulus_mpa=30000.0,
            compressive_strength_mpa=150.0,
            tensile_strength_mpa=25.0,
            toughness_kj_m2=0.5
        ),
        ThermalProperties(
            melting_point_c=1670.0,
            decomposition_temp_c=1100.0,
            thermal_conductivity_w_mk=1.1
        ),
        ElectricalProperties(is_conductive=false),
        OpticalProperties(color="White"),
        SurfaceProperties(
            hydrophilicity=:hydrophilic,
            cell_adhesion="Good"
        ),
        RheologicalProperties(),
        StructuralProperties(
            density_g_cm3=3.14,
            crystal_structure="Monoclinic (alpha) or Rhombohedral (beta)",
            surface_area_m2_g=40.0
        )
    ),

    "Bioglass_45S5" => PhysicalProperties(
        "CHEBI:52254", "45S5 Bioactive Glass", :ceramic,
        MechanicalProperties(
            elastic_modulus_mpa=35000.0,
            compressive_strength_mpa=500.0,
            tensile_strength_mpa=42.0,
            flexural_strength_mpa=50.0,
            toughness_kj_m2=0.7,
            hardness_shore="Vickers 460"
        ),
        ThermalProperties(
            melting_point_c=1050.0,
            glass_transition_c=550.0,
            thermal_conductivity_w_mk=1.4,
            thermal_expansion_ppm_k=15.1
        ),
        ElectricalProperties(is_conductive=false),
        OpticalProperties(
            transparency_percent=0.0,
            color="White to tan"
        ),
        SurfaceProperties(
            hydrophilicity=:hydrophilic,
            cell_adhesion="Excellent"
        ),
        RheologicalProperties(),
        StructuralProperties(
            density_g_cm3=2.7,
            crystal_structure="Amorphous"
        )
    ),

    # =========================================================================
    # METALS
    # =========================================================================

    "Titanium" => PhysicalProperties(
        "CHEBI:33341", "Titanium (CP Grade 2)", :metal,
        MechanicalProperties(
            elastic_modulus_mpa=110000.0,
            shear_modulus_mpa=44000.0,
            poisson_ratio=0.34,
            tensile_strength_mpa=345.0,
            yield_strength_mpa=275.0,
            compressive_strength_mpa=480.0,
            elongation_percent=20.0,
            hardness_shore="HRC 36",
            toughness_kj_m2=70.0,
            fatigue_strength_mpa=300.0
        ),
        ThermalProperties(
            melting_point_c=1668.0,
            thermal_conductivity_w_mk=21.9,
            specific_heat_j_gk=0.52,
            thermal_expansion_ppm_k=8.6
        ),
        ElectricalProperties(
            electrical_resistivity_ohm_m=5.5e-7,
            electrical_conductivity_s_m=1.8e6,
            is_conductive=true
        ),
        OpticalProperties(
            color="Silver-gray"
        ),
        SurfaceProperties(
            water_contact_angle_deg=60.0,
            protein_adsorption="Medium",
            cell_adhesion="Good"
        ),
        RheologicalProperties(),
        StructuralProperties(
            density_g_cm3=4.51,
            crystal_structure="Hexagonal close-packed (alpha)"
        )
    ),

    "Ti6Al4V" => PhysicalProperties(
        "Ti6Al4V", "Titanium Alloy (Ti-6Al-4V)", :metal,
        MechanicalProperties(
            elastic_modulus_mpa=114000.0,
            shear_modulus_mpa=44000.0,
            poisson_ratio=0.34,
            tensile_strength_mpa=950.0,
            yield_strength_mpa=830.0,
            compressive_strength_mpa=970.0,
            elongation_percent=14.0,
            hardness_shore="HRC 36",
            toughness_kj_m2=75.0,
            fatigue_strength_mpa=500.0
        ),
        ThermalProperties(
            melting_point_c=1660.0,
            thermal_conductivity_w_mk=6.7,
            specific_heat_j_gk=0.56,
            thermal_expansion_ppm_k=8.6
        ),
        ElectricalProperties(
            electrical_resistivity_ohm_m=1.7e-6,
            is_conductive=true
        ),
        OpticalProperties(color="Silver-gray"),
        SurfaceProperties(
            cell_adhesion="Good"
        ),
        RheologicalProperties(),
        StructuralProperties(
            density_g_cm3=4.43,
            crystal_structure="Alpha-beta"
        )
    ),

    "316L_SS" => PhysicalProperties(
        "316L_SS", "316L Stainless Steel", :metal,
        MechanicalProperties(
            elastic_modulus_mpa=193000.0,
            shear_modulus_mpa=77000.0,
            poisson_ratio=0.27,
            tensile_strength_mpa=485.0,
            yield_strength_mpa=170.0,
            elongation_percent=40.0,
            hardness_shore="HRB 79",
            fatigue_strength_mpa=250.0
        ),
        ThermalProperties(
            melting_point_c=1400.0,
            thermal_conductivity_w_mk=16.3,
            thermal_expansion_ppm_k=16.0
        ),
        ElectricalProperties(
            electrical_resistivity_ohm_m=7.4e-7,
            is_conductive=true
        ),
        OpticalProperties(color="Silver"),
        SurfaceProperties(
            cell_adhesion="Moderate"
        ),
        RheologicalProperties(),
        StructuralProperties(
            density_g_cm3=8.0,
            crystal_structure="Face-centered cubic (austenite)"
        )
    ),

    "Magnesium" => PhysicalProperties(
        "CHEBI:22977", "Magnesium (Biodegradable)", :metal,
        MechanicalProperties(
            elastic_modulus_mpa=45000.0,
            shear_modulus_mpa=17000.0,
            poisson_ratio=0.35,
            tensile_strength_mpa=220.0,
            yield_strength_mpa=160.0,
            elongation_percent=8.0,
            hardness_shore="HV 45"
        ),
        ThermalProperties(
            melting_point_c=650.0,
            thermal_conductivity_w_mk=156.0,
            thermal_expansion_ppm_k=26.0
        ),
        ElectricalProperties(
            electrical_resistivity_ohm_m=4.4e-8,
            is_conductive=true
        ),
        OpticalProperties(color="Silver-white"),
        SurfaceProperties(
            cell_adhesion="Good"
        ),
        RheologicalProperties(),
        StructuralProperties(
            density_g_cm3=1.74,
            crystal_structure="Hexagonal close-packed"
        )
    )
)

# =============================================================================
# Tissue Mechanical Properties (Reference)
# =============================================================================

const TISSUE_MECHANICAL_PROPERTIES = Dict{Symbol,NamedTuple}(
    :cortical_bone => (E_mpa = 15000.0, sigma_mpa = 150.0, description = "Cortical bone"),
    :trabecular_bone => (E_mpa = 500.0, sigma_mpa = 5.0, description = "Trabecular bone"),
    :cartilage => (E_mpa = 10.0, sigma_mpa = 25.0, description = "Articular cartilage"),
    :tendon => (E_mpa = 1500.0, sigma_mpa = 100.0, description = "Tendon"),
    :ligament => (E_mpa = 400.0, sigma_mpa = 40.0, description = "Ligament"),
    :skin => (E_mpa = 50.0, sigma_mpa = 20.0, description = "Skin (dermis)"),
    :muscle => (E_mpa = 0.1, sigma_mpa = 0.3, description = "Skeletal muscle"),
    :artery => (E_mpa = 1.0, sigma_mpa = 2.0, description = "Arterial wall"),
    :cardiac => (E_mpa = 0.5, sigma_mpa = 0.1, description = "Cardiac tissue"),
    :nerve => (E_mpa = 0.5, sigma_mpa = 0.05, description = "Peripheral nerve"),
    :brain => (E_mpa = 0.003, sigma_mpa = 0.001, description = "Brain tissue")
)

# =============================================================================
# Calculation Functions
# =============================================================================

"""
    gibson_ashby_modulus(E_solid, porosity; n=2.0)

Calculate scaffold modulus using Gibson-Ashby model.
E_scaffold = E_solid × (1 - porosity)^n

Parameters:
- E_solid: Bulk material modulus (MPa)
- porosity: Porosity fraction (0-1)
- n: Gibson-Ashby exponent (2.0 for open-cell foam)
"""
function gibson_ashby_modulus(E_solid::Float64, porosity::Float64; n::Float64=2.0)
    @assert 0.0 <= porosity <= 1.0 "Porosity must be between 0 and 1"
    return E_solid * (1 - porosity)^n
end

"""
    calculate_scaffold_modulus(material_id, porosity; architecture=:foam)

Calculate scaffold modulus for given material and porosity.
"""
function calculate_scaffold_modulus(material_id::String, porosity::Float64;
                                    architecture::Symbol=:foam)
    props = get(PHYSICAL_DB, material_id, nothing)
    if isnothing(props)
        error("Material $material_id not found in database")
    end

    E_solid = props.mechanical.elastic_modulus_mpa
    if isnan(E_solid)
        error("Elastic modulus not available for $material_id")
    end

    # Different architectures have different exponents
    n = if architecture == :foam
        2.0
    elseif architecture == :gyroid
        1.8
    elseif architecture == :lattice
        1.5
    else
        2.0
    end

    return gibson_ashby_modulus(E_solid, porosity; n=n)
end

"""
    get_tissue_matching_materials(tissue, porosity; tolerance=0.5)

Find materials that match tissue mechanical properties.
"""
function get_tissue_matching_materials(tissue::Symbol, porosity::Float64;
                                       tolerance::Float64=0.5)
    tissue_props = get(TISSUE_MECHANICAL_PROPERTIES, tissue, nothing)
    if isnothing(tissue_props)
        error("Tissue $tissue not found in database")
    end

    target_E = tissue_props.E_mpa
    matches = Tuple{String,Float64,Float64}[]

    for (id, props) in PHYSICAL_DB
        E_solid = props.mechanical.elastic_modulus_mpa
        if !isnan(E_solid)
            E_scaffold = gibson_ashby_modulus(E_solid, porosity)

            # Check if within tolerance
            ratio = E_scaffold / target_E
            if (1 - tolerance) <= ratio <= (1 + tolerance)
                push!(matches, (id, E_scaffold, ratio))
            end
        end
    end

    # Sort by closeness to target
    sort!(matches, by=x -> abs(x[3] - 1.0))

    return matches
end

"""
    compare_materials(material_ids, properties)

Compare materials across specified properties.
"""
function compare_materials(material_ids::Vector{String},
                          properties::Vector{Symbol}=[:elastic_modulus_mpa, :tensile_strength_mpa])
    result = Dict{String,Dict{Symbol,Any}}()

    for id in material_ids
        props = get(PHYSICAL_DB, id, nothing)
        if !isnothing(props)
            result[id] = Dict{Symbol,Any}()
            for prop in properties
                # Try to get from each property struct
                for field in [:mechanical, :thermal, :electrical, :optical, :surface, :structural]
                    substruct = getfield(props, field)
                    if hasfield(typeof(substruct), prop)
                        result[id][prop] = getfield(substruct, prop)
                        break
                    end
                end
            end
        end
    end

    return result
end

"""
    rank_by_property(property, ascending=true)

Rank materials by a specific property.
"""
function rank_by_property(property::Symbol; ascending::Bool=true)
    values = Tuple{String,Float64}[]

    for (id, props) in PHYSICAL_DB
        for field in [:mechanical, :thermal, :electrical, :optical, :surface, :structural]
            substruct = getfield(props, field)
            if hasfield(typeof(substruct), property)
                val = getfield(substruct, property)
                if val isa Number && !isnan(val)
                    push!(values, (id, val))
                end
                break
            end
        end
    end

    sort!(values, by=x -> x[2], rev=!ascending)
    return values
end

"""Get physical properties by material ID."""
function get_physical_properties(id::String)
    return get(PHYSICAL_DB, id, nothing)
end

end # module
