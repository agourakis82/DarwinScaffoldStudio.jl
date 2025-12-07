"""
    DrugDeliveryLibrary

Comprehensive drug delivery and pharmacokinetics database for scaffold applications.

Contains:
- Drug pharmacokinetic parameters (absorption, distribution, metabolism, excretion)
- Drug release kinetics models (Higuchi, Korsmeyer-Peppas, zero-order)
- Loading capacities for scaffold-drug combinations
- Therapeutic windows and dosing information
- Drug stability in scaffold environments
- Controlled release formulation parameters

Data sources:
- DrugBank (https://go.drugbank.com/)
- FDA Drug Labels
- Published Q1 literature on drug-eluting scaffolds

Author: Dr. Demetrios Agourakis
"""
module DrugDeliveryLibrary

export DrugPharmacokinetics, DrugFormulation, ReleaseProfile
export DRUG_PK_DB, FORMULATION_DB, RELEASE_MODELS
export get_drug_pk, calculate_release, predict_plasma_concentration
export design_release_profile, optimize_loading
export THERAPEUTIC_CATEGORIES, get_drugs_by_category
export check_scaffold_drug_compatibility

# =============================================================================
# Pharmacokinetic Parameters Structure
# =============================================================================

"""
    DrugPharmacokinetics

Complete pharmacokinetic profile for a drug.
"""
struct DrugPharmacokinetics
    id::String                          # DrugBank/ChEBI ID
    name::String                        # Drug name
    therapeutic_class::String           # Category

    # ADME Parameters
    bioavailability_percent::Float64    # F (oral)
    volume_distribution_l_kg::Float64   # Vd
    protein_binding_percent::Float64    # Plasma protein binding
    half_life_h::Float64               # t1/2
    clearance_ml_min_kg::Float64       # CL

    # Absorption
    tmax_h::Float64                    # Time to peak concentration
    cmax_ng_ml::Float64                # Peak concentration (for standard dose)
    ka_h::Float64                      # Absorption rate constant

    # Distribution
    blood_brain_barrier::Bool          # Crosses BBB
    placental_transfer::Bool           # Crosses placenta
    milk_excretion::Bool              # Excreted in breast milk
    tissue_distribution::Dict{String,Float64}  # Tissue:plasma ratio

    # Metabolism
    primary_metabolism::String         # Main metabolic pathway
    cyp_enzymes::Vector{String}       # CYP450 enzymes involved
    active_metabolites::Vector{String} # Active metabolites
    metabolic_ratio::Float64          # Parent:metabolite ratio

    # Excretion
    renal_excretion_percent::Float64  # % excreted in urine
    fecal_excretion_percent::Float64  # % excreted in feces
    biliary_excretion::Bool           # Biliary elimination

    # Therapeutic parameters
    therapeutic_range_min::Float64     # Minimum effective concentration (ng/mL)
    therapeutic_range_max::Float64     # Maximum safe concentration (ng/mL)
    toxic_concentration::Float64       # Toxic threshold (ng/mL)
    standard_dose_mg::Float64         # Typical single dose
    dosing_frequency::String          # e.g., "q12h", "daily"

    # Local delivery parameters (scaffold-relevant)
    local_tissue_mic::Float64         # Minimum inhibitory concentration (for abx)
    local_therapeutic_dose_ug::Float64 # Effective local dose
    local_duration_days::Float64      # Duration of local effect needed
end

function DrugPharmacokinetics(id::String, name::String;
    therapeutic_class::String="",
    bioavailability_percent::Float64=NaN,
    volume_distribution_l_kg::Float64=NaN,
    protein_binding_percent::Float64=NaN,
    half_life_h::Float64=NaN,
    clearance_ml_min_kg::Float64=NaN,
    tmax_h::Float64=NaN,
    cmax_ng_ml::Float64=NaN,
    ka_h::Float64=NaN,
    blood_brain_barrier::Bool=false,
    placental_transfer::Bool=false,
    milk_excretion::Bool=false,
    tissue_distribution::Dict{String,Float64}=Dict{String,Float64}(),
    primary_metabolism::String="",
    cyp_enzymes::Vector{String}=String[],
    active_metabolites::Vector{String}=String[],
    metabolic_ratio::Float64=NaN,
    renal_excretion_percent::Float64=NaN,
    fecal_excretion_percent::Float64=NaN,
    biliary_excretion::Bool=false,
    therapeutic_range_min::Float64=NaN,
    therapeutic_range_max::Float64=NaN,
    toxic_concentration::Float64=NaN,
    standard_dose_mg::Float64=NaN,
    dosing_frequency::String="",
    local_tissue_mic::Float64=NaN,
    local_therapeutic_dose_ug::Float64=NaN,
    local_duration_days::Float64=NaN)

    DrugPharmacokinetics(id, name, therapeutic_class,
        bioavailability_percent, volume_distribution_l_kg, protein_binding_percent,
        half_life_h, clearance_ml_min_kg, tmax_h, cmax_ng_ml, ka_h,
        blood_brain_barrier, placental_transfer, milk_excretion, tissue_distribution,
        primary_metabolism, cyp_enzymes, active_metabolites, metabolic_ratio,
        renal_excretion_percent, fecal_excretion_percent, biliary_excretion,
        therapeutic_range_min, therapeutic_range_max, toxic_concentration,
        standard_dose_mg, dosing_frequency,
        local_tissue_mic, local_therapeutic_dose_ug, local_duration_days)
end

# =============================================================================
# Drug Release Models
# =============================================================================

"""
    ReleaseProfile

Drug release profile from scaffold.
"""
struct ReleaseProfile
    model::Symbol                      # :zero_order, :first_order, :higuchi, :korsmeyer_peppas
    parameters::Dict{Symbol,Float64}   # Model parameters
    r_squared::Float64                 # Goodness of fit
    release_mechanism::String          # Interpretation
end

"""Release kinetics model parameters."""
const RELEASE_MODELS = Dict{Symbol,NamedTuple}(
    :zero_order => (
        name = "Zero-order",
        equation = "Mt/M∞ = k₀t",
        parameters = [:k0],
        mechanism = "Constant release rate, ideal for controlled delivery"
    ),
    :first_order => (
        name = "First-order",
        equation = "Mt/M∞ = 1 - exp(-k₁t)",
        parameters = [:k1],
        mechanism = "Concentration-dependent release"
    ),
    :higuchi => (
        name = "Higuchi",
        equation = "Mt/M∞ = kH√t",
        parameters = [:kH],
        mechanism = "Diffusion-controlled release from matrix"
    ),
    :korsmeyer_peppas => (
        name = "Korsmeyer-Peppas",
        equation = "Mt/M∞ = kKP × t^n",
        parameters = [:kKP, :n],
        mechanism = "n<0.45: Fickian diffusion; 0.45<n<0.89: Anomalous; n>0.89: Case II transport"
    ),
    :hixson_crowell => (
        name = "Hixson-Crowell",
        equation = "∛W₀ - ∛Wt = kHC × t",
        parameters = [:kHC],
        mechanism = "Surface-erosion controlled"
    ),
    :weibull => (
        name = "Weibull",
        equation = "Mt/M∞ = 1 - exp(-(t/τ)^β)",
        parameters = [:tau, :beta],
        mechanism = "Empirical model for complex release"
    ),
    :baker_lonsdale => (
        name = "Baker-Lonsdale",
        equation = "3/2[1-(1-Mt/M∞)^(2/3)] - Mt/M∞ = kt",
        parameters = [:k],
        mechanism = "Diffusion from spherical matrix"
    )
)

"""
    DrugFormulation

Drug formulation for scaffold incorporation.
"""
struct DrugFormulation
    drug_id::String
    scaffold_material::String
    incorporation_method::Symbol  # :encapsulation, :surface_adsorption, :covalent, :physical_mixing

    # Loading parameters
    loading_percent::Float64      # Drug loading (w/w %)
    encapsulation_efficiency::Float64  # %
    particle_size_um::Float64     # For microparticles/nanoparticles

    # Release characteristics
    release_profile::ReleaseProfile
    burst_release_percent::Float64  # Initial burst
    t50_h::Float64                # Time to 50% release
    t90_h::Float64                # Time to 90% release
    total_release_percent::Float64  # Maximum release achieved

    # Stability
    shelf_life_months::Float64    # At recommended storage
    storage_temperature_c::Float64
    light_sensitive::Bool
    moisture_sensitive::Bool

    # Sterilization compatibility
    gamma_stable::Bool            # Gamma irradiation
    eto_stable::Bool              # Ethylene oxide
    autoclave_stable::Bool        # Steam sterilization
end

# =============================================================================
# Pharmacokinetics Database
# =============================================================================

const DRUG_PK_DB = Dict{String,DrugPharmacokinetics}(
    # =========================================================================
    # ANTIBIOTICS
    # =========================================================================

    "vancomycin" => DrugPharmacokinetics(
        "DB00512", "Vancomycin";
        therapeutic_class="Glycopeptide antibiotic",
        bioavailability_percent=0.0,  # IV only, negligible oral
        volume_distribution_l_kg=0.4,
        protein_binding_percent=55.0,
        half_life_h=6.0,
        clearance_ml_min_kg=1.3,
        blood_brain_barrier=false,
        primary_metabolism="Minimal",
        renal_excretion_percent=90.0,
        therapeutic_range_min=10000.0,  # 10 μg/mL trough
        therapeutic_range_max=20000.0,  # 20 μg/mL peak
        toxic_concentration=80000.0,
        standard_dose_mg=1000.0,
        dosing_frequency="q12h",
        local_tissue_mic=1000.0,  # 1 μg/mL for MRSA
        local_therapeutic_dose_ug=5000.0,
        local_duration_days=14.0
    ),

    "gentamicin" => DrugPharmacokinetics(
        "DB00798", "Gentamicin";
        therapeutic_class="Aminoglycoside antibiotic",
        bioavailability_percent=0.0,  # IV/IM only
        volume_distribution_l_kg=0.25,
        protein_binding_percent=10.0,
        half_life_h=2.5,
        clearance_ml_min_kg=1.2,
        blood_brain_barrier=false,
        primary_metabolism="None",
        renal_excretion_percent=95.0,
        therapeutic_range_min=5000.0,  # Peak
        therapeutic_range_max=10000.0,
        toxic_concentration=12000.0,  # Nephrotoxicity
        standard_dose_mg=5.0,  # mg/kg
        dosing_frequency="q8h or q24h",
        local_tissue_mic=500.0,  # 0.5 μg/mL
        local_therapeutic_dose_ug=2000.0,
        local_duration_days=7.0
    ),

    "ciprofloxacin" => DrugPharmacokinetics(
        "DB00537", "Ciprofloxacin";
        therapeutic_class="Fluoroquinolone antibiotic",
        bioavailability_percent=70.0,
        volume_distribution_l_kg=2.5,
        protein_binding_percent=30.0,
        half_life_h=4.0,
        clearance_ml_min_kg=8.0,
        tmax_h=1.5,
        cmax_ng_ml=2500.0,  # 500mg dose
        blood_brain_barrier=true,
        primary_metabolism="Hepatic (CYP1A2)",
        cyp_enzymes=["CYP1A2"],
        renal_excretion_percent=45.0,
        fecal_excretion_percent=25.0,
        therapeutic_range_min=1000.0,
        therapeutic_range_max=4000.0,
        standard_dose_mg=500.0,
        dosing_frequency="q12h",
        local_tissue_mic=100.0,
        local_therapeutic_dose_ug=1000.0,
        local_duration_days=10.0
    ),

    "rifampicin" => DrugPharmacokinetics(
        "DB01045", "Rifampicin";
        therapeutic_class="Rifamycin antibiotic",
        bioavailability_percent=95.0,
        volume_distribution_l_kg=0.7,
        protein_binding_percent=85.0,
        half_life_h=3.0,
        clearance_ml_min_kg=3.5,
        tmax_h=2.0,
        blood_brain_barrier=true,
        primary_metabolism="Hepatic (deacetylation)",
        biliary_excretion=true,
        renal_excretion_percent=15.0,
        fecal_excretion_percent=60.0,
        standard_dose_mg=600.0,
        dosing_frequency="daily",
        local_tissue_mic=4.0,  # Very low MIC for S. aureus
        local_therapeutic_dose_ug=500.0,
        local_duration_days=21.0
    ),

    # =========================================================================
    # ANTI-INFLAMMATORY
    # =========================================================================

    "dexamethasone" => DrugPharmacokinetics(
        "DB01234", "Dexamethasone";
        therapeutic_class="Corticosteroid",
        bioavailability_percent=80.0,
        volume_distribution_l_kg=2.0,
        protein_binding_percent=70.0,
        half_life_h=4.0,
        clearance_ml_min_kg=3.5,
        tmax_h=1.5,
        blood_brain_barrier=true,
        primary_metabolism="Hepatic (CYP3A4)",
        cyp_enzymes=["CYP3A4"],
        renal_excretion_percent=65.0,
        standard_dose_mg=4.0,
        dosing_frequency="daily or divided",
        local_therapeutic_dose_ug=100.0,
        local_duration_days=7.0
    ),

    "ibuprofen" => DrugPharmacokinetics(
        "DB01050", "Ibuprofen";
        therapeutic_class="NSAID",
        bioavailability_percent=80.0,
        volume_distribution_l_kg=0.12,
        protein_binding_percent=99.0,
        half_life_h=2.0,
        clearance_ml_min_kg=0.8,
        tmax_h=1.5,
        cmax_ng_ml=25000.0,  # 400mg dose
        primary_metabolism="Hepatic (CYP2C9)",
        cyp_enzymes=["CYP2C9"],
        renal_excretion_percent=90.0,
        therapeutic_range_min=10000.0,
        therapeutic_range_max=50000.0,
        standard_dose_mg=400.0,
        dosing_frequency="q6-8h",
        local_therapeutic_dose_ug=500.0,
        local_duration_days=3.0
    ),

    "indomethacin" => DrugPharmacokinetics(
        "DB00328", "Indomethacin";
        therapeutic_class="NSAID",
        bioavailability_percent=100.0,
        volume_distribution_l_kg=0.3,
        protein_binding_percent=99.0,
        half_life_h=4.5,
        clearance_ml_min_kg=1.2,
        tmax_h=2.0,
        blood_brain_barrier=true,
        primary_metabolism="Hepatic (glucuronidation)",
        renal_excretion_percent=60.0,
        fecal_excretion_percent=33.0,
        standard_dose_mg=50.0,
        dosing_frequency="q8h",
        local_therapeutic_dose_ug=200.0,
        local_duration_days=5.0
    ),

    # =========================================================================
    # OSTEOGENIC/BONE ACTIVE
    # =========================================================================

    "alendronate" => DrugPharmacokinetics(
        "DB00630", "Alendronate";
        therapeutic_class="Bisphosphonate",
        bioavailability_percent=0.7,  # Very low oral
        volume_distribution_l_kg=0.4,
        protein_binding_percent=78.0,
        half_life_h=720.0,  # 30 days (terminal)
        blood_brain_barrier=false,
        primary_metabolism="None",
        renal_excretion_percent=50.0,  # Rest binds to bone
        standard_dose_mg=70.0,
        dosing_frequency="weekly",
        local_therapeutic_dose_ug=100.0,
        local_duration_days=90.0  # Long-term bone effect
    ),

    "simvastatin" => DrugPharmacokinetics(
        "DB00641", "Simvastatin";
        therapeutic_class="HMG-CoA reductase inhibitor (osteogenic)",
        bioavailability_percent=5.0,  # Extensive first-pass
        volume_distribution_l_kg=1.0,
        protein_binding_percent=95.0,
        half_life_h=2.0,
        primary_metabolism="Hepatic (CYP3A4)",
        cyp_enzymes=["CYP3A4"],
        active_metabolites=["simvastatin acid"],
        standard_dose_mg=20.0,
        dosing_frequency="daily",
        local_therapeutic_dose_ug=50.0,  # For BMP-2 enhancement
        local_duration_days=28.0
    ),

    # =========================================================================
    # CHEMOTHERAPEUTICS
    # =========================================================================

    "doxorubicin" => DrugPharmacokinetics(
        "DB00997", "Doxorubicin";
        therapeutic_class="Anthracycline chemotherapy",
        bioavailability_percent=5.0,  # IV only in practice
        volume_distribution_l_kg=25.0,  # Very high tissue distribution
        protein_binding_percent=75.0,
        half_life_h=30.0,  # Terminal
        clearance_ml_min_kg=8.0,
        blood_brain_barrier=false,
        primary_metabolism="Hepatic (reduction)",
        active_metabolites=["doxorubicinol"],
        biliary_excretion=true,
        renal_excretion_percent=10.0,
        fecal_excretion_percent=40.0,
        therapeutic_range_min=50.0,
        therapeutic_range_max=500.0,
        toxic_concentration=1000.0,  # Cardiotoxicity risk
        standard_dose_mg=60.0,  # mg/m²
        dosing_frequency="q3weeks",
        local_therapeutic_dose_ug=1000.0,
        local_duration_days=21.0
    ),

    "methotrexate" => DrugPharmacokinetics(
        "DB00563", "Methotrexate";
        therapeutic_class="Antimetabolite",
        bioavailability_percent=60.0,
        volume_distribution_l_kg=0.5,
        protein_binding_percent=50.0,
        half_life_h=8.0,  # Dose-dependent
        clearance_ml_min_kg=2.0,
        tmax_h=2.0,
        blood_brain_barrier=true,  # At high doses
        primary_metabolism="Intracellular polyglutamation",
        renal_excretion_percent=90.0,
        therapeutic_range_min=10.0,
        therapeutic_range_max=1000.0,
        standard_dose_mg=15.0,  # RA dose
        dosing_frequency="weekly",
        local_therapeutic_dose_ug=500.0,
        local_duration_days=14.0
    ),

    "paclitaxel" => DrugPharmacokinetics(
        "DB01229", "Paclitaxel";
        therapeutic_class="Taxane chemotherapy",
        bioavailability_percent=0.0,  # IV only
        volume_distribution_l_kg=50.0,  # Very high
        protein_binding_percent=95.0,
        half_life_h=20.0,
        clearance_ml_min_kg=6.0,
        blood_brain_barrier=false,
        primary_metabolism="Hepatic (CYP2C8, CYP3A4)",
        cyp_enzymes=["CYP2C8", "CYP3A4"],
        biliary_excretion=true,
        fecal_excretion_percent=70.0,
        standard_dose_mg=175.0,  # mg/m²
        dosing_frequency="q3weeks",
        local_therapeutic_dose_ug=500.0,  # Anti-restenosis
        local_duration_days=30.0
    ),

    # =========================================================================
    # LOCAL ANESTHETICS
    # =========================================================================

    "lidocaine" => DrugPharmacokinetics(
        "DB00281", "Lidocaine";
        therapeutic_class="Local anesthetic",
        bioavailability_percent=35.0,
        volume_distribution_l_kg=1.3,
        protein_binding_percent=70.0,
        half_life_h=1.8,
        clearance_ml_min_kg=10.0,
        blood_brain_barrier=true,
        primary_metabolism="Hepatic (CYP3A4, CYP1A2)",
        cyp_enzymes=["CYP3A4", "CYP1A2"],
        active_metabolites=["MEGX", "GX"],
        therapeutic_range_min=1500.0,
        therapeutic_range_max=5000.0,
        toxic_concentration=6000.0,  # CNS toxicity
        standard_dose_mg=300.0,  # Maximum local infiltration
        local_therapeutic_dose_ug=2000.0,
        local_duration_days=0.5  # 12 hours
    ),

    "bupivacaine" => DrugPharmacokinetics(
        "DB00297", "Bupivacaine";
        therapeutic_class="Local anesthetic",
        bioavailability_percent=0.0,  # Local/regional
        volume_distribution_l_kg=1.0,
        protein_binding_percent=95.0,
        half_life_h=3.5,
        clearance_ml_min_kg=0.5,
        blood_brain_barrier=true,
        primary_metabolism="Hepatic (CYP3A4)",
        cyp_enzymes=["CYP3A4"],
        therapeutic_range_min=1000.0,
        therapeutic_range_max=3000.0,
        toxic_concentration=4000.0,  # Cardiotoxicity
        standard_dose_mg=175.0,  # Maximum single dose
        local_therapeutic_dose_ug=500.0,
        local_duration_days=1.0
    ),

    # =========================================================================
    # GROWTH FACTORS
    # =========================================================================

    "rhBMP2" => DrugPharmacokinetics(
        "rhBMP2", "Recombinant human BMP-2";
        therapeutic_class="Osteogenic growth factor",
        volume_distribution_l_kg=0.1,  # Local retention
        half_life_h=24.0,  # Local tissue
        primary_metabolism="Proteolytic degradation",
        local_therapeutic_dose_ug=1500.0,  # Per level (spine)
        local_duration_days=14.0
    ),

    "VEGF" => DrugPharmacokinetics(
        "VEGF165", "VEGF-165";
        therapeutic_class="Angiogenic growth factor",
        half_life_h=0.5,  # 30 min in circulation
        primary_metabolism="Proteolytic degradation",
        local_therapeutic_dose_ug=100.0,
        local_duration_days=7.0
    )
)

# =============================================================================
# Therapeutic Categories
# =============================================================================

const THERAPEUTIC_CATEGORIES = Dict{Symbol,Vector{String}}(
    :antibiotics => ["vancomycin", "gentamicin", "ciprofloxacin", "rifampicin"],
    :anti_inflammatory => ["dexamethasone", "ibuprofen", "indomethacin"],
    :osteogenic => ["alendronate", "simvastatin", "rhBMP2"],
    :chemotherapy => ["doxorubicin", "methotrexate", "paclitaxel"],
    :anesthetic => ["lidocaine", "bupivacaine"],
    :angiogenic => ["VEGF"]
)

# =============================================================================
# Release Kinetics Calculations
# =============================================================================

"""
    calculate_release(model, t, params)

Calculate fractional drug release at time t using specified model.
"""
function calculate_release(model::Symbol, t::Float64, params::Dict{Symbol,Float64})
    if model == :zero_order
        k0 = params[:k0]
        return min(k0 * t, 1.0)

    elseif model == :first_order
        k1 = params[:k1]
        return 1.0 - exp(-k1 * t)

    elseif model == :higuchi
        kH = params[:kH]
        return min(kH * sqrt(t), 1.0)

    elseif model == :korsmeyer_peppas
        kKP = params[:kKP]
        n = params[:n]
        return min(kKP * t^n, 1.0)

    elseif model == :weibull
        tau = params[:tau]
        beta = params[:beta]
        return 1.0 - exp(-(t/tau)^beta)

    else
        error("Unknown release model: $model")
    end
end

"""
    fit_release_model(time_data, release_data, model)

Fit release kinetics model to experimental data.
"""
function fit_release_model(time_data::Vector{Float64},
                          release_data::Vector{Float64},
                          model::Symbol)
    n = length(time_data)

    if model == :zero_order
        # Linear regression: Mt/M∞ = k0 * t
        k0 = sum(release_data .* time_data) / sum(time_data.^2)
        predicted = k0 .* time_data
        ss_res = sum((release_data .- predicted).^2)
        ss_tot = sum((release_data .- mean(release_data)).^2)
        r2 = 1 - ss_res/ss_tot

        return ReleaseProfile(model, Dict(:k0 => k0), r2,
            "Constant release rate")

    elseif model == :higuchi
        # Linear in sqrt(t): Mt/M∞ = kH * √t
        sqrt_t = sqrt.(time_data)
        kH = sum(release_data .* sqrt_t) / sum(sqrt_t.^2)
        predicted = kH .* sqrt_t
        ss_res = sum((release_data .- predicted).^2)
        ss_tot = sum((release_data .- mean(release_data)).^2)
        r2 = 1 - ss_res/ss_tot

        return ReleaseProfile(model, Dict(:kH => kH), r2,
            "Diffusion-controlled from matrix")

    elseif model == :korsmeyer_peppas
        # Log-linear: log(Mt/M∞) = log(kKP) + n*log(t)
        # Only use data up to 60% release
        valid = release_data .< 0.6
        if sum(valid) < 3
            valid = trues(n)
        end

        log_t = log.(time_data[valid])
        log_release = log.(release_data[valid])

        # Linear regression in log space
        n_pts = sum(valid)
        sum_x = sum(log_t)
        sum_y = sum(log_release)
        sum_xy = sum(log_t .* log_release)
        sum_xx = sum(log_t.^2)

        slope = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_xx - sum_x^2)
        intercept = (sum_y - slope * sum_x) / n_pts

        n_exp = slope
        kKP = exp(intercept)

        predicted = kKP .* time_data.^n_exp
        predicted = min.(predicted, 1.0)
        ss_res = sum((release_data .- predicted).^2)
        ss_tot = sum((release_data .- mean(release_data)).^2)
        r2 = 1 - ss_res/ss_tot

        # Interpret n value
        mechanism = if n_exp < 0.45
            "Fickian diffusion"
        elseif n_exp < 0.89
            "Anomalous transport (diffusion + erosion)"
        else
            "Case II transport (erosion-controlled)"
        end

        return ReleaseProfile(model, Dict(:kKP => kKP, :n => n_exp), r2, mechanism)
    else
        error("Model fitting not implemented for: $model")
    end
end

"""
    design_release_profile(drug_id, target_duration_h, release_type)

Design optimal release parameters for therapeutic effect.
"""
function design_release_profile(drug_id::String,
                                target_duration_h::Float64,
                                release_type::Symbol=:sustained)
    pk = get(DRUG_PK_DB, drug_id, nothing)
    if isnothing(pk)
        error("Drug $drug_id not found in database")
    end

    # Calculate target release rate
    local_dose = pk.local_therapeutic_dose_ug

    if release_type == :sustained
        # Zero-order for sustained
        k0 = 1.0 / target_duration_h
        return (
            model = :zero_order,
            parameters = Dict(:k0 => k0),
            loading_ug = local_dose * 1.5,  # 50% excess for burst
            predicted_duration_h = target_duration_h,
            therapeutic_concentration = local_dose / target_duration_h
        )

    elseif release_type == :pulsatile
        # Multiple bursts
        return (
            model = :pulsatile,
            parameters = Dict(:pulses => 3, :interval_h => target_duration_h/3),
            loading_ug = local_dose * 3.0,
            predicted_duration_h = target_duration_h
        )

    elseif release_type == :burst_then_sustained
        # Initial burst followed by sustained
        return (
            model = :korsmeyer_peppas,
            parameters = Dict(:kKP => 0.3, :n => 0.45),
            loading_ug = local_dose * 2.0,
            predicted_duration_h = target_duration_h,
            burst_percent = 30.0
        )
    else
        error("Unknown release type: $release_type")
    end
end

# =============================================================================
# Plasma Concentration Prediction
# =============================================================================

"""
    predict_plasma_concentration(drug_id, dose_mg, time_h; route=:iv)

Predict plasma concentration using one-compartment PK model.
"""
function predict_plasma_concentration(drug_id::String,
                                      dose_mg::Float64,
                                      time_h::Float64;
                                      route::Symbol=:iv)
    pk = get(DRUG_PK_DB, drug_id, nothing)
    if isnothing(pk)
        error("Drug $drug_id not found")
    end

    # Convert to consistent units
    Vd = pk.volume_distribution_l_kg * 70  # Assume 70 kg
    ke = 0.693 / pk.half_life_h  # Elimination rate constant

    if route == :iv
        # IV bolus: C = (D/Vd) * exp(-ke*t)
        C0 = (dose_mg * 1000) / Vd  # ng/mL
        return C0 * exp(-ke * time_h)

    elseif route == :oral
        F = pk.bioavailability_percent / 100
        ka = pk.ka_h > 0 ? pk.ka_h : 1.0  # Default absorption rate

        # One-compartment with first-order absorption
        C = (F * dose_mg * 1000 * ka) / (Vd * (ka - ke)) *
            (exp(-ke * time_h) - exp(-ka * time_h))
        return max(C, 0.0)

    elseif route == :local
        # Local release (simplified)
        # Assumes drug enters systemic circulation slowly
        local_release_rate = dose_mg * 1000 / 24  # ng/h
        C_ss = local_release_rate / (pk.clearance_ml_min_kg * 70 * 60 / 1000)
        return C_ss * (1 - exp(-ke * time_h))
    end
end

"""
    check_therapeutic_window(drug_id, concentration)

Check if concentration is within therapeutic window.
"""
function check_therapeutic_window(drug_id::String, concentration::Float64)
    pk = get(DRUG_PK_DB, drug_id, nothing)
    if isnothing(pk)
        return (status = :unknown, message = "Drug not found")
    end

    if concentration < pk.therapeutic_range_min
        return (status = :subtherapeutic,
                message = "Below minimum effective concentration")
    elseif concentration > pk.therapeutic_range_max
        if concentration > pk.toxic_concentration
            return (status = :toxic,
                    message = "Above toxic threshold!")
        else
            return (status = :supratherapeutic,
                    message = "Above optimal but below toxic")
        end
    else
        return (status = :therapeutic,
                message = "Within therapeutic window")
    end
end

# =============================================================================
# Scaffold-Drug Compatibility
# =============================================================================

"""
    check_scaffold_drug_compatibility(material, drug_id)

Assess compatibility of drug with scaffold material.
"""
function check_scaffold_drug_compatibility(material::String, drug_id::String)
    compatibility_issues = String[]
    recommendations = String[]

    pk = get(DRUG_PK_DB, drug_id, nothing)
    if isnothing(pk)
        return (compatible = false, issues = ["Drug not found"], recommendations = [])
    end

    # Check pH stability for acidic degradation products
    acidic_materials = ["PLA", "PLGA", "PGA"]
    basic_drugs = ["gentamicin", "ciprofloxacin", "lidocaine", "bupivacaine"]

    if material in acidic_materials
        if drug_id in basic_drugs
            push!(compatibility_issues,
                "Acidic degradation products may affect drug stability")
            push!(recommendations,
                "Consider buffering agents or coating")
        end
    end

    # Check for hydrolysis-sensitive drugs with hydrogels
    hydrogel_materials = ["Alginate", "GelMA", "Hyaluronic_Acid", "Collagen"]
    hydrolysis_sensitive = ["paclitaxel", "doxorubicin"]

    if material in hydrogel_materials
        if drug_id in hydrolysis_sensitive
            push!(compatibility_issues,
                "High water content may cause drug hydrolysis")
            push!(recommendations,
                "Use protective nanoparticle encapsulation")
        end
    end

    # Check protein drugs with crosslinking
    protein_drugs = ["rhBMP2", "VEGF"]
    crosslinked_materials = ["GelMA", "Collagen"]

    if material in crosslinked_materials
        if drug_id in protein_drugs
            push!(compatibility_issues,
                "Crosslinking may denature protein drugs")
            push!(recommendations,
                "Add drug after crosslinking or use gentle methods")
        end
    end

    # Temperature-sensitive drugs and processing
    heat_sensitive = ["rhBMP2", "VEGF", "vancomycin"]
    high_temp_processing = ["PCL", "PLA", "PLGA"]  # Melt processing

    if material in high_temp_processing
        if drug_id in heat_sensitive
            push!(compatibility_issues,
                "Melt processing temperatures may degrade drug")
            push!(recommendations,
                "Use solution/emulsion methods or post-loading")
        end
    end

    compatible = isempty(compatibility_issues)

    return (
        compatible = compatible,
        issues = compatibility_issues,
        recommendations = recommendations
    )
end

# =============================================================================
# Lookup Functions
# =============================================================================

"""Get drug pharmacokinetics by ID or name."""
function get_drug_pk(id::String)
    pk = get(DRUG_PK_DB, id, nothing)
    if !isnothing(pk)
        return pk
    end

    # Try case-insensitive match
    id_lower = lowercase(id)
    for (key, pk) in DRUG_PK_DB
        if lowercase(pk.name) == id_lower || lowercase(key) == id_lower
            return pk
        end
    end

    return nothing
end

"""Get all drugs in a therapeutic category."""
function get_drugs_by_category(category::Symbol)
    drug_ids = get(THERAPEUTIC_CATEGORIES, category, String[])
    return [get_drug_pk(id) for id in drug_ids if !isnothing(get_drug_pk(id))]
end

"""Calculate optimal drug loading for target duration."""
function optimize_loading(drug_id::String,
                          target_duration_h::Float64,
                          scaffold_mass_mg::Float64;
                          max_loading_percent::Float64=30.0)
    pk = get(DRUG_PK_DB, drug_id, nothing)
    if isnothing(pk)
        error("Drug not found: $drug_id")
    end

    # Calculate required drug amount
    required_dose = pk.local_therapeutic_dose_ug

    # Account for incomplete release (typically 80%)
    release_efficiency = 0.8
    total_drug_needed = required_dose / release_efficiency

    # Calculate loading percentage
    loading_percent = (total_drug_needed / 1000) / scaffold_mass_mg * 100

    if loading_percent > max_loading_percent
        return (
            feasible = false,
            loading_percent = loading_percent,
            message = "Required loading ($loading_percent%) exceeds maximum ($max_loading_percent%)",
            alternative = "Consider larger scaffold or multiple scaffolds"
        )
    else
        return (
            feasible = true,
            loading_percent = loading_percent,
            drug_amount_ug = total_drug_needed,
            expected_duration_h = target_duration_h
        )
    end
end

end # module
