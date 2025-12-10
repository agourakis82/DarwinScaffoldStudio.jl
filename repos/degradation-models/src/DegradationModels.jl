"""
    DegradationModels

Micro-repository for polymer degradation modeling.

MODELS AVAILABLE:
=================

1. NeuralModel - Neural network with material embeddings (95.5% accuracy)
2. BronstedModel - Brønsted-Lowry acid catalysis + VFT
3. ThermodynamicModel - First principles (Eyring + Fick)
4. ConservativeModel - Empirical with saturating autocatalysis
5. HybridPINN - Physics-Informed Neural Network

QUICK START:
============

```julia
using DegradationModels

# Train the neural model (best accuracy)
model = train_neural(epochs=3000)

# Predict Mn at time t
Mn = predict(model, material="PLDLA", Mn0=51.3, t=30.0)

# Validate against experimental data
results = validate(model)
```

Author: Darwin Scaffold Studio
Version: 0.1.0
"""
module DegradationModels

# Core exports
export AbstractDegradationModel
export NeuralModel, BronstedModel, ThermodynamicModel
export ConservativeModel, HybridPINN, UniversalModel, UniversalModelV2, UniversalModelV3, UniversalModelV4
export PLDLA3DPrintModel, PLDLA3DPrintModelV2, PLDLAHybridModel, PLDLANeuralODE, PLDLANeuralODEFast

# Functions
export train, predict, validate
export train_neural, train_bronsted, train_thermodynamic, train_universal
export estimate_halflife, predict_degradation_curve
export compare_conditions, list_body_regions, predict_with_uncertainty
export BODY_TEMPERATURES
export compare_models, extract_parameters

# Data
export EXPERIMENTAL_DATA, MATERIAL_NAMES

# Re-export Printf for convenience
export @printf, @sprintf

using Statistics
using Printf
using Random
using LinearAlgebra

# =============================================================================
# ABSTRACT TYPE
# =============================================================================

abstract type AbstractDegradationModel end

# =============================================================================
# EXPERIMENTAL DATA
# =============================================================================

const EXPERIMENTAL_DATA = Dict(
    "Kaique_PLDLA" => (
        Mn0=51.3,
        times=[0.0, 30.0, 60.0, 90.0],
        Mn=[51.3, 25.4, 18.3, 7.9],
        T=310.15,
        pH=7.4,
        TEC=0.0,
        condition=:in_vitro
    ),
    "Kaique_TEC1" => (
        Mn0=45.0,
        times=[0.0, 30.0, 60.0, 90.0],
        Mn=[45.0, 19.3, 11.7, 8.1],
        T=310.15,
        pH=7.4,
        TEC=1.0,
        condition=:in_vitro
    ),
    "Kaique_TEC2" => (
        Mn0=32.7,
        times=[0.0, 30.0, 60.0, 90.0],
        Mn=[32.7, 15.0, 12.6, 6.6],
        T=310.15,
        pH=7.4,
        TEC=2.0,
        condition=:in_vitro
    ),
    "InVivo_Subcutaneous" => (
        Mn0=99.0,
        times=[0.0, 28.0, 56.0],
        Mn=[99.0, 92.0, 85.0],
        T=310.15,
        pH=7.35,
        TEC=0.0,
        condition=:in_vivo
    )
)

const MATERIAL_NAMES = Dict(
    1 => "Kaique_PLDLA",
    2 => "Kaique_TEC1",
    3 => "Kaique_TEC2",
    4 => "InVivo_Subcutaneous"
)

const MATERIAL_IDS = Dict(v => k for (k, v) in MATERIAL_NAMES)

# =============================================================================
# INCLUDE MODELS
# =============================================================================

include("models/NeuralModel.jl")
include("models/BronstedModel.jl")
include("models/ThermodynamicModel.jl")
include("models/ConservativeModel.jl")
include("models/HybridPINN.jl")
include("models/UniversalModel.jl")
include("models/UniversalModelV2.jl")
include("models/UniversalModelV3.jl")
include("models/UniversalModelV4.jl")
include("models/PLDLA3DPrintModel.jl")
include("models/PLDLA3DPrintModelV2.jl")
include("models/PLDLAHybridModel.jl")
include("models/PLDLANeuralODE.jl")
include("models/PLDLANeuralODEFast.jl")

# =============================================================================
# UNIFIED API
# =============================================================================

"""
    train(ModelType; kwargs...)

Train a degradation model.

# Examples
```julia
model = train(NeuralModel, epochs=3000)
model = train(BronstedModel)
model = train(ThermodynamicModel)
```
"""
function train end

"""
    predict(model, material, Mn0, t; T=310.15, pH=7.4, TEC=0.0)

Predict molecular weight at time t.

# Arguments
- `model`: Trained model
- `material`: Material name or ID
- `Mn0`: Initial molecular weight (kg/mol)
- `t`: Time (days)
- `T`: Temperature (K), default 310.15 (37°C)
- `pH`: pH of medium, default 7.4
- `TEC`: Plasticizer content (%), default 0.0

# Returns
- Predicted Mn (kg/mol)
"""
function predict end

"""
    validate(model; datasets=keys(EXPERIMENTAL_DATA))

Validate model against experimental data.

# Returns
- Dict with MAPE for each dataset
"""
function validate(model::AbstractDegradationModel;
    datasets=keys(EXPERIMENTAL_DATA))
    results = Dict{String,Float64}()

    for name in datasets
        if !haskey(EXPERIMENTAL_DATA, name)
            continue
        end

        data = EXPERIMENTAL_DATA[name]
        errors = Float64[]

        for (i, t) in enumerate(data.times)
            if t == 0.0
                continue
            end

            Mn_pred = predict(model, name, data.Mn0, t,
                T=data.T, pH=data.pH, TEC=data.TEC)
            err = abs(Mn_pred - data.Mn[i]) / data.Mn[i] * 100
            push!(errors, err)
        end

        results[name] = isempty(errors) ? 0.0 : mean(errors)
    end

    return results
end

"""
    compare_models(models::Vector; verbose=true)

Compare multiple models on experimental data.
"""
function compare_models(models::Vector{<:AbstractDegradationModel};
    verbose::Bool=true)
    results = Dict{String,Dict{String,Float64}}()

    for model in models
        name = string(typeof(model))
        results[name] = validate(model)
    end

    if verbose
        println("\n" * "="^70)
        println("  MODEL COMPARISON")
        println("="^70)

        # Header
        print("\n  Dataset              ")
        for name in keys(results)
            @printf("│ %12s ", name[1:min(12, length(name))])
        end
        println()
        println("  " * "-"^22 * repeat("┼" * "-"^14, length(results)))

        # Data rows
        for dataset in keys(EXPERIMENTAL_DATA)
            @printf("  %-20s ", dataset)
            for (_, res) in results
                mape = get(res, dataset, NaN)
                @printf("│ %10.1f%% ", mape)
            end
            println()
        end

        # Global
        println("  " * "-"^22 * repeat("┼" * "-"^14, length(results)))
        @printf("  %-20s ", "GLOBAL")
        for (_, res) in results
            global_mape = mean(values(res))
            @printf("│ %10.1f%% ", global_mape)
        end
        println()
    end

    return results
end

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

"""
    train_neural(; epochs=3000, verbose=true)

Train the neural model (highest accuracy).
"""
train_neural(; kwargs...) = train(NeuralModel; kwargs...)

"""
    train_bronsted(; kwargs...)

Train the Brønsted-Lowry model.
"""
train_bronsted(; kwargs...) = train(BronstedModel; kwargs...)

"""
    train_thermodynamic(; kwargs...)

Train/initialize the thermodynamic model.
"""
train_thermodynamic(; kwargs...) = train(ThermodynamicModel; kwargs...)

end # module
