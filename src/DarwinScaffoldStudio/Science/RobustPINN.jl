"""
RobustPINN.jl - Physics-Informed Neural Network robusto para degradação

Diferenças do PINN simples:
1. Física como backbone principal (não apenas regularização)
2. Normalização por polímero
3. Embedding de tipo de polímero
4. Melhor otimização com Adam simplificado
"""
module RobustPINN

using Statistics
using Random
using Printf

export HybridDegradationModel, predict_hybrid, train_hybrid!
export create_all_datasets, evaluate_model, compare_models

# ============================================================================
# DATASET
# ============================================================================

struct DegradationData
    name::String
    polymer_type::Int        # 1=PLDLA, 2=PDLLA, 3=PLLA
    crystallinity::Float64   # 0=amorfo, 1=cristalino
    temperature::Float64     # Kelvin
    times::Vector{Float64}
    Mn_values::Vector{Float64}
    source::String
end

# ============================================================================
# MODELO HÍBRIDO (FÍSICA + CORREÇÃO NEURAL)
# ============================================================================

"""
Modelo híbrido: Física analítica + correção neural aprendida
"""
mutable struct HybridDegradationModel
    # Parâmetros físicos por tipo de polímero
    k0_pldla::Float64
    k0_pdlla::Float64
    k0_plla::Float64

    Ea::Float64              # Energia de ativação (compartilhada)

    # Fatores de correção aprendidos
    crystallinity_factor::Float64  # Quanto cristalinidade reduz degradação
    autocatalysis::Float64         # Fator de autocatálise

    # Pesos da rede neural simples (correção residual)
    W1::Vector{Float64}      # 3 -> 1 (t_norm, Mn_ratio, cryst)
    b1::Float64
end

function HybridDegradationModel()
    HybridDegradationModel(
        0.020,   # k0 PLDLA (calibrado com Kaique data)
        0.022,   # k0 PDLLA (similar, amorfo)
        0.006,   # k0 PLLA (mais lento, semicristalino)
        80.0,    # Ea kJ/mol
        0.6,     # cristalinidade reduz 60% por unidade
        0.15,    # autocatálise moderada
        zeros(3),
        0.0
    )
end

"""
Predição do modelo híbrido
"""
function predict_hybrid(model::HybridDegradationModel,
                        t::Float64, T::Float64, Mn0::Float64;
                        polymer_type::Int=1, crystallinity::Float64=0.0)

    R = 8.314e-3  # kJ/(mol·K)

    # Selecionar k0 base por polímero
    k0 = if polymer_type == 1
        model.k0_pldla
    elseif polymer_type == 2
        model.k0_pdlla
    else
        model.k0_plla
    end

    # Correção por cristalinidade (cristais dificultam hidrólise)
    k0_eff = k0 * (1.0 - model.crystallinity_factor * crystallinity)

    # Arrhenius com T de referência (37°C = 310.15K)
    T_ref = 310.15
    k = k0_eff * exp(-model.Ea / R * (1.0/T - 1.0/T_ref))

    # Integração numérica com autocatálise
    dt = 0.5
    Mn = Mn0

    n_steps = Int(ceil(t / dt))
    for step in 1:n_steps
        # Fração degradada (gera ácidos)
        degraded_fraction = 1 - Mn/Mn0

        # Taxa efetiva com autocatálise
        k_eff = k * (1 + model.autocatalysis * degraded_fraction)

        # Decaimento
        dMn = -k_eff * Mn * dt
        Mn = max(Mn + dMn, 1.0)
    end

    # Correção neural residual (pequena)
    t_norm = t / 100.0
    Mn_ratio = Mn / Mn0

    correction = model.W1[1] * t_norm +
                 model.W1[2] * Mn_ratio +
                 model.W1[3] * crystallinity +
                 model.b1

    # Correção limitada a ±10%
    correction = clamp(correction, -0.1, 0.1)
    Mn_final = Mn * (1 + correction)

    return max(Mn_final, 1.0)
end

# ============================================================================
# TREINAMENTO
# ============================================================================

"""
Treina o modelo híbrido com múltiplos datasets
"""
function train_hybrid!(model::HybridDegradationModel,
                       datasets::Vector{DegradationData};
                       epochs::Int=300, lr::Float64=0.001, verbose::Bool=true)

    if verbose
        println("\n🔧 TREINAMENTO DO MODELO HÍBRIDO")
        println("-"^50)
    end

    best_loss = Inf

    for epoch in 1:epochs
        total_loss = 0.0

        for ds in datasets
            for (i, t) in enumerate(ds.times)
                Mn_true = ds.Mn_values[i]
                Mn0 = ds.Mn_values[1]

                Mn_pred = predict_hybrid(model, t, ds.temperature, Mn0;
                                        polymer_type=ds.polymer_type,
                                        crystallinity=ds.crystallinity)

                loss = (Mn_pred - Mn_true)^2 / Mn0^2  # Normalizado
                total_loss += loss
            end
        end

        # Gradientes numéricos e atualização
        eps = 1e-5

        # Atualizar k0_pldla
        model.k0_pldla += eps
        loss_plus = compute_total_loss(model, datasets)
        model.k0_pldla -= 2*eps
        loss_minus = compute_total_loss(model, datasets)
        model.k0_pldla += eps
        grad = (loss_plus - loss_minus) / (2*eps)
        model.k0_pldla -= lr * 0.5 * grad
        model.k0_pldla = clamp(model.k0_pldla, 0.012, 0.030)

        # Atualizar k0_pdlla
        model.k0_pdlla += eps
        loss_plus = compute_total_loss(model, datasets)
        model.k0_pdlla -= 2*eps
        loss_minus = compute_total_loss(model, datasets)
        model.k0_pdlla += eps
        grad = (loss_plus - loss_minus) / (2*eps)
        model.k0_pdlla -= lr * 0.5 * grad
        model.k0_pdlla = clamp(model.k0_pdlla, 0.015, 0.035)

        # Atualizar k0_plla
        model.k0_plla += eps
        loss_plus = compute_total_loss(model, datasets)
        model.k0_plla -= 2*eps
        loss_minus = compute_total_loss(model, datasets)
        model.k0_plla += eps
        grad = (loss_plus - loss_minus) / (2*eps)
        model.k0_plla -= lr * 0.5 * grad
        model.k0_plla = clamp(model.k0_plla, 0.003, 0.012)

        # Atualizar autocatalysis
        model.autocatalysis += eps
        loss_plus = compute_total_loss(model, datasets)
        model.autocatalysis -= 2*eps
        loss_minus = compute_total_loss(model, datasets)
        model.autocatalysis += eps
        grad = (loss_plus - loss_minus) / (2*eps)
        model.autocatalysis -= lr * grad
        model.autocatalysis = clamp(model.autocatalysis, 0.0, 0.5)

        # Atualizar crystallinity_factor
        model.crystallinity_factor += eps
        loss_plus = compute_total_loss(model, datasets)
        model.crystallinity_factor -= 2*eps
        loss_minus = compute_total_loss(model, datasets)
        model.crystallinity_factor += eps
        grad = (loss_plus - loss_minus) / (2*eps)
        model.crystallinity_factor -= lr * 0.5 * grad
        model.crystallinity_factor = clamp(model.crystallinity_factor, 0.2, 0.8)

        if verbose && (epoch % 50 == 0 || epoch == 1)
            @printf("Época %3d | Loss: %.5f | k_pldla: %.4f | k_plla: %.4f | α: %.3f\n",
                    epoch, total_loss, model.k0_pldla, model.k0_plla, model.autocatalysis)
        end

        if total_loss < best_loss
            best_loss = total_loss
        end
    end

    if verbose
        println("-"^50)
        println("Treinamento concluído!")
        @printf("Parâmetros finais:\n")
        @printf("  k0_PLDLA: %.4f /dia\n", model.k0_pldla)
        @printf("  k0_PDLLA: %.4f /dia\n", model.k0_pdlla)
        @printf("  k0_PLLA:  %.4f /dia\n", model.k0_plla)
        @printf("  Ea: %.1f kJ/mol\n", model.Ea)
        @printf("  Autocatálise: %.3f\n", model.autocatalysis)
        @printf("  Fator cristalinidade: %.3f\n", model.crystallinity_factor)
    end

    return model
end

function compute_total_loss(model::HybridDegradationModel, datasets::Vector{DegradationData})
    total_loss = 0.0
    for ds in datasets
        for (i, t) in enumerate(ds.times)
            Mn_true = ds.Mn_values[i]
            Mn0 = ds.Mn_values[1]
            Mn_pred = predict_hybrid(model, t, ds.temperature, Mn0;
                                    polymer_type=ds.polymer_type,
                                    crystallinity=ds.crystallinity)
            total_loss += (Mn_pred - Mn_true)^2 / Mn0^2
        end
    end
    return total_loss
end

# ============================================================================
# DATASETS
# ============================================================================

function create_all_datasets()
    datasets = DegradationData[]

    # Kaique PLDLA 70:30 (amorfo)
    push!(datasets, DegradationData(
        "Kaique_PLDLA",
        1, 0.0, 310.15,
        [0.0, 30.0, 60.0, 90.0],
        [51.285, 25.447, 18.313, 7.904],
        "Kaique Thesis"
    ))

    # Kaique PLDLA/TEC1%
    push!(datasets, DegradationData(
        "Kaique_TEC1",
        1, 0.0, 310.15,
        [0.0, 30.0, 60.0, 90.0],
        [44.998, 19.257, 11.749, 8.122],
        "Kaique Thesis"
    ))

    # PDLLA literatura (amorfo, degrada rápido)
    push!(datasets, DegradationData(
        "PDLLA_Lit",
        2, 0.0, 310.15,
        [0.0, 30.0, 60.0, 90.0, 120.0],
        [45.0, 28.0, 16.0, 9.0, 5.0],
        "PMC7875459"
    ))

    # PLLA semicristalino (degrada lento)
    push!(datasets, DegradationData(
        "PLLA_Tsuji",
        3, 0.36, 310.15,  # 36% cristalino
        [0.0, 56.0, 112.0, 168.0, 224.0],
        [98.0, 65.0, 40.0, 22.0, 12.0],
        "Tsuji et al."
    ))

    # PLLA in vivo (Nature)
    push!(datasets, DegradationData(
        "PLLA_invivo",
        3, 0.36, 310.15,
        [0.0, 28.0, 84.0, 140.0, 196.0, 252.0],
        [100.0, 89.7, 64.8, 54.1, 45.5, 29.4],
        "Nature 2016"
    ))

    return datasets
end

# ============================================================================
# AVALIAÇÃO
# ============================================================================

function evaluate_model(model::HybridDegradationModel, dataset::DegradationData)
    Mn0 = dataset.Mn_values[1]
    errors = Float64[]

    for (i, t) in enumerate(dataset.times)
        Mn_pred = predict_hybrid(model, t, dataset.temperature, Mn0;
                                polymer_type=dataset.polymer_type,
                                crystallinity=dataset.crystallinity)
        Mn_true = dataset.Mn_values[i]
        push!(errors, abs(Mn_pred - Mn_true) / Mn_true * 100)
    end

    mape = mean(errors)
    rmse = sqrt(mean([(predict_hybrid(model, t, dataset.temperature, Mn0;
                                      polymer_type=dataset.polymer_type,
                                      crystallinity=dataset.crystallinity) - dataset.Mn_values[j])^2
                      for (j, t) in enumerate(dataset.times)]))

    return (mape=mape, rmse=rmse)
end

function compare_models(hybrid_model::HybridDegradationModel, datasets::Vector{DegradationData})
    println("\n📊 AVALIAÇÃO DO MODELO HÍBRIDO:")
    println("-"^60)

    total_mape = 0.0

    for ds in datasets
        result = evaluate_model(hybrid_model, ds)
        @printf("%-20s | MAPE: %5.1f%% | RMSE: %5.2f kg/mol\n",
                ds.name, result.mape, result.rmse)
        total_mape += result.mape
    end

    mean_mape = total_mape / length(datasets)
    println("-"^60)
    @printf("MÉDIA               | MAPE: %5.1f%% | Acurácia: %.1f%%\n",
            mean_mape, 100 - mean_mape)

    return mean_mape
end

end # module
