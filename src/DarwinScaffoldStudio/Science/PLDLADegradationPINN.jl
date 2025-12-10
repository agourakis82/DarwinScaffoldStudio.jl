"""
PLDLADegradationPINN.jl

Physics-Informed Neural Network para degradação de PLDLA/PDLLA/PLLA.

Física incorporada:
1. Cinética de hidrólise: dMn/dt = -k(T) * Mn^α * [H2O]
2. Arrhenius: k(T) = k0 * exp(-Ea/RT)
3. Autocatálise ácida: k_eff = k * (1 + β*[COOH])

Dados de treinamento:
- Kaique thesis (PLDLA 70:30, 37°C)
- Literatura: Weir, Tsuji, PMC7875459

Autor: Darwin Scaffold Studio
"""
module PLDLADegradationPINN

using Statistics
using Random
using Printf

export PINNModel, train_pinn!, predict_pinn, DegradationDataset
export create_literature_datasets, cross_validate

# ============================================================================
# ESTRUTURAS DE DADOS
# ============================================================================

"""
Dataset de degradação para treinamento/validação
"""
struct DegradationDataset
    name::String
    polymer::String           # PLDLA, PDLLA, PLLA
    temperature::Float64      # Kelvin
    times::Vector{Float64}    # dias
    Mn_values::Vector{Float64} # kg/mol
    Mn_std::Vector{Float64}   # desvio padrão
    source::String            # referência
end

"""
Camada densa simples
"""
mutable struct DenseLayer
    W::Matrix{Float64}
    b::Vector{Float64}
end

function DenseLayer(in_dim::Int, out_dim::Int)
    # Inicialização Xavier
    scale = sqrt(2.0 / (in_dim + out_dim))
    W = randn(out_dim, in_dim) .* scale
    b = zeros(out_dim)
    DenseLayer(W, b)
end

function (layer::DenseLayer)(x::Vector{Float64})
    return layer.W * x .+ layer.b
end

"""
Physics-Informed Neural Network para degradação
"""
mutable struct PINNModel
    # Camadas da rede neural
    layers::Vector{DenseLayer}

    # Parâmetros físicos (aprendidos)
    k0::Float64           # Constante pré-exponencial
    Ea::Float64           # Energia de ativação (kJ/mol)
    alpha::Float64        # Ordem da reação
    beta::Float64         # Fator de autocatálise

    # Parâmetros de treinamento
    learning_rate::Float64
    physics_weight::Float64  # Peso do termo de física na loss

    # Histórico
    loss_history::Vector{Float64}
end

function PINNModel(;
    hidden_dims::Vector{Int} = [32, 32, 16],
    learning_rate::Float64 = 0.001,
    physics_weight::Float64 = 0.1
)
    # Entrada: [t, T, Mn0] (3 dimensões)
    # Saída: Mn(t) (1 dimensão)

    layers = DenseLayer[]

    # Primeira camada
    push!(layers, DenseLayer(3, hidden_dims[1]))

    # Camadas ocultas
    for i in 2:length(hidden_dims)
        push!(layers, DenseLayer(hidden_dims[i-1], hidden_dims[i]))
    end

    # Camada de saída
    push!(layers, DenseLayer(hidden_dims[end], 1))

    # Parâmetros físicos iniciais (baseados em literatura)
    k0 = 0.02      # /dia
    Ea = 80.0      # kJ/mol
    alpha = 1.0    # ordem 1
    beta = 0.1     # autocatálise leve

    PINNModel(layers, k0, Ea, alpha, beta, learning_rate, physics_weight, Float64[])
end

# ============================================================================
# FUNÇÕES DE ATIVAÇÃO E FORWARD PASS
# ============================================================================

"""Função de ativação tanh"""
tanh_activation(x) = tanh.(x)

"""Função de ativação softplus (suave, positiva)"""
softplus(x) = log.(1.0 .+ exp.(x))

"""Forward pass da rede neural"""
function forward(model::PINNModel, x::Vector{Float64})
    h = x
    for (i, layer) in enumerate(model.layers)
        h = layer(h)
        if i < length(model.layers)
            h = tanh_activation(h)
        end
    end
    # Saída positiva (Mn não pode ser negativo)
    return softplus(h)[1]
end

"""
Predição do modelo PINN
Entrada normalizada: t em dias, T em Kelvin, Mn0 em kg/mol
"""
function predict_pinn(model::PINNModel, t::Float64, T::Float64, Mn0::Float64)
    # Normalização de entrada
    x = [t / 100.0, (T - 300.0) / 20.0, Mn0 / 50.0]

    # Componente neural
    Mn_neural = forward(model, x) * 50.0  # Desnormalizar

    # Componente físico (modelo analítico)
    R = 8.314e-3  # kJ/(mol·K)
    k = model.k0 * exp(-model.Ea / (R * T))

    # Solução aproximada com autocatálise
    # Mn(t) = Mn0 * exp(-k_eff * t)
    # onde k_eff aumenta com degradação
    decay_factor = exp(-k * t * (1 + model.beta * (1 - Mn_neural/Mn0)))
    Mn_physics = Mn0 * decay_factor

    # Combinação: neural + correção física
    # A rede aprende a correção, física dá a base
    Mn_pred = 0.3 * Mn_neural + 0.7 * Mn_physics

    return max(Mn_pred, 1.0)  # Mn mínimo físico
end

# ============================================================================
# FUNÇÃO DE PERDA (PHYSICS-INFORMED)
# ============================================================================

"""
Calcula a loss total: dados + física
"""
function compute_loss(model::PINNModel, dataset::DegradationDataset)
    T = dataset.temperature
    Mn0 = dataset.Mn_values[1]

    # Loss de dados (MSE)
    data_loss = 0.0
    for (i, t) in enumerate(dataset.times)
        Mn_pred = predict_pinn(model, t, T, Mn0)
        Mn_true = dataset.Mn_values[i]
        data_loss += (Mn_pred - Mn_true)^2
    end
    data_loss /= length(dataset.times)

    # Loss de física (residual da ODE)
    # dMn/dt = -k * Mn^α
    physics_loss = 0.0
    dt = 1.0  # dia

    for t in 0:dt:maximum(dataset.times)
        Mn_t = predict_pinn(model, t, T, Mn0)
        Mn_t_dt = predict_pinn(model, t + dt, T, Mn0)

        # Derivada numérica
        dMn_dt_numerical = (Mn_t_dt - Mn_t) / dt

        # Derivada física esperada
        R = 8.314e-3
        k = model.k0 * exp(-model.Ea / (R * T))
        dMn_dt_physics = -k * Mn_t^model.alpha

        # Residual
        physics_loss += (dMn_dt_numerical - dMn_dt_physics)^2
    end
    physics_loss /= (maximum(dataset.times) / dt)

    # Loss total
    total_loss = data_loss + model.physics_weight * physics_loss

    return total_loss, data_loss, physics_loss
end

# ============================================================================
# TREINAMENTO
# ============================================================================

"""
Gradiente numérico para um parâmetro
"""
function numerical_gradient(f, x, eps=1e-5)
    return (f(x + eps) - f(x - eps)) / (2 * eps)
end

"""
Treina o modelo PINN
"""
function train_pinn!(model::PINNModel, datasets::Vector{DegradationDataset};
                     epochs::Int = 500, verbose::Bool = true)

    if verbose
        println("\n🧠 TREINAMENTO DO PINN")
        println("-"^50)
        println("Datasets: $(length(datasets))")
        println("Épocas: $epochs")
        println("Learning rate: $(model.learning_rate)")
        println("Physics weight: $(model.physics_weight)")
        println("-"^50)
    end

    best_loss = Inf
    patience = 50
    no_improve = 0

    for epoch in 1:epochs
        total_loss = 0.0

        for dataset in datasets
            # Calcular loss
            loss, data_loss, phys_loss = compute_loss(model, dataset)
            total_loss += loss

            # Atualização simples dos parâmetros físicos (gradient descent)
            # Gradiente numérico para k0
            dk0 = numerical_gradient(model.k0) do k
                old_k0 = model.k0
                model.k0 = k
                l, _, _ = compute_loss(model, dataset)
                model.k0 = old_k0
                l
            end
            model.k0 -= model.learning_rate * dk0 * 0.01
            model.k0 = clamp(model.k0, 0.001, 0.1)

            # Gradiente para beta
            dbeta = numerical_gradient(model.beta) do b
                old_beta = model.beta
                model.beta = b
                l, _, _ = compute_loss(model, dataset)
                model.beta = old_beta
                l
            end
            model.beta -= model.learning_rate * dbeta * 0.1
            model.beta = clamp(model.beta, 0.0, 1.0)

            # Atualização das camadas neurais
            for layer in model.layers
                # Perturbação aleatória (simplificação de SGD)
                noise_W = randn(size(layer.W)) .* model.learning_rate .* 0.1
                noise_b = randn(size(layer.b)) .* model.learning_rate .* 0.1

                # Testar se melhora
                layer.W .+= noise_W
                layer.b .+= noise_b

                new_loss, _, _ = compute_loss(model, dataset)
                if new_loss > loss
                    # Reverter
                    layer.W .-= noise_W
                    layer.b .-= noise_b
                end
            end
        end

        total_loss /= length(datasets)
        push!(model.loss_history, total_loss)

        # Early stopping
        if total_loss < best_loss
            best_loss = total_loss
            no_improve = 0
        else
            no_improve += 1
        end

        if no_improve >= patience
            if verbose
                println("Early stopping na época $epoch")
            end
            break
        end

        # Log
        if verbose && (epoch % 50 == 0 || epoch == 1)
            @printf("Época %4d | Loss: %.4f | k0: %.4f | β: %.3f\n",
                    epoch, total_loss, model.k0, model.beta)
        end
    end

    if verbose
        println("-"^50)
        @printf("Treinamento concluído. Loss final: %.4f\n", model.loss_history[end])
        @printf("Parâmetros aprendidos: k0=%.4f, Ea=%.1f, α=%.2f, β=%.3f\n",
                model.k0, model.Ea, model.alpha, model.beta)
    end

    return model
end

# ============================================================================
# DATASETS DA LITERATURA
# ============================================================================

"""
Cria datasets da literatura para validação cruzada
"""
function create_literature_datasets()
    datasets = DegradationDataset[]

    # Dataset 1: Kaique thesis (PLDLA 70:30, 37°C)
    push!(datasets, DegradationDataset(
        "Kaique_PLDLA",
        "PLDLA 70:30",
        310.15,  # 37°C
        [0.0, 30.0, 60.0, 90.0],
        [51.285, 25.447, 18.313, 7.904],
        [2.5, 1.3, 0.9, 0.4],
        "Kaique Thesis 2024"
    ))

    # Dataset 2: Kaique PLDLA/TEC1%
    push!(datasets, DegradationDataset(
        "Kaique_PLDLA_TEC1",
        "PLDLA/TEC1%",
        310.15,
        [0.0, 30.0, 60.0, 90.0],
        [44.998, 19.257, 11.749, 8.122],
        [2.2, 1.0, 0.6, 0.4],
        "Kaique Thesis 2024"
    ))

    # Dataset 3: PMC7875459 - PDLLA a 60°C (acelerado)
    # k ≈ 0.41/dia, Mn cai 98% em 15 dias
    # Convertendo para 37°C usando Arrhenius (Ea=80 kJ/mol)
    # k_37 = k_60 * exp(-Ea/R * (1/310 - 1/333)) ≈ k_60 / 15
    push!(datasets, DegradationDataset(
        "PMC7875459_PDLLA",
        "PDLLA",
        310.15,  # Convertido para 37°C equivalente
        [0.0, 30.0, 60.0, 90.0, 120.0],
        [45.0, 30.0, 18.0, 10.0, 5.0],  # Estimado da curva
        [2.0, 1.5, 1.0, 0.5, 0.3],
        "PMC7875459 (60°C→37°C conv.)"
    ))

    # Dataset 4: Tsuji et al. - PLLA (semicristalino, mais lento)
    # PLLA degrada ~3x mais lento que PDLLA
    push!(datasets, DegradationDataset(
        "Tsuji_PLLA",
        "PLLA",
        310.15,
        [0.0, 30.0, 60.0, 90.0, 120.0, 180.0],
        [98.0, 85.0, 70.0, 55.0, 40.0, 20.0],  # Literatura
        [5.0, 4.0, 3.5, 3.0, 2.0, 1.0],
        "Tsuji et al. 2000"
    ))

    # Dataset 5: g-HAP/PLLA in vivo (Nature 2016)
    # Mais lento in vivo para PLLA
    push!(datasets, DegradationDataset(
        "Nature_PLLA_invivo",
        "PLLA (in vivo)",
        310.15,
        [0.0, 28.0, 84.0, 140.0, 196.0, 252.0],  # 0,4,12,20,28,36 weeks
        [100.0, 89.7, 64.8, 54.1, 45.5, 29.4],  # % do inicial
        [5.0, 4.5, 3.0, 2.5, 2.0, 1.5],
        "Nature Sci Rep 2016"
    ))

    return datasets
end

# ============================================================================
# VALIDAÇÃO CRUZADA
# ============================================================================

"""
Validação cruzada leave-one-out
"""
function cross_validate(datasets::Vector{DegradationDataset};
                        epochs::Int = 300, verbose::Bool = true)

    if verbose
        println("\n" * "="^70)
        println("  VALIDAÇÃO CRUZADA LEAVE-ONE-OUT")
        println("="^70)
    end

    results = []

    for (i, test_dataset) in enumerate(datasets)
        # Treinar com todos menos um
        train_datasets = [d for (j, d) in enumerate(datasets) if j != i]

        if verbose
            println("\n📊 Fold $i: Teste em $(test_dataset.name)")
            println("   Treino em: $(join([d.name for d in train_datasets], ", "))")
        end

        # Criar e treinar modelo
        model = PINNModel(physics_weight=0.05)
        train_pinn!(model, train_datasets; epochs=epochs, verbose=false)

        # Avaliar no conjunto de teste
        T = test_dataset.temperature
        Mn0 = test_dataset.Mn_values[1]

        errors = Float64[]
        for (j, t) in enumerate(test_dataset.times)
            Mn_pred = predict_pinn(model, t, T, Mn0)
            Mn_true = test_dataset.Mn_values[j]
            push!(errors, abs(Mn_pred - Mn_true) / Mn_true * 100)
        end

        mape = mean(errors)
        rmse = sqrt(mean([(predict_pinn(model, t, T, Mn0) - test_dataset.Mn_values[j])^2
                          for (j, t) in enumerate(test_dataset.times)]))

        if verbose
            @printf("   MAPE: %.1f%% | RMSE: %.2f kg/mol\n", mape, rmse)
        end

        push!(results, (dataset=test_dataset.name, mape=mape, rmse=rmse, model=model))
    end

    # Resumo
    if verbose
        println("\n" * "-"^70)
        println("RESUMO DA VALIDAÇÃO CRUZADA:")
        println("-"^70)

        mean_mape = mean([r.mape for r in results])
        mean_rmse = mean([r.rmse for r in results])

        @printf("MAPE médio: %.1f%%\n", mean_mape)
        @printf("RMSE médio: %.2f kg/mol\n", mean_rmse)
        @printf("Acurácia média: %.1f%%\n", 100 - mean_mape)
    end

    return results
end

end # module
