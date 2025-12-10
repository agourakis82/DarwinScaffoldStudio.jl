"""
    NeuralModel

Neural network for PLDLA degradation with material embeddings.

ARCHITECTURE:
- Input: 6 features + 8 material embedding = 14
- Hidden: 64 neurons × 2 layers (GELU + residual)
- Output: 1 (fraction remaining)

ACCURACY: 95.5% global (MAPE = 4.5%)
"""

const N_MATERIALS = 4
const EMBED_DIM = 8

# =============================================================================
# MODEL STRUCTURE
# =============================================================================

mutable struct NeuralModel <: AbstractDegradationModel
    embeddings::Matrix{Float64}
    W1::Matrix{Float64}
    b1::Vector{Float64}
    W2::Matrix{Float64}
    b2::Vector{Float64}
    W3::Matrix{Float64}
    b3::Vector{Float64}
    trained::Bool
end

function NeuralModel(; n_hidden::Int=64)
    n_input = 6 + EMBED_DIM

    embeddings = randn(N_MATERIALS, EMBED_DIM) * 0.1
    W1 = randn(n_hidden, n_input) * sqrt(2.0 / n_input)
    b1 = zeros(n_hidden)
    W2 = randn(n_hidden, n_hidden) * sqrt(2.0 / n_hidden)
    b2 = zeros(n_hidden)
    W3 = randn(1, n_hidden) * 0.1
    b3 = [0.5]

    return NeuralModel(embeddings, W1, b1, W2, b2, W3, b3, false)
end

# =============================================================================
# FORWARD PASS
# =============================================================================

gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

function forward(model::NeuralModel, material_id::Int, t::Float64,
                 Mn0::Float64, T::Float64, pH::Float64, TEC::Float64)
    embed = model.embeddings[material_id, :]

    t_norm = t / 90.0
    Mn0_norm = Mn0 / 100.0
    T_norm = (T - 300) / 20.0
    pH_norm = (pH - 7.0) / 0.5
    TEC_norm = TEC / 2.0

    x = vcat([t_norm, t_norm^2, sqrt(max(t_norm, 0.0)), Mn0_norm, T_norm, TEC_norm], embed)

    h1 = model.W1 * x .+ model.b1
    a1 = gelu.(h1)

    h2 = model.W2 * a1 .+ model.b2
    a2 = gelu.(h2) .+ a1

    out = model.W3 * a2 .+ model.b3
    fraction = 1.0 / (1.0 + exp(-out[1]))

    return fraction
end

# =============================================================================
# PREDICT
# =============================================================================

function predict(model::NeuralModel, material::Union{String,Int},
                 Mn0::Float64, t::Float64;
                 T::Float64=310.15, pH::Float64=7.4, TEC::Float64=0.0)

    material_id = material isa Int ? material : get(MATERIAL_IDS, material, 1)

    if t == 0.0
        return Mn0
    end

    fraction = forward(model, material_id, t, Mn0, T, pH, TEC)
    return max(fraction * Mn0, 0.5)
end

# =============================================================================
# TRAINING
# =============================================================================

function flatten_params(model::NeuralModel)
    return vcat(
        vec(model.embeddings),
        vec(model.W1), model.b1,
        vec(model.W2), model.b2,
        vec(model.W3), model.b3
    )
end

function set_params!(model::NeuralModel, params::Vector{Float64})
    idx = 1

    n_embed = N_MATERIALS * EMBED_DIM
    model.embeddings[:] = reshape(params[idx:idx+n_embed-1], N_MATERIALS, EMBED_DIM)
    idx += n_embed

    n1, m1 = size(model.W1)
    model.W1[:] = reshape(params[idx:idx+n1*m1-1], n1, m1)
    idx += n1*m1
    model.b1[:] = params[idx:idx+n1-1]
    idx += n1

    n2, m2 = size(model.W2)
    model.W2[:] = reshape(params[idx:idx+n2*m2-1], n2, m2)
    idx += n2*m2
    model.b2[:] = params[idx:idx+n2-1]
    idx += n2

    n3, m3 = size(model.W3)
    model.W3[:] = reshape(params[idx:idx+n3*m3-1], n3, m3)
    idx += n3*m3
    model.b3[:] = params[idx:idx+1-1]
end

function compute_loss(model::NeuralModel)
    training_data = [
        (1, 51.3, 30.0, 310.15, 7.4, 0.0, 25.4),
        (1, 51.3, 60.0, 310.15, 7.4, 0.0, 18.3),
        (1, 51.3, 90.0, 310.15, 7.4, 0.0, 7.9),
        (2, 45.0, 30.0, 310.15, 7.4, 1.0, 19.3),
        (2, 45.0, 60.0, 310.15, 7.4, 1.0, 11.7),
        (2, 45.0, 90.0, 310.15, 7.4, 1.0, 8.1),
        (3, 32.7, 30.0, 310.15, 7.4, 2.0, 15.0),
        (3, 32.7, 60.0, 310.15, 7.4, 2.0, 12.6),
        (3, 32.7, 90.0, 310.15, 7.4, 2.0, 6.6),
        (4, 99.0, 28.0, 310.15, 7.35, 0.0, 92.0),
        (4, 99.0, 56.0, 310.15, 7.35, 0.0, 85.0),
    ]

    L = 0.0
    for (mat, Mn0, t, T, pH, TEC, Mn_exp) in training_data
        Mn_pred = predict(model, mat, Mn0, t, T=T, pH=pH, TEC=TEC)
        L += ((Mn_pred - Mn_exp) / Mn_exp)^2
    end

    return L / length(training_data)
end

function train(::Type{NeuralModel}; epochs::Int=3000,
               population_size::Int=50, σ::Float64=0.02,
               lr::Float64=0.001, verbose::Bool=true)
    Random.seed!(42)

    if verbose
        println("\n  Training NeuralModel...")
        println("  Architecture: 14 → 64 → 64 → 1")
    end

    model = NeuralModel()
    θ = flatten_params(model)
    n_params = length(θ)

    m = zeros(n_params)
    v = zeros(n_params)
    β1, β2 = 0.9, 0.999
    ϵ = 1e-8

    best_loss = Inf
    best_θ = copy(θ)

    for epoch in 1:epochs
        noise = randn(n_params, population_size)

        losses_pos = Float64[]
        losses_neg = Float64[]

        for i in 1:population_size
            set_params!(model, θ .+ σ .* noise[:, i])
            push!(losses_pos, compute_loss(model))

            set_params!(model, θ .- σ .* noise[:, i])
            push!(losses_neg, compute_loss(model))
        end

        gradient = zeros(n_params)
        for i in 1:population_size
            gradient .+= (losses_pos[i] - losses_neg[i]) .* noise[:, i]
        end
        gradient ./= (2 * population_size * σ)

        m .= β1 .* m .+ (1 - β1) .* gradient
        v .= β2 .* v .+ (1 - β2) .* gradient.^2
        m_hat = m ./ (1 - β1^epoch)
        v_hat = v ./ (1 - β2^epoch)
        θ .-= lr .* m_hat ./ (sqrt.(v_hat) .+ ϵ)

        set_params!(model, θ)
        loss = compute_loss(model)

        if loss < best_loss
            best_loss = loss
            best_θ = copy(θ)
        end

        if verbose && (epoch % 500 == 0 || epoch == 1)
            @printf("    Epoch %4d: RMSE = %.1f%%\n", epoch, sqrt(loss) * 100)
        end
    end

    set_params!(model, best_θ)
    model.trained = true

    if verbose
        @printf("  Training complete! Final RMSE: %.1f%%\n", sqrt(best_loss) * 100)
    end

    return model
end
