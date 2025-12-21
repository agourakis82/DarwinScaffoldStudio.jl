# RSSM.jl - Recurrent State-Space Model
# Core component for learning world models with stochastic dynamics
# Based on Hafner et al. "Learning Latent Dynamics for Planning"

module RSSM

using LinearAlgebra
using Statistics
using Random

export RSSMConfig, RSSMState, RSSMModel
export prior_step, posterior_step, observe_sequence
export stochastic_state, deterministic_state, full_state
export RSSMCell, DiscreteRSSM, ContinuousRSSM
export sample_discrete, sample_gaussian, kl_discrete, kl_gaussian

# ============================================================================
# Configuration
# ============================================================================

"""
    RSSMConfig

Configuration for Recurrent State-Space Model.

# Fields
- `deterministic_size::Int`: Size of deterministic GRU state (h)
- `stochastic_size::Int`: Size of stochastic state (z)
- `hidden_size::Int`: Hidden layer size for networks
- `embedding_size::Int`: Size of observation embedding
- `action_size::Int`: Size of action input
- `discrete::Bool`: Use discrete (categorical) vs continuous (Gaussian) latents
- `num_classes::Int`: Number of classes per categorical (if discrete)
- `num_categoricals::Int`: Number of categorical distributions (if discrete)
- `activation::Symbol`: Activation function (:elu, :relu, :silu)
- `min_std::Float64`: Minimum standard deviation for continuous latents
"""
struct RSSMConfig
    deterministic_size::Int
    stochastic_size::Int
    hidden_size::Int
    embedding_size::Int
    action_size::Int
    discrete::Bool
    num_classes::Int
    num_categoricals::Int
    activation::Symbol
    min_std::Float64

    function RSSMConfig(;
        deterministic_size::Int = 512,
        stochastic_size::Int = 32,
        hidden_size::Int = 512,
        embedding_size::Int = 256,
        action_size::Int = 4,
        discrete::Bool = true,
        num_classes::Int = 32,
        num_categoricals::Int = 32,
        activation::Symbol = :elu,
        min_std::Float64 = 0.1
    )
        new(deterministic_size, stochastic_size, hidden_size, embedding_size,
            action_size, discrete, num_classes, num_categoricals, activation, min_std)
    end
end

# ============================================================================
# Activations
# ============================================================================

function get_activation(name::Symbol)
    if name == :elu
        return x -> x >= 0 ? x : exp(x) - 1
    elseif name == :relu
        return x -> max(0, x)
    elseif name == :silu
        return x -> x * (1 / (1 + exp(-x)))
    elseif name == :tanh
        return tanh
    else
        return identity
    end
end

# ============================================================================
# RSSM State
# ============================================================================

"""
    RSSMState

Complete state of the RSSM at a single timestep.

# Fields
- `h::Vector{Float64}`: Deterministic recurrent state
- `z::Vector{Float64}`: Stochastic latent state (sampled)
- `logits::Vector{Float64}`: Distribution parameters (logits for discrete, mean/logstd for continuous)
- `discrete::Bool`: Whether state uses discrete latents
"""
struct RSSMState
    h::Vector{Float64}
    z::Vector{Float64}
    logits::Vector{Float64}
    discrete::Bool

    function RSSMState(h::Vector{Float64}, z::Vector{Float64},
                       logits::Vector{Float64}; discrete::Bool = true)
        new(h, z, logits, discrete)
    end
end

"""Get the deterministic component of state."""
deterministic_state(s::RSSMState) = s.h

"""Get the stochastic component of state."""
stochastic_state(s::RSSMState) = s.z

"""Get the full concatenated state."""
full_state(s::RSSMState) = vcat(s.h, s.z)

"""Create initial zero state."""
function initial_state(config::RSSMConfig)
    h = zeros(config.deterministic_size)
    if config.discrete
        z = zeros(config.num_categoricals)
        logits = zeros(config.num_categoricals * config.num_classes)
    else
        z = zeros(config.stochastic_size)
        logits = zeros(2 * config.stochastic_size)  # mean + logstd
    end
    RSSMState(h, z, logits, discrete = config.discrete)
end

# ============================================================================
# Dense Layer
# ============================================================================

struct DenseLayer
    W::Matrix{Float64}
    b::Vector{Float64}
    activation::Function
end

function DenseLayer(in_dim::Int, out_dim::Int;
                    activation::Function = identity, rng = Random.GLOBAL_RNG)
    scale = sqrt(2.0 / (in_dim + out_dim))
    W = randn(rng, out_dim, in_dim) .* scale
    b = zeros(out_dim)
    DenseLayer(W, b, activation)
end

(layer::DenseLayer)(x::Vector{Float64}) = layer.activation.(layer.W * x .+ layer.b)

function (layer::DenseLayer)(x::Matrix{Float64})
    hcat([layer(x[:, i]) for i in 1:size(x, 2)]...)
end

# ============================================================================
# GRU Cell
# ============================================================================

struct GRUCell
    Wz::Matrix{Float64}
    Wr::Matrix{Float64}
    Wh::Matrix{Float64}
    Uz::Matrix{Float64}
    Ur::Matrix{Float64}
    Uh::Matrix{Float64}
    bz::Vector{Float64}
    br::Vector{Float64}
    bh::Vector{Float64}
    hidden_size::Int
end

function GRUCell(input_size::Int, hidden_size::Int; rng = Random.GLOBAL_RNG)
    scale = sqrt(2.0 / (input_size + hidden_size))
    GRUCell(
        randn(rng, hidden_size, input_size) .* scale,
        randn(rng, hidden_size, input_size) .* scale,
        randn(rng, hidden_size, input_size) .* scale,
        randn(rng, hidden_size, hidden_size) .* scale,
        randn(rng, hidden_size, hidden_size) .* scale,
        randn(rng, hidden_size, hidden_size) .* scale,
        zeros(hidden_size),
        zeros(hidden_size),
        zeros(hidden_size),
        hidden_size
    )
end

sigmoid(x) = 1.0 / (1.0 + exp(-clamp(x, -20, 20)))

function (gru::GRUCell)(h::Vector{Float64}, x::Vector{Float64})
    z = sigmoid.(gru.Wz * x .+ gru.Uz * h .+ gru.bz)
    r = sigmoid.(gru.Wr * x .+ gru.Ur * h .+ gru.br)
    h_tilde = tanh.(gru.Wh * x .+ gru.Uh * (r .* h) .+ gru.bh)
    return (1 .- z) .* h .+ z .* h_tilde
end

# ============================================================================
# Discrete Sampling (Categorical with Straight-Through)
# ============================================================================

"""
    sample_discrete(logits, num_categoricals, num_classes)

Sample from categorical distribution with straight-through gradient estimator.
"""
function sample_discrete(logits::Vector{Float64}, num_categoricals::Int, num_classes::Int;
                         temperature::Float64 = 1.0)
    @assert length(logits) == num_categoricals * num_classes

    samples = zeros(num_categoricals)

    for i in 1:num_categoricals
        start_idx = (i - 1) * num_classes + 1
        end_idx = i * num_classes
        cat_logits = logits[start_idx:end_idx]

        # Apply temperature and softmax
        probs = softmax(cat_logits ./ temperature)

        # Sample using Gumbel-argmax
        gumbels = -log.(-log.(rand(num_classes) .+ 1e-10) .+ 1e-10)
        samples[i] = argmax(log.(probs .+ 1e-10) .+ gumbels)
    end

    return samples
end

function softmax(x::Vector{Float64})
    x_shifted = x .- maximum(x)
    exp_x = exp.(x_shifted)
    return exp_x ./ sum(exp_x)
end

"""
    one_hot_discrete(samples, num_classes)

Convert discrete samples to one-hot representation.
"""
function one_hot_discrete(samples::Vector{Float64}, num_classes::Int)
    result = zeros(length(samples) * num_classes)
    for (i, s) in enumerate(samples)
        idx = (i - 1) * num_classes + Int(s)
        result[idx] = 1.0
    end
    return result
end

# ============================================================================
# Continuous Sampling (Gaussian)
# ============================================================================

"""
    sample_gaussian(mean, logstd; min_std)

Sample from Gaussian using reparameterization trick.
"""
function sample_gaussian(mean::Vector{Float64}, logstd::Vector{Float64};
                         min_std::Float64 = 0.1)
    std = max.(exp.(logstd), min_std)
    eps = randn(length(mean))
    return mean .+ std .* eps
end

# ============================================================================
# KL Divergence
# ============================================================================

"""
    kl_discrete(post_logits, prior_logits, num_categoricals, num_classes)

KL divergence between discrete categorical distributions.
"""
function kl_discrete(post_logits::Vector{Float64}, prior_logits::Vector{Float64},
                     num_categoricals::Int, num_classes::Int)
    kl = 0.0

    for i in 1:num_categoricals
        start_idx = (i - 1) * num_classes + 1
        end_idx = i * num_classes

        post_probs = softmax(post_logits[start_idx:end_idx])
        prior_probs = softmax(prior_logits[start_idx:end_idx])

        for j in 1:num_classes
            if post_probs[j] > 1e-8
                kl += post_probs[j] * (log(post_probs[j] + 1e-8) - log(prior_probs[j] + 1e-8))
            end
        end
    end

    return kl
end

"""
    kl_gaussian(post_mean, post_logstd, prior_mean, prior_logstd)

KL divergence between Gaussian distributions.
KL(q||p) where q = N(post_mean, post_std^2), p = N(prior_mean, prior_std^2)
"""
function kl_gaussian(post_mean::Vector{Float64}, post_logstd::Vector{Float64},
                     prior_mean::Vector{Float64}, prior_logstd::Vector{Float64})
    post_var = exp.(2 .* post_logstd)
    prior_var = exp.(2 .* prior_logstd)

    kl = 0.5 * sum(
        prior_logstd .- post_logstd .-
        (prior_var .+ (prior_mean .- post_mean).^2) ./ (2 .* post_var) .+ 0.5
    )

    return -kl  # Return positive KL
end

# ============================================================================
# RSSM Model
# ============================================================================

"""
    RSSMModel

Complete RSSM implementation with prior and posterior networks.
"""
struct RSSMModel
    config::RSSMConfig

    # Recurrent model
    gru::GRUCell
    gru_input_layer::DenseLayer

    # Prior network (h -> z distribution)
    prior_layers::Vector{DenseLayer}
    prior_out::DenseLayer

    # Posterior network (h, embed -> z distribution)
    posterior_layers::Vector{DenseLayer}
    posterior_out::DenseLayer

    # Embedding network
    embed_layer::DenseLayer
end

function RSSMModel(config::RSSMConfig; rng = Random.GLOBAL_RNG)
    act = get_activation(config.activation)

    # Compute stochastic input size
    if config.discrete
        stoch_input_size = config.num_categoricals  # Sampled category indices
    else
        stoch_input_size = config.stochastic_size
    end

    # GRU input combines previous stochastic state and action
    gru_input_size = stoch_input_size + config.action_size
    gru_input_layer = DenseLayer(gru_input_size, config.deterministic_size,
                                  activation = act, rng = rng)

    # GRU cell
    gru = GRUCell(config.deterministic_size, config.deterministic_size, rng = rng)

    # Prior output size
    if config.discrete
        prior_output_size = config.num_categoricals * config.num_classes
    else
        prior_output_size = 2 * config.stochastic_size  # mean + logstd
    end

    # Prior network: h -> z distribution
    prior_layers = [
        DenseLayer(config.deterministic_size, config.hidden_size, activation = act, rng = rng)
    ]
    prior_out = DenseLayer(config.hidden_size, prior_output_size, rng = rng)

    # Posterior network: (h, embed) -> z distribution
    posterior_input_size = config.deterministic_size + config.embedding_size
    posterior_layers = [
        DenseLayer(posterior_input_size, config.hidden_size, activation = act, rng = rng)
    ]
    posterior_out = DenseLayer(config.hidden_size, prior_output_size, rng = rng)

    # Embedding layer (observation -> embed)
    embed_layer = DenseLayer(config.embedding_size, config.embedding_size,
                             activation = act, rng = rng)

    RSSMModel(config, gru, gru_input_layer, prior_layers, prior_out,
              posterior_layers, posterior_out, embed_layer)
end

function apply_layers(layers::Vector{DenseLayer}, x::Vector{Float64})
    for layer in layers
        x = layer(x)
    end
    return x
end

"""
    prior_step(model, prev_state, action)

Compute prior prediction (imagination without observation).
Returns new RSSMState with prior distribution.
"""
function prior_step(model::RSSMModel, prev_state::RSSMState, action::Vector{Float64})
    config = model.config

    # Combine previous stochastic and action
    za = vcat(prev_state.z, action)

    # Transform to GRU input dimension
    gru_input = model.gru_input_layer(za)

    # Update deterministic state
    h_new = model.gru(prev_state.h, gru_input)

    # Compute prior distribution
    prior_hidden = apply_layers(model.prior_layers, h_new)
    prior_logits = model.prior_out(prior_hidden)

    # Sample stochastic state
    if config.discrete
        z_new = sample_discrete(prior_logits, config.num_categoricals, config.num_classes)
    else
        mid = config.stochastic_size
        mean = prior_logits[1:mid]
        logstd = prior_logits[mid+1:end]
        z_new = sample_gaussian(mean, logstd, min_std = config.min_std)
    end

    RSSMState(h_new, z_new, prior_logits, discrete = config.discrete)
end

"""
    posterior_step(model, prev_state, action, embedding)

Compute posterior (with observation).
Returns new RSSMState with posterior distribution.
"""
function posterior_step(model::RSSMModel, prev_state::RSSMState,
                        action::Vector{Float64}, embedding::Vector{Float64})
    config = model.config

    # Combine previous stochastic and action
    za = vcat(prev_state.z, action)

    # Transform to GRU input dimension
    gru_input = model.gru_input_layer(za)

    # Update deterministic state
    h_new = model.gru(prev_state.h, gru_input)

    # Compute posterior distribution (uses observation embedding)
    embed = model.embed_layer(embedding)
    posterior_input = vcat(h_new, embed)
    posterior_hidden = apply_layers(model.posterior_layers, posterior_input)
    posterior_logits = model.posterior_out(posterior_hidden)

    # Sample stochastic state
    if config.discrete
        z_new = sample_discrete(posterior_logits, config.num_categoricals, config.num_classes)
    else
        mid = config.stochastic_size
        mean = posterior_logits[1:mid]
        logstd = posterior_logits[mid+1:end]
        z_new = sample_gaussian(mean, logstd, min_std = config.min_std)
    end

    RSSMState(h_new, z_new, posterior_logits, discrete = config.discrete)
end

"""
    get_prior_logits(model, h)

Get prior distribution logits from deterministic state.
"""
function get_prior_logits(model::RSSMModel, h::Vector{Float64})
    prior_hidden = apply_layers(model.prior_layers, h)
    model.prior_out(prior_hidden)
end

"""
    observe_sequence(model, embeddings, actions)

Process a sequence of observations through the RSSM.
Returns lists of prior and posterior states.
"""
function observe_sequence(model::RSSMModel, embeddings::Vector{Vector{Float64}},
                          actions::Vector{Vector{Float64}})
    T = length(embeddings)
    @assert length(actions) == T

    config = model.config
    state = initial_state(config)

    prior_states = RSSMState[]
    posterior_states = RSSMState[]

    zero_action = zeros(config.action_size)

    for t in 1:T
        # Use previous action (or zero for t=1)
        prev_action = t == 1 ? zero_action : actions[t-1]

        # Prior (imagination without observation)
        prior_state = prior_step(model, state, prev_action)
        push!(prior_states, prior_state)

        # Posterior (with observation)
        posterior_state = posterior_step(model, state, prev_action, embeddings[t])
        push!(posterior_states, posterior_state)

        # Update state for next step (use posterior)
        state = posterior_state
    end

    return (prior = prior_states, posterior = posterior_states)
end

"""
    imagine_sequence(model, initial_state, actions)

Imagine a sequence using only prior predictions.
"""
function imagine_sequence(model::RSSMModel, initial_state::RSSMState,
                          actions::Vector{Vector{Float64}})
    T = length(actions)
    states = RSSMState[initial_state]

    state = initial_state
    for t in 1:T
        state = prior_step(model, state, actions[t])
        push!(states, state)
    end

    return states
end

# ============================================================================
# Loss Functions
# ============================================================================

"""
    compute_kl_loss(model, prior_states, posterior_states; free_nats, kl_balance)

Compute KL divergence loss between prior and posterior.
Supports KL balancing from DreamerV2.
"""
function compute_kl_loss(model::RSSMModel, prior_states::Vector{RSSMState},
                         posterior_states::Vector{RSSMState};
                         free_nats::Float64 = 1.0, kl_balance::Float64 = 0.8)
    config = model.config
    T = length(prior_states)

    kl_total = 0.0
    kl_prior = 0.0  # For KL balancing

    for t in 1:T
        if config.discrete
            kl = kl_discrete(posterior_states[t].logits, prior_states[t].logits,
                           config.num_categoricals, config.num_classes)
            kl_rev = kl_discrete(prior_states[t].logits, posterior_states[t].logits,
                                config.num_categoricals, config.num_classes)
        else
            mid = config.stochastic_size
            post_mean = posterior_states[t].logits[1:mid]
            post_logstd = posterior_states[t].logits[mid+1:end]
            prior_mean = prior_states[t].logits[1:mid]
            prior_logstd = prior_states[t].logits[mid+1:end]

            kl = kl_gaussian(post_mean, post_logstd, prior_mean, prior_logstd)
            kl_rev = kl_gaussian(prior_mean, prior_logstd, post_mean, post_logstd)
        end

        kl_total += max(kl, free_nats)
        kl_prior += max(kl_rev, free_nats)
    end

    # KL balancing: mostly train prior to match posterior
    balanced_kl = kl_balance * kl_prior + (1 - kl_balance) * kl_total

    return balanced_kl / T
end

# ============================================================================
# Specialized RSSM Variants
# ============================================================================

"""
Discrete RSSM with categorical latents (DreamerV2/V3 style).
"""
function DiscreteRSSM(; deterministic_size = 512, num_categoricals = 32, num_classes = 32,
                      hidden_size = 512, embedding_size = 256, action_size = 4,
                      rng = Random.GLOBAL_RNG)
    config = RSSMConfig(
        deterministic_size = deterministic_size,
        stochastic_size = num_categoricals,  # Will be overridden
        hidden_size = hidden_size,
        embedding_size = embedding_size,
        action_size = action_size,
        discrete = true,
        num_classes = num_classes,
        num_categoricals = num_categoricals
    )
    RSSMModel(config, rng = rng)
end

"""
Continuous RSSM with Gaussian latents (PlaNet/DreamerV1 style).
"""
function ContinuousRSSM(; deterministic_size = 200, stochastic_size = 30,
                        hidden_size = 200, embedding_size = 256, action_size = 4,
                        min_std = 0.1, rng = Random.GLOBAL_RNG)
    config = RSSMConfig(
        deterministic_size = deterministic_size,
        stochastic_size = stochastic_size,
        hidden_size = hidden_size,
        embedding_size = embedding_size,
        action_size = action_size,
        discrete = false,
        min_std = min_std
    )
    RSSMModel(config, rng = rng)
end

# ============================================================================
# Scaffold-Specific Functions
# ============================================================================

"""
    create_scaffold_rssm(; scaffold_features, design_params)

Create an RSSM configured for scaffold dynamics modeling.

# Arguments
- `scaffold_features::Int`: Number of scaffold observation features
- `design_params::Int`: Number of design parameters (actions)
- `use_discrete::Bool`: Use discrete (DreamerV3) or continuous (PlaNet) latents
"""
function create_scaffold_rssm(; scaffold_features::Int = 20, design_params::Int = 8,
                              use_discrete::Bool = true, rng = Random.GLOBAL_RNG)
    if use_discrete
        DiscreteRSSM(
            deterministic_size = 256,
            num_categoricals = 16,
            num_classes = 16,
            hidden_size = 256,
            embedding_size = scaffold_features,
            action_size = design_params,
            rng = rng
        )
    else
        ContinuousRSSM(
            deterministic_size = 128,
            stochastic_size = 32,
            hidden_size = 128,
            embedding_size = scaffold_features,
            action_size = design_params,
            rng = rng
        )
    end
end

end # module
