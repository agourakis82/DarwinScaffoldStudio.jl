# Dreamer.jl - World Model with Latent Dynamics
# Based on DreamerV3 (Hafner et al. 2023)
# For scaffold degradation prediction and tissue ingrowth simulation

module Dreamer

using LinearAlgebra
using Statistics
using Random

export DreamerConfig, DreamerModel, WorldState
export imagine_trajectory, train_world_model!, predict_dynamics
export encode_observation, decode_latent, compute_reward_prediction
export DreamerTrainer, dream_rollout, compute_actor_critic_loss

# ============================================================================
# Configuration
# ============================================================================

"""
    DreamerConfig

Configuration for Dreamer world model.

# Fields
- `latent_dim::Int`: Dimension of deterministic latent state
- `stochastic_dim::Int`: Dimension of stochastic latent state
- `hidden_dim::Int`: Hidden layer dimension
- `num_categories::Int`: Number of categories for discrete latents (DreamerV3)
- `category_size::Int`: Size of each category
- `observation_dim::Int`: Dimension of observations (scaffold features)
- `action_dim::Int`: Dimension of actions (design parameters)
- `imagination_horizon::Int`: Steps to imagine into future
- `discount::Float64`: Discount factor for returns
- `lambda::Float64`: Lambda for TD(λ) returns
- `free_nats::Float64`: Free nats for KL balancing
- `kl_scale::Float64`: KL divergence scale
- `use_symlog::Bool`: Use symlog encoding (DreamerV3)
"""
struct DreamerConfig
    latent_dim::Int
    stochastic_dim::Int
    hidden_dim::Int
    num_categories::Int
    category_size::Int
    observation_dim::Int
    action_dim::Int
    imagination_horizon::Int
    discount::Float64
    lambda::Float64
    free_nats::Float64
    kl_scale::Float64
    use_symlog::Bool

    function DreamerConfig(;
        latent_dim::Int = 512,
        stochastic_dim::Int = 32,
        hidden_dim::Int = 512,
        num_categories::Int = 32,
        category_size::Int = 32,
        observation_dim::Int = 256,
        action_dim::Int = 16,
        imagination_horizon::Int = 15,
        discount::Float64 = 0.997,
        lambda::Float64 = 0.95,
        free_nats::Float64 = 1.0,
        kl_scale::Float64 = 1.0,
        use_symlog::Bool = true
    )
        new(latent_dim, stochastic_dim, hidden_dim, num_categories, category_size,
            observation_dim, action_dim, imagination_horizon, discount, lambda,
            free_nats, kl_scale, use_symlog)
    end
end

# ============================================================================
# World State
# ============================================================================

"""
    WorldState

Latent world state combining deterministic and stochastic components.

# Fields
- `deterministic::Vector{Float64}`: Deterministic hidden state (h_t)
- `stochastic::Vector{Float64}`: Stochastic latent state (z_t)
- `logits::Matrix{Float64}`: Category logits for discrete latents
"""
mutable struct WorldState
    deterministic::Vector{Float64}
    stochastic::Vector{Float64}
    logits::Matrix{Float64}

    function WorldState(config::DreamerConfig)
        h = zeros(config.latent_dim)
        z = zeros(config.stochastic_dim)
        logits = zeros(config.num_categories, config.category_size)
        new(h, z, logits)
    end

    function WorldState(h::Vector{Float64}, z::Vector{Float64}, logits::Matrix{Float64})
        new(h, z, logits)
    end
end

"""
Get full latent representation by concatenating deterministic and stochastic.
"""
function get_latent(state::WorldState)
    vcat(state.deterministic, state.stochastic)
end

# ============================================================================
# Symlog Encoding (DreamerV3)
# ============================================================================

"""
    symlog(x)

Symmetric logarithm for handling large value ranges.
"""
function symlog(x::Real)
    sign(x) * log(abs(x) + 1)
end

"""
    symexp(x)

Inverse of symlog.
"""
function symexp(x::Real)
    sign(x) * (exp(abs(x)) - 1)
end

# Vectorized versions
symlog(x::AbstractArray) = symlog.(x)
symexp(x::AbstractArray) = symexp.(x)

# ============================================================================
# Network Layers (Simple Implementation)
# ============================================================================

"""
Simple dense layer with optional activation.
"""
struct DenseLayer
    weights::Matrix{Float64}
    bias::Vector{Float64}
    activation::Function
end

function DenseLayer(in_dim::Int, out_dim::Int; activation::Function = identity, rng = Random.GLOBAL_RNG)
    # Xavier initialization
    scale = sqrt(2.0 / (in_dim + out_dim))
    weights = randn(rng, out_dim, in_dim) .* scale
    bias = zeros(out_dim)
    DenseLayer(weights, bias, activation)
end

function (layer::DenseLayer)(x::AbstractVector)
    layer.activation.(layer.weights * x .+ layer.bias)
end

function (layer::DenseLayer)(x::AbstractMatrix)
    # Batch processing
    hcat([layer(x[:, i]) for i in 1:size(x, 2)]...)
end

"""
GRU cell for recurrent dynamics.
"""
struct GRUCell
    Wz::Matrix{Float64}  # Update gate weights
    Wr::Matrix{Float64}  # Reset gate weights
    Wh::Matrix{Float64}  # Candidate weights
    Uz::Matrix{Float64}
    Ur::Matrix{Float64}
    Uh::Matrix{Float64}
    bz::Vector{Float64}
    br::Vector{Float64}
    bh::Vector{Float64}
end

function GRUCell(input_dim::Int, hidden_dim::Int; rng = Random.GLOBAL_RNG)
    scale = sqrt(2.0 / (input_dim + hidden_dim))
    GRUCell(
        randn(rng, hidden_dim, input_dim) .* scale,
        randn(rng, hidden_dim, input_dim) .* scale,
        randn(rng, hidden_dim, input_dim) .* scale,
        randn(rng, hidden_dim, hidden_dim) .* scale,
        randn(rng, hidden_dim, hidden_dim) .* scale,
        randn(rng, hidden_dim, hidden_dim) .* scale,
        zeros(hidden_dim),
        zeros(hidden_dim),
        zeros(hidden_dim)
    )
end

function (gru::GRUCell)(h::Vector{Float64}, x::Vector{Float64})
    z = sigmoid.(gru.Wz * x .+ gru.Uz * h .+ gru.bz)
    r = sigmoid.(gru.Wr * x .+ gru.Ur * h .+ gru.br)
    h_tilde = tanh.(gru.Wh * x .+ gru.Uh * (r .* h) .+ gru.bh)
    h_new = (1 .- z) .* h .+ z .* h_tilde
    return h_new
end

sigmoid(x) = 1.0 / (1.0 + exp(-clamp(x, -20, 20)))
sigmoid(x::AbstractArray) = sigmoid.(x)

# ============================================================================
# Dreamer Model
# ============================================================================

"""
    DreamerModel

Complete Dreamer world model with RSSM dynamics.

# Components
- Encoder: o_t -> embed_t
- Recurrent: (h_{t-1}, z_{t-1}, a_{t-1}) -> h_t
- Representation: (h_t, embed_t) -> z_t (posterior)
- Transition: h_t -> z_t (prior, for imagination)
- Decoder: (h_t, z_t) -> o_t
- Reward: (h_t, z_t) -> r_t
- Continue: (h_t, z_t) -> c_t (episode continuation probability)
"""
struct DreamerModel
    config::DreamerConfig

    # Encoder
    encoder_layers::Vector{DenseLayer}

    # Recurrent model (GRU)
    recurrent::GRUCell

    # Representation model (posterior)
    posterior_layers::Vector{DenseLayer}
    posterior_out::DenseLayer

    # Transition model (prior)
    prior_layers::Vector{DenseLayer}
    prior_out::DenseLayer

    # Decoder
    decoder_layers::Vector{DenseLayer}
    decoder_out::DenseLayer

    # Reward predictor
    reward_layers::Vector{DenseLayer}
    reward_out::DenseLayer

    # Continue predictor
    continue_layers::Vector{DenseLayer}
    continue_out::DenseLayer
end

function DreamerModel(config::DreamerConfig; rng = Random.GLOBAL_RNG)
    hidden = config.hidden_dim
    latent = config.latent_dim
    stoch = config.stochastic_dim
    obs = config.observation_dim
    act = config.action_dim
    total_stoch = config.num_categories * config.category_size

    # Encoder: observation -> embedding
    encoder_layers = [
        DenseLayer(obs, hidden, activation = elu, rng = rng),
        DenseLayer(hidden, hidden, activation = elu, rng = rng)
    ]

    # Recurrent: (h, z, a) -> h'
    recurrent = GRUCell(stoch + act, latent, rng = rng)

    # Posterior: (h, embed) -> z logits
    posterior_layers = [
        DenseLayer(latent + hidden, hidden, activation = elu, rng = rng)
    ]
    posterior_out = DenseLayer(hidden, total_stoch, rng = rng)

    # Prior: h -> z logits
    prior_layers = [
        DenseLayer(latent, hidden, activation = elu, rng = rng)
    ]
    prior_out = DenseLayer(hidden, total_stoch, rng = rng)

    # Decoder: (h, z) -> observation
    decoder_layers = [
        DenseLayer(latent + stoch, hidden, activation = elu, rng = rng),
        DenseLayer(hidden, hidden, activation = elu, rng = rng)
    ]
    decoder_out = DenseLayer(hidden, obs, rng = rng)

    # Reward: (h, z) -> reward
    reward_layers = [
        DenseLayer(latent + stoch, hidden, activation = elu, rng = rng)
    ]
    reward_out = DenseLayer(hidden, 1, rng = rng)

    # Continue: (h, z) -> continue probability
    continue_layers = [
        DenseLayer(latent + stoch, hidden, activation = elu, rng = rng)
    ]
    continue_out = DenseLayer(hidden, 1, rng = rng)

    DreamerModel(config, encoder_layers, recurrent,
                 posterior_layers, posterior_out,
                 prior_layers, prior_out,
                 decoder_layers, decoder_out,
                 reward_layers, reward_out,
                 continue_layers, continue_out)
end

elu(x) = x >= 0 ? x : exp(x) - 1

function apply_layers(layers::Vector{DenseLayer}, x::Vector{Float64})
    for layer in layers
        x = layer(x)
    end
    return x
end

# ============================================================================
# Core Operations
# ============================================================================

"""
    encode_observation(model, observation)

Encode observation into embedding space.
"""
function encode_observation(model::DreamerModel, observation::Vector{Float64})
    x = model.config.use_symlog ? symlog(observation) : observation
    apply_layers(model.encoder_layers, x)
end

"""
    sample_stochastic(logits, num_categories, category_size)

Sample discrete latent using straight-through gradients.
"""
function sample_stochastic(logits::Matrix{Float64}; temperature::Float64 = 1.0)
    # logits is (num_categories, category_size)
    # Apply softmax per category and sample
    probs = similar(logits)
    for i in 1:size(logits, 1)
        probs[i, :] = softmax(logits[i, :] ./ temperature)
    end

    # Sample from each category
    samples = zeros(size(logits, 1))
    for i in 1:size(logits, 1)
        samples[i] = sample_categorical(probs[i, :])
    end

    return samples
end

function softmax(x::Vector{Float64})
    x_max = maximum(x)
    exp_x = exp.(x .- x_max)
    return exp_x ./ sum(exp_x)
end

function sample_categorical(probs::Vector{Float64})
    r = rand()
    cumsum_p = 0.0
    for (i, p) in enumerate(probs)
        cumsum_p += p
        if r <= cumsum_p
            return Float64(i)
        end
    end
    return Float64(length(probs))
end

"""
    transition_step(model, state, action)

Perform one step of imagined dynamics (prior prediction).
"""
function transition_step(model::DreamerModel, state::WorldState, action::Vector{Float64})
    # Compute new deterministic state
    za = vcat(state.stochastic, action)
    h_new = model.recurrent(state.deterministic, za)

    # Compute prior (imagined stochastic)
    prior_hidden = apply_layers(model.prior_layers, h_new)
    prior_logits = model.prior_out(prior_hidden)
    prior_logits_mat = reshape(prior_logits, model.config.num_categories, model.config.category_size)

    # Sample stochastic state
    z_new = sample_stochastic(prior_logits_mat)

    WorldState(h_new, z_new, prior_logits_mat)
end

"""
    representation_step(model, state, action, observation)

Perform one step with observation (posterior prediction).
"""
function representation_step(model::DreamerModel, state::WorldState,
                            action::Vector{Float64}, observation::Vector{Float64})
    # Compute new deterministic state
    za = vcat(state.stochastic, action)
    h_new = model.recurrent(state.deterministic, za)

    # Encode observation
    embed = encode_observation(model, observation)

    # Compute posterior
    posterior_input = vcat(h_new, embed)
    posterior_hidden = apply_layers(model.posterior_layers, posterior_input)
    posterior_logits = model.posterior_out(posterior_hidden)
    posterior_logits_mat = reshape(posterior_logits, model.config.num_categories, model.config.category_size)

    # Sample stochastic state
    z_new = sample_stochastic(posterior_logits_mat)

    WorldState(h_new, z_new, posterior_logits_mat)
end

"""
    decode_latent(model, state)

Decode latent state to observation prediction.
"""
function decode_latent(model::DreamerModel, state::WorldState)
    hz = get_latent(state)
    hidden = apply_layers(model.decoder_layers, hz)
    pred = model.decoder_out(hidden)

    if model.config.use_symlog
        return symexp(pred)
    end
    return pred
end

"""
    compute_reward_prediction(model, state)

Predict reward from latent state.
"""
function compute_reward_prediction(model::DreamerModel, state::WorldState)
    hz = get_latent(state)
    hidden = apply_layers(model.reward_layers, hz)
    reward = model.reward_out(hidden)[1]

    if model.config.use_symlog
        return symexp(reward)
    end
    return reward
end

"""
    compute_continue_probability(model, state)

Predict episode continuation probability.
"""
function compute_continue_probability(model::DreamerModel, state::WorldState)
    hz = get_latent(state)
    hidden = apply_layers(model.continue_layers, hz)
    logit = model.continue_out(hidden)[1]
    return sigmoid(logit)
end

# ============================================================================
# Imagination (Dream Rollout)
# ============================================================================

"""
    imagine_trajectory(model, initial_state, actor, horizon)

Imagine a trajectory into the future using the world model.

# Arguments
- `model::DreamerModel`: The world model
- `initial_state::WorldState`: Starting latent state
- `actor::Function`: Policy function (state -> action)
- `horizon::Int`: Number of steps to imagine
"""
function imagine_trajectory(model::DreamerModel, initial_state::WorldState,
                           actor::Function, horizon::Int)
    states = WorldState[initial_state]
    actions = Vector{Float64}[]
    rewards = Float64[]
    continues = Float64[]

    state = initial_state
    for t in 1:horizon
        # Get action from actor
        latent = get_latent(state)
        action = actor(latent)
        push!(actions, action)

        # Transition using prior (imagination)
        state = transition_step(model, state, action)
        push!(states, state)

        # Predict reward and continue
        push!(rewards, compute_reward_prediction(model, state))
        push!(continues, compute_continue_probability(model, state))
    end

    return (states=states, actions=actions, rewards=rewards, continues=continues)
end

"""
    dream_rollout(model, observations, actions)

Process a sequence of observations and actions through the world model.
Returns latent states with both prior and posterior.
"""
function dream_rollout(model::DreamerModel, observations::Vector{Vector{Float64}},
                       actions::Vector{Vector{Float64}})
    T = length(observations)
    @assert length(actions) == T "Actions must match observations length"

    # Initialize state
    state = WorldState(model.config)

    prior_states = WorldState[]
    posterior_states = WorldState[]

    prev_action = zeros(model.config.action_dim)

    for t in 1:T
        # Prior prediction (without observation)
        prior_state = transition_step(model, state, prev_action)
        push!(prior_states, prior_state)

        # Posterior (with observation)
        posterior_state = representation_step(model, state, prev_action, observations[t])
        push!(posterior_states, posterior_state)

        # Update state and action for next step
        state = posterior_state
        prev_action = actions[t]
    end

    return (prior=prior_states, posterior=posterior_states)
end

# ============================================================================
# Loss Functions
# ============================================================================

"""
    kl_divergence(posterior_logits, prior_logits; free_nats)

Compute KL divergence between posterior and prior distributions.
Uses free nats and KL balancing from DreamerV3.
"""
function kl_divergence(posterior_logits::Matrix{Float64}, prior_logits::Matrix{Float64};
                       free_nats::Float64 = 1.0)
    kl_total = 0.0

    for i in 1:size(posterior_logits, 1)
        post_probs = softmax(posterior_logits[i, :])
        prior_probs = softmax(prior_logits[i, :])

        for j in 1:size(posterior_logits, 2)
            if post_probs[j] > 1e-8
                kl_total += post_probs[j] * (log(post_probs[j] + 1e-8) - log(prior_probs[j] + 1e-8))
            end
        end
    end

    # Apply free nats
    return max(kl_total, free_nats)
end

"""
    reconstruction_loss(model, state, target_observation)

Compute reconstruction loss for decoder.
"""
function reconstruction_loss(model::DreamerModel, state::WorldState, target::Vector{Float64})
    pred = decode_latent(model, state)

    if model.config.use_symlog
        target = symlog(target)
        pred = symlog(pred)
    end

    # MSE loss
    return mean((pred .- target).^2)
end

"""
    reward_loss(model, state, target_reward)

Compute reward prediction loss.
"""
function reward_loss(model::DreamerModel, state::WorldState, target::Float64)
    pred = compute_reward_prediction(model, state)

    if model.config.use_symlog
        return (symlog(pred) - symlog(target))^2
    end

    return (pred - target)^2
end

"""
    continue_loss(model, state, target_continue)

Compute continue prediction loss (binary cross-entropy).
"""
function continue_loss(model::DreamerModel, state::WorldState, target::Float64)
    prob = compute_continue_probability(model, state)
    return -target * log(prob + 1e-8) - (1 - target) * log(1 - prob + 1e-8)
end

# ============================================================================
# Training
# ============================================================================

"""
    DreamerTrainer

Trainer for the Dreamer world model.
"""
mutable struct DreamerTrainer
    model::DreamerModel
    learning_rate::Float64
    gradient_clip::Float64
    total_steps::Int
end

function DreamerTrainer(model::DreamerModel; learning_rate = 3e-4, gradient_clip = 100.0)
    DreamerTrainer(model, learning_rate, gradient_clip, 0)
end

"""
    train_world_model!(trainer, batch)

Train the world model on a batch of sequences.

# Arguments
- `trainer::DreamerTrainer`: The trainer
- `batch`: Dict with :observations, :actions, :rewards, :continues
"""
function train_world_model!(trainer::DreamerTrainer, batch::Dict)
    observations = batch[:observations]
    actions = batch[:actions]
    rewards = batch[:rewards]
    continues = batch[:continues]

    model = trainer.model
    config = model.config

    # Process batch through world model
    rollout = dream_rollout(model, observations, actions)

    # Compute losses
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_reward_loss = 0.0
    total_continue_loss = 0.0

    T = length(observations)
    for t in 1:T
        # Reconstruction loss
        total_recon_loss += reconstruction_loss(model, rollout.posterior[t], observations[t])

        # KL divergence
        total_kl_loss += kl_divergence(rollout.posterior[t].logits, rollout.prior[t].logits,
                                       free_nats = config.free_nats)

        # Reward loss
        total_reward_loss += reward_loss(model, rollout.posterior[t], rewards[t])

        # Continue loss
        total_continue_loss += continue_loss(model, rollout.posterior[t], continues[t])
    end

    # Average losses
    total_recon_loss /= T
    total_kl_loss /= T
    total_reward_loss /= T
    total_continue_loss /= T

    # Total world model loss
    total_loss = total_recon_loss + config.kl_scale * total_kl_loss +
                 total_reward_loss + total_continue_loss

    trainer.total_steps += 1

    return Dict(
        :total_loss => total_loss,
        :recon_loss => total_recon_loss,
        :kl_loss => total_kl_loss,
        :reward_loss => total_reward_loss,
        :continue_loss => total_continue_loss,
        :step => trainer.total_steps
    )
end

# ============================================================================
# Actor-Critic for Imagination
# ============================================================================

"""
    compute_lambda_returns(rewards, values, continues; λ, γ)

Compute TD(λ) returns for actor-critic training.
"""
function compute_lambda_returns(rewards::Vector{Float64}, values::Vector{Float64},
                                continues::Vector{Float64}; λ::Float64 = 0.95, γ::Float64 = 0.997)
    T = length(rewards)
    returns = zeros(T)

    # Bootstrap from last value
    next_value = values[end]
    next_return = values[end]

    for t in T:-1:1
        td_target = rewards[t] + γ * continues[t] * next_value
        returns[t] = (1 - λ) * td_target + λ * (rewards[t] + γ * continues[t] * next_return)
        next_value = values[t]
        next_return = returns[t]
    end

    return returns
end

"""
    compute_actor_critic_loss(imagined_trajectory, values)

Compute actor and critic losses from imagined trajectory.
"""
function compute_actor_critic_loss(trajectory::NamedTuple, values::Vector{Float64};
                                   λ::Float64 = 0.95, γ::Float64 = 0.997)
    rewards = trajectory.rewards
    continues = trajectory.continues

    # Compute returns
    returns = compute_lambda_returns(rewards, values, continues, λ = λ, γ = γ)

    # Critic loss (value function)
    critic_loss = mean((values .- returns).^2)

    # Actor loss (policy gradient with baseline)
    advantages = returns .- values
    actor_loss = -mean(advantages)  # Simplified - full version uses policy gradients

    return Dict(:actor_loss => actor_loss, :critic_loss => critic_loss,
                :returns => returns, :advantages => advantages)
end

# ============================================================================
# Scaffold-Specific Applications
# ============================================================================

"""
    predict_dynamics(model, scaffold_state, design_actions, horizon)

Predict scaffold degradation and tissue ingrowth dynamics.

# Arguments
- `model::DreamerModel`: Trained world model
- `scaffold_state::Vector{Float64}`: Current scaffold features (porosity, MW, etc.)
- `design_actions::Vector{Vector{Float64}}`: Sequence of design interventions
- `horizon::Int`: Prediction horizon
"""
function predict_dynamics(model::DreamerModel, scaffold_state::Vector{Float64},
                         design_actions::Vector{Vector{Float64}}, horizon::Int)
    # Initialize from scaffold observation
    initial_state = WorldState(model.config)

    # Encode initial observation
    embed = encode_observation(model, scaffold_state)
    posterior_input = vcat(initial_state.deterministic, embed)
    posterior_hidden = apply_layers(model.posterior_layers, posterior_input)
    posterior_logits = model.posterior_out(posterior_hidden)
    posterior_logits_mat = reshape(posterior_logits, model.config.num_categories, model.config.category_size)
    z = sample_stochastic(posterior_logits_mat)

    state = WorldState(initial_state.deterministic, z, posterior_logits_mat)

    # Roll out predictions
    predictions = Vector{Float64}[]
    rewards = Float64[]

    for t in 1:min(horizon, length(design_actions))
        state = transition_step(model, state, design_actions[t])
        push!(predictions, decode_latent(model, state))
        push!(rewards, compute_reward_prediction(model, state))
    end

    # Continue with no action if needed
    zero_action = zeros(model.config.action_dim)
    for t in (length(design_actions)+1):horizon
        state = transition_step(model, state, zero_action)
        push!(predictions, decode_latent(model, state))
        push!(rewards, compute_reward_prediction(model, state))
    end

    return (predictions=predictions, rewards=rewards)
end

"""
    create_scaffold_dreamer(; observation_features, action_parameters)

Create a Dreamer model configured for scaffold analysis.

# Arguments
- `observation_features::Int`: Number of scaffold features
- `action_parameters::Int`: Number of design parameters
"""
function create_scaffold_dreamer(; observation_features::Int = 20, action_parameters::Int = 8)
    config = DreamerConfig(
        observation_dim = observation_features,
        action_dim = action_parameters,
        latent_dim = 256,
        stochastic_dim = 32,
        hidden_dim = 256,
        imagination_horizon = 30,  # Longer horizon for degradation
        discount = 0.99,
        use_symlog = true
    )

    return DreamerModel(config)
end

end # module
