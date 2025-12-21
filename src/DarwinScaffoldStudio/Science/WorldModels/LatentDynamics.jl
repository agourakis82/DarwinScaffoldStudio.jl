# LatentDynamics.jl - Neural ODEs for Scaffold Evolution
# Continuous-time dynamics modeling in latent space
# Based on Neural ODEs (Chen et al. 2018) and Latent ODEs (Rubanova et al. 2019)

module LatentDynamics

using LinearAlgebra
using Statistics
using Random

export LatentODEConfig, LatentODEModel, NeuralODEFunc
export encode_trajectory, decode_trajectory, integrate_ode
export ODEIntegrator, euler_step, rk4_step, dopri5_step
export LatentODETrainer, train_latent_ode!, sample_latent
export create_scaffold_dynamics_model, predict_scaffold_evolution

# ============================================================================
# Configuration
# ============================================================================

"""
    LatentODEConfig

Configuration for Latent ODE model.

# Fields
- `input_dim::Int`: Dimension of input observations
- `latent_dim::Int`: Dimension of latent space
- `hidden_dim::Int`: Hidden layer dimension for networks
- `encoder_hidden::Int`: Hidden dimension for encoder RNN
- `ode_hidden::Int`: Hidden dimension for ODE function network
- `output_dim::Int`: Dimension of output (can differ from input)
- `integration_method::Symbol`: ODE integration method (:euler, :rk4, :dopri5)
- `dt::Float64`: Default time step for integration
- `adjoint::Bool`: Use adjoint sensitivity method
- `use_attention::Bool`: Use attention mechanism in encoder
"""
struct LatentODEConfig
    input_dim::Int
    latent_dim::Int
    hidden_dim::Int
    encoder_hidden::Int
    ode_hidden::Int
    output_dim::Int
    integration_method::Symbol
    dt::Float64
    adjoint::Bool
    use_attention::Bool

    function LatentODEConfig(;
        input_dim::Int = 20,
        latent_dim::Int = 32,
        hidden_dim::Int = 128,
        encoder_hidden::Int = 64,
        ode_hidden::Int = 64,
        output_dim::Int = 20,
        integration_method::Symbol = :rk4,
        dt::Float64 = 0.01,
        adjoint::Bool = false,
        use_attention::Bool = false
    )
        new(input_dim, latent_dim, hidden_dim, encoder_hidden, ode_hidden,
            output_dim, integration_method, dt, adjoint, use_attention)
    end
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

elu(x) = x >= 0 ? x : exp(x) - 1
softplus(x) = log(1 + exp(clamp(x, -20, 20)))
tanh_act(x) = tanh(x)
sigmoid(x) = 1.0 / (1.0 + exp(-clamp(x, -20, 20)))

# ============================================================================
# GRU Cell for Encoder
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
    hidden_dim::Int
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
        zeros(hidden_dim),
        hidden_dim
    )
end

function (gru::GRUCell)(h::Vector{Float64}, x::Vector{Float64})
    z = sigmoid.(gru.Wz * x .+ gru.Uz * h .+ gru.bz)
    r = sigmoid.(gru.Wr * x .+ gru.Ur * h .+ gru.br)
    h_tilde = tanh.(gru.Wh * x .+ gru.Uh * (r .* h) .+ gru.bh)
    return (1 .- z) .* h .+ z .* h_tilde
end

# ============================================================================
# Neural ODE Function
# ============================================================================

"""
    NeuralODEFunc

Neural network that defines the ODE dynamics: dz/dt = f(z, t)
"""
struct NeuralODEFunc
    layers::Vector{DenseLayer}
    use_time::Bool
end

function NeuralODEFunc(latent_dim::Int, hidden_dim::Int;
                       n_layers::Int = 3, use_time::Bool = true,
                       rng = Random.GLOBAL_RNG)
    input_dim = use_time ? latent_dim + 1 : latent_dim

    layers = DenseLayer[]

    # Input layer
    push!(layers, DenseLayer(input_dim, hidden_dim, activation = elu, rng = rng))

    # Hidden layers
    for _ in 2:(n_layers-1)
        push!(layers, DenseLayer(hidden_dim, hidden_dim, activation = elu, rng = rng))
    end

    # Output layer (no activation - output is dz/dt)
    push!(layers, DenseLayer(hidden_dim, latent_dim, rng = rng))

    NeuralODEFunc(layers, use_time)
end

function (f::NeuralODEFunc)(z::Vector{Float64}, t::Float64)
    x = f.use_time ? vcat(z, t) : z

    for layer in f.layers
        x = layer(x)
    end

    return x
end

# ============================================================================
# ODE Integrators
# ============================================================================

"""
    ODEIntegrator

Wrapper for ODE integration methods.
"""
struct ODEIntegrator
    method::Symbol
    dt::Float64
    rtol::Float64
    atol::Float64
end

function ODEIntegrator(method::Symbol = :rk4; dt::Float64 = 0.01,
                       rtol::Float64 = 1e-6, atol::Float64 = 1e-6)
    ODEIntegrator(method, dt, rtol, atol)
end

"""
    euler_step(f, z, t, dt)

Single Euler integration step.
"""
function euler_step(f::NeuralODEFunc, z::Vector{Float64}, t::Float64, dt::Float64)
    dz = f(z, t)
    return z .+ dt .* dz
end

"""
    rk4_step(f, z, t, dt)

Single Runge-Kutta 4th order step.
"""
function rk4_step(f::NeuralODEFunc, z::Vector{Float64}, t::Float64, dt::Float64)
    k1 = f(z, t)
    k2 = f(z .+ 0.5 * dt .* k1, t + 0.5 * dt)
    k3 = f(z .+ 0.5 * dt .* k2, t + 0.5 * dt)
    k4 = f(z .+ dt .* k3, t + dt)

    return z .+ (dt / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

"""
    dopri5_step(f, z, t, dt)

Single Dormand-Prince 5th order step (for adaptive stepping).
Returns (z_new, error_estimate).
"""
function dopri5_step(f::NeuralODEFunc, z::Vector{Float64}, t::Float64, dt::Float64)
    # Butcher tableau coefficients for DOPRI5
    a21 = 1/5
    a31, a32 = 3/40, 9/40
    a41, a42, a43 = 44/45, -56/15, 32/9
    a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    a71, a72, a73, a74, a75, a76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84

    c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1

    # 5th order weights
    b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

    # 4th order weights (for error estimation)
    e1, e3, e4, e5, e6, e7 = 71/57600, -71/16695, 71/1920, -17253/339200, 22/525, -1/40

    k1 = f(z, t)
    k2 = f(z .+ dt * a21 .* k1, t + c2 * dt)
    k3 = f(z .+ dt .* (a31 .* k1 .+ a32 .* k2), t + c3 * dt)
    k4 = f(z .+ dt .* (a41 .* k1 .+ a42 .* k2 .+ a43 .* k3), t + c4 * dt)
    k5 = f(z .+ dt .* (a51 .* k1 .+ a52 .* k2 .+ a53 .* k3 .+ a54 .* k4), t + c5 * dt)
    k6 = f(z .+ dt .* (a61 .* k1 .+ a62 .* k2 .+ a63 .* k3 .+ a64 .* k4 .+ a65 .* k5), t + c6 * dt)

    # 5th order solution
    z_new = z .+ dt .* (b1 .* k1 .+ b3 .* k3 .+ b4 .* k4 .+ b5 .* k5 .+ b6 .* k6)

    # Error estimate
    k7 = f(z_new, t + dt)
    err = dt .* (e1 .* k1 .+ e3 .* k3 .+ e4 .* k4 .+ e5 .* k5 .+ e6 .* k6 .+ e7 .* k7)
    error_norm = sqrt(mean(err.^2))

    return (z_new, error_norm)
end

"""
    integrate_ode(f, z0, t_span, integrator)

Integrate ODE from z0 over time span.
"""
function integrate_ode(f::NeuralODEFunc, z0::Vector{Float64},
                       t_span::Tuple{Float64, Float64}, integrator::ODEIntegrator)
    t0, tf = t_span
    dt = integrator.dt

    # Number of steps
    n_steps = ceil(Int, (tf - t0) / dt)
    dt = (tf - t0) / n_steps  # Adjust dt to hit exact endpoint

    trajectory = Vector{Float64}[z0]
    times = Float64[t0]

    z = z0
    t = t0

    for _ in 1:n_steps
        if integrator.method == :euler
            z = euler_step(f, z, t, dt)
        elseif integrator.method == :rk4
            z = rk4_step(f, z, t, dt)
        elseif integrator.method == :dopri5
            z, _ = dopri5_step(f, z, t, dt)
        end

        t += dt
        push!(trajectory, z)
        push!(times, t)
    end

    return (trajectory = trajectory, times = times)
end

"""
    integrate_ode_to_times(f, z0, t0, target_times, integrator)

Integrate ODE to specific target times.
"""
function integrate_ode_to_times(f::NeuralODEFunc, z0::Vector{Float64}, t0::Float64,
                                target_times::Vector{Float64}, integrator::ODEIntegrator)
    trajectory = Vector{Float64}[]
    z = z0
    current_t = t0

    for target_t in target_times
        if target_t <= current_t
            push!(trajectory, z)
            continue
        end

        result = integrate_ode(f, z, (current_t, target_t), integrator)
        z = result.trajectory[end]
        current_t = target_t
        push!(trajectory, z)
    end

    return trajectory
end

# ============================================================================
# Latent ODE Model
# ============================================================================

"""
    LatentODEModel

Complete Latent ODE model with encoder, ODE dynamics, and decoder.

# Components
- Encoder: Processes observations backward in time to produce initial latent
- ODE Function: Defines latent dynamics dz/dt = f(z, t)
- Decoder: Maps latent states to observations
"""
struct LatentODEModel
    config::LatentODEConfig

    # Encoder (reverse-time GRU)
    encoder_gru::GRUCell
    encoder_fc::DenseLayer
    latent_mean::DenseLayer
    latent_logvar::DenseLayer

    # ODE dynamics
    ode_func::NeuralODEFunc
    integrator::ODEIntegrator

    # Decoder
    decoder_layers::Vector{DenseLayer}
    decoder_out::DenseLayer
end

function LatentODEModel(config::LatentODEConfig; rng = Random.GLOBAL_RNG)
    # Encoder
    encoder_gru = GRUCell(config.input_dim + 1, config.encoder_hidden, rng = rng)  # +1 for time
    encoder_fc = DenseLayer(config.encoder_hidden, config.hidden_dim, activation = elu, rng = rng)
    latent_mean = DenseLayer(config.hidden_dim, config.latent_dim, rng = rng)
    latent_logvar = DenseLayer(config.hidden_dim, config.latent_dim, rng = rng)

    # ODE function
    ode_func = NeuralODEFunc(config.latent_dim, config.ode_hidden, use_time = true, rng = rng)
    integrator = ODEIntegrator(config.integration_method, dt = config.dt)

    # Decoder
    decoder_layers = [
        DenseLayer(config.latent_dim, config.hidden_dim, activation = elu, rng = rng),
        DenseLayer(config.hidden_dim, config.hidden_dim, activation = elu, rng = rng)
    ]
    decoder_out = DenseLayer(config.hidden_dim, config.output_dim, rng = rng)

    LatentODEModel(config, encoder_gru, encoder_fc, latent_mean, latent_logvar,
                   ode_func, integrator, decoder_layers, decoder_out)
end

"""
    encode_trajectory(model, observations, times)

Encode a trajectory of observations to initial latent distribution.
Processes observations in reverse time (ODE-RNN style).
"""
function encode_trajectory(model::LatentODEModel, observations::Vector{Vector{Float64}},
                          times::Vector{Float64})
    T = length(observations)
    @assert length(times) == T

    # Initialize hidden state
    h = zeros(model.encoder_gru.hidden_dim)

    # Process backwards in time
    for t in T:-1:1
        # Input is observation concatenated with time
        x = vcat(observations[t], times[t])
        h = model.encoder_gru(h, x)
    end

    # Map to latent distribution parameters
    h = model.encoder_fc(h)
    mean = model.latent_mean(h)
    logvar = model.latent_logvar(h)

    return (mean = mean, logvar = logvar)
end

"""
    sample_latent(mean, logvar)

Sample from latent distribution using reparameterization trick.
"""
function sample_latent(mean::Vector{Float64}, logvar::Vector{Float64})
    std = exp.(0.5 .* logvar)
    eps = randn(length(mean))
    return mean .+ std .* eps
end

"""
    decode_latent(model, z)

Decode latent state to observation.
"""
function decode_latent(model::LatentODEModel, z::Vector{Float64})
    x = z
    for layer in model.decoder_layers
        x = layer(x)
    end
    model.decoder_out(x)
end

"""
    decode_trajectory(model, latent_trajectory)

Decode a trajectory of latent states to observations.
"""
function decode_trajectory(model::LatentODEModel, latent_trajectory::Vector{Vector{Float64}})
    [decode_latent(model, z) for z in latent_trajectory]
end

"""
    forward(model, observations, times, target_times)

Full forward pass: encode, integrate, decode.
"""
function forward(model::LatentODEModel, observations::Vector{Vector{Float64}},
                 times::Vector{Float64}, target_times::Vector{Float64})
    # Encode to initial latent
    params = encode_trajectory(model, observations, times)
    z0 = sample_latent(params.mean, params.logvar)

    # Integrate through time
    t0 = times[1]
    latent_trajectory = integrate_ode_to_times(model.ode_func, z0, t0,
                                                target_times, model.integrator)

    # Decode to observations
    pred_observations = decode_trajectory(model, latent_trajectory)

    return (predictions = pred_observations,
            latent_trajectory = latent_trajectory,
            z0 = z0,
            mean = params.mean,
            logvar = params.logvar)
end

# ============================================================================
# Loss Functions
# ============================================================================

"""
    reconstruction_loss(predictions, targets)

Mean squared error reconstruction loss.
"""
function reconstruction_loss(predictions::Vector{Vector{Float64}},
                            targets::Vector{Vector{Float64}})
    total = 0.0
    for (pred, target) in zip(predictions, targets)
        total += mean((pred .- target).^2)
    end
    return total / length(predictions)
end

"""
    kl_divergence(mean, logvar)

KL divergence from N(mean, exp(logvar)) to N(0, 1).
"""
function kl_divergence(mean::Vector{Float64}, logvar::Vector{Float64})
    return -0.5 * sum(1 .+ logvar .- mean.^2 .- exp.(logvar))
end

"""
    elbo_loss(model, observations, times, target_times, target_observations; β)

Evidence lower bound loss with β-VAE weighting.
"""
function elbo_loss(model::LatentODEModel, observations::Vector{Vector{Float64}},
                   times::Vector{Float64}, target_times::Vector{Float64},
                   target_observations::Vector{Vector{Float64}}; β::Float64 = 1.0)
    result = forward(model, observations, times, target_times)

    recon_loss = reconstruction_loss(result.predictions, target_observations)
    kl_loss = kl_divergence(result.mean, result.logvar)

    return recon_loss + β * kl_loss, Dict(:recon => recon_loss, :kl => kl_loss)
end

# ============================================================================
# Training
# ============================================================================

"""
    LatentODETrainer

Trainer for Latent ODE models.
"""
mutable struct LatentODETrainer
    model::LatentODEModel
    learning_rate::Float64
    β::Float64  # KL weight
    total_steps::Int
    best_loss::Float64
end

function LatentODETrainer(model::LatentODEModel;
                          learning_rate::Float64 = 1e-3, β::Float64 = 1.0)
    LatentODETrainer(model, learning_rate, β, 0, Inf)
end

"""
    train_latent_ode!(trainer, batch)

Train on a batch of trajectories.

# Arguments
- `trainer::LatentODETrainer`: The trainer
- `batch`: Dict with :observations, :times, :target_times, :targets
"""
function train_latent_ode!(trainer::LatentODETrainer, batch::Dict)
    observations = batch[:observations]
    times = batch[:times]
    target_times = batch[:target_times]
    targets = batch[:targets]

    loss, components = elbo_loss(trainer.model, observations, times,
                                 target_times, targets, β = trainer.β)

    trainer.total_steps += 1
    if loss < trainer.best_loss
        trainer.best_loss = loss
    end

    return Dict(
        :loss => loss,
        :recon_loss => components[:recon],
        :kl_loss => components[:kl],
        :step => trainer.total_steps,
        :best_loss => trainer.best_loss
    )
end

# ============================================================================
# Scaffold-Specific Functions
# ============================================================================

"""
    create_scaffold_dynamics_model(; feature_dim, latent_dim)

Create a Latent ODE model configured for scaffold degradation dynamics.
"""
function create_scaffold_dynamics_model(; feature_dim::Int = 20, latent_dim::Int = 16,
                                        rng = Random.GLOBAL_RNG)
    config = LatentODEConfig(
        input_dim = feature_dim,
        latent_dim = latent_dim,
        hidden_dim = 64,
        encoder_hidden = 32,
        ode_hidden = 32,
        output_dim = feature_dim,
        integration_method = :rk4,
        dt = 0.1,  # Day-scale for scaffold degradation
        adjoint = false,
        use_attention = false
    )

    return LatentODEModel(config, rng = rng)
end

"""
    predict_scaffold_evolution(model, initial_state, times; n_samples)

Predict scaffold feature evolution over time.

# Arguments
- `model::LatentODEModel`: Trained model
- `initial_state::Vector{Float64}`: Initial scaffold features
- `times::Vector{Float64}`: Target prediction times (in days)
- `n_samples::Int`: Number of samples for uncertainty estimation
"""
function predict_scaffold_evolution(model::LatentODEModel, initial_state::Vector{Float64},
                                   times::Vector{Float64}; n_samples::Int = 10)
    # Encode initial state
    params = encode_trajectory(model, [initial_state], [0.0])

    predictions = Vector{Vector{Float64}}[]
    latent_trajectories = Vector{Vector{Float64}}[]

    for _ in 1:n_samples
        # Sample initial latent
        z0 = sample_latent(params.mean, params.logvar)

        # Integrate
        latent_traj = integrate_ode_to_times(model.ode_func, z0, 0.0,
                                             times, model.integrator)
        push!(latent_trajectories, latent_traj)

        # Decode
        push!(predictions, decode_trajectory(model, latent_traj))
    end

    # Compute mean and std
    n_times = length(times)
    mean_predictions = Vector{Float64}[]
    std_predictions = Vector{Float64}[]

    for t in 1:n_times
        samples_at_t = [predictions[s][t] for s in 1:n_samples]
        mean_pred = mean(hcat(samples_at_t...), dims=2)[:, 1]
        std_pred = std(hcat(samples_at_t...), dims=2)[:, 1]
        push!(mean_predictions, mean_pred)
        push!(std_predictions, std_pred)
    end

    return Dict(
        :mean => mean_predictions,
        :std => std_predictions,
        :samples => predictions,
        :latent_trajectories => latent_trajectories,
        :times => times
    )
end

"""
    ScaffoldDynamicsFeatures

Standard scaffold features for dynamics modeling.
"""
const SCAFFOLD_FEATURES = [
    :molecular_weight,
    :crystallinity,
    :porosity,
    :pore_size_mean,
    :pore_size_std,
    :interconnectivity,
    :surface_area,
    :water_uptake,
    :mass_remaining,
    :tensile_modulus,
    :compressive_modulus,
    :degradation_rate,
    :pH_local,
    :cell_density,
    :tissue_volume,
    :mineralization,
    :vascularization,
    :inflammation_score,
    :collagen_content,
    :GAG_content
]

"""
    prepare_scaffold_timeseries(data::Dict, feature_names)

Prepare scaffold time series data for Latent ODE training.
"""
function prepare_scaffold_timeseries(data::Dict; feature_names = SCAFFOLD_FEATURES)
    times = data[:times]
    T = length(times)

    observations = Vector{Float64}[]
    for t in 1:T
        obs = Float64[]
        for feat in feature_names
            if haskey(data, feat)
                push!(obs, data[feat][t])
            else
                push!(obs, 0.0)
            end
        end
        push!(observations, obs)
    end

    return (observations = observations, times = times)
end

end # module
