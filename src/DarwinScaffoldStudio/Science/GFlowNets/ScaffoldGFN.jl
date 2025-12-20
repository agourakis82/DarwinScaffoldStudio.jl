"""
ScaffoldGFN.jl - GFlowNet for Diverse Scaffold Generation

Applies GFlowNets to tissue engineering scaffold design, generating
diverse scaffolds that satisfy multiple constraints (porosity, pore size,
mechanical properties, permeability).

SOTA 2024-2025 Features:
- Multi-objective scaffold optimization
- Constraint satisfaction (biological + mechanical)
- Diverse solution generation (not just one optimum)
- Hierarchical scaffold construction
- Integration with TPMS surfaces

Key Advantage:
GFlowNets sample proportionally to reward P(x) ∝ R(x), meaning:
- High-reward scaffolds are sampled more often
- But medium-reward scaffolds are ALSO sampled
- This gives diverse valid designs for experimental validation

References:
- Bengio et al. 2021: "GFlowNets"
- Jain et al. 2023: "GFlowNets for Molecule Generation"
- Murphy et al. 2010: Scaffold design criteria
"""
module ScaffoldGFN

using LinearAlgebra
using Statistics
using Random

# Import GFlowNet - handles both standalone and integrated loading
if !@isdefined(GFlowNet)
    include("GFlowNet.jl")
end
using .GFlowNet

export ScaffoldGFNConfig, ScaffoldGFNTrainer
export ScaffoldState, ScaffoldAction
export generate_diverse_scaffolds, optimize_scaffold_gfn
export reward_scaffold, create_scaffold_env
export TPMSType, GridCell, ScaffoldGrid

# ============================================================================
# CONFIGURATION (must be defined before use in functions)
# ============================================================================

"""
    ScaffoldGFNConfig

Configuration for scaffold-specific GFlowNet.
"""
struct ScaffoldGFNConfig
    # Grid settings
    grid_resolution::NTuple{3,Int}
    physical_size::NTuple{3,Float64}

    # Target properties
    target_porosity::Float64         # 0-1
    target_pore_size::Float64        # μm
    target_modulus::Float64          # MPa
    target_interconnectivity::Float64  # 0-1

    # Property tolerances
    porosity_tolerance::Float64
    pore_size_tolerance::Float64
    modulus_tolerance::Float64

    # GFlowNet settings
    state_dim::Int
    action_dim::Int
    hidden_dim::Int
    num_layers::Int
    learning_rate::Float64

    # Reward shaping
    reward_sharpness::Float64  # Higher = more peaked rewards
    constraint_penalty::Float64
end

function ScaffoldGFNConfig(;
    grid_resolution::NTuple{3,Int}=(4, 4, 4),
    physical_size::NTuple{3,Float64}=(10.0, 10.0, 10.0),
    target_porosity::Float64=0.85,
    target_pore_size::Float64=200.0,
    target_modulus::Float64=5.0,
    target_interconnectivity::Float64=0.95,
    porosity_tolerance::Float64=0.05,
    pore_size_tolerance::Float64=50.0,
    modulus_tolerance::Float64=2.0,
    state_dim::Int=64,
    action_dim::Int=17,
    hidden_dim::Int=128,
    num_layers::Int=3,
    learning_rate::Float64=1e-3,
    reward_sharpness::Float64=2.0,
    constraint_penalty::Float64=0.1
)
    ScaffoldGFNConfig(
        grid_resolution, physical_size,
        target_porosity, target_pore_size, target_modulus, target_interconnectivity,
        porosity_tolerance, pore_size_tolerance, modulus_tolerance,
        state_dim, action_dim, hidden_dim, num_layers, learning_rate,
        reward_sharpness, constraint_penalty
    )
end

# ============================================================================
# SCAFFOLD REPRESENTATION
# ============================================================================

"""
    TPMSType

Types of TPMS surfaces for scaffold generation.
"""
@enum TPMSType begin
    GYROID = 1
    DIAMOND = 2
    SCHWARZ_P = 3
    IWP = 4
    NEOVIUS = 5
    LIDINOID = 6
    NONE = 7
end

"""
    GridCell

A single cell in the scaffold grid.
"""
struct GridCell
    tpms_type::TPMSType
    iso_value::Float64      # Iso-value (controls local porosity)
    scale::Float64          # Unit cell size
    rotation::NTuple{3,Float64}  # Rotation angles (α, β, γ)
end

function GridCell()
    GridCell(NONE, 0.0, 1.0, (0.0, 0.0, 0.0))
end

"""
    ScaffoldGrid

A 3D grid of scaffold cells for hierarchical generation.
"""
struct ScaffoldGrid
    cells::Array{GridCell,3}
    resolution::NTuple{3,Int}
    physical_size::NTuple{3,Float64}  # mm
end

function ScaffoldGrid(resolution::NTuple{3,Int}; physical_size=(10.0, 10.0, 10.0))
    cells = [GridCell() for _ in 1:resolution[1], _ in 1:resolution[2], _ in 1:resolution[3]]
    ScaffoldGrid(cells, resolution, physical_size)
end

"""
    ScaffoldState

State representation for scaffold generation.
"""
mutable struct ScaffoldState
    grid::ScaffoldGrid
    current_cell::NTuple{3,Int}  # Current cell being edited
    features::Vector{Float64}    # Feature vector for policy
    is_terminal::Bool
    depth::Int

    # Computed properties (cached)
    estimated_porosity::Float64
    estimated_pore_size::Float64
    estimated_modulus::Float64
end

function ScaffoldState(resolution::NTuple{3,Int}; feature_dim::Int=64)
    grid = ScaffoldGrid(resolution)
    ScaffoldState(
        grid,
        (1, 1, 1),
        zeros(feature_dim),
        false,
        0,
        0.0, 0.0, 0.0
    )
end

function initial_scaffold_state(config::ScaffoldGFNConfig)
    state = ScaffoldState(config.grid_resolution; feature_dim=config.state_dim)
    update_features!(state)
    return state
end

"""
    ScaffoldAction

Actions for scaffold generation.
"""
struct ScaffoldAction
    action_type::Symbol  # :set_tpms, :set_iso, :set_scale, :next_cell, :finish
    value::Float64       # Action parameter
end

# Encode actions as integers
function action_to_int(action::ScaffoldAction, config::ScaffoldGFNConfig)
    if action.action_type == :set_tpms
        return Int(action.value)  # 1-7 for TPMS types
    elseif action.action_type == :set_iso
        return 7 + Int(floor(action.value * 5)) + 1  # 8-12
    elseif action.action_type == :set_scale
        return 13 + Int(floor(action.value * 3)) + 1  # 13-15
    elseif action.action_type == :next_cell
        return 16
    elseif action.action_type == :finish
        return 17
    end
    return 1
end

function int_to_action(action_int::Int)
    if action_int <= 7
        return ScaffoldAction(:set_tpms, Float64(action_int))
    elseif action_int <= 12
        iso_value = (action_int - 8) / 5.0
        return ScaffoldAction(:set_iso, iso_value)
    elseif action_int <= 15
        scale = 0.5 + (action_int - 13) * 0.5
        return ScaffoldAction(:set_scale, scale)
    elseif action_int == 16
        return ScaffoldAction(:next_cell, 0.0)
    else
        return ScaffoldAction(:finish, 0.0)
    end
end

# ============================================================================
# STATE FEATURES AND ENVIRONMENT
# ============================================================================

"""
    update_features!(state)

Update the feature vector from current grid state.
"""
function update_features!(state::ScaffoldState)
    grid = state.grid
    n_cells = prod(grid.resolution)

    # Encode current grid state
    features = zeros(length(state.features))

    # Cell type histogram
    for cell in grid.cells
        if cell.tpms_type != NONE
            idx = Int(cell.tpms_type)
            features[idx] += 1.0 / n_cells
        end
    end

    # Average iso-value and scale
    iso_sum = 0.0
    scale_sum = 0.0
    count = 0
    for cell in grid.cells
        if cell.tpms_type != NONE
            iso_sum += cell.iso_value
            scale_sum += cell.scale
            count += 1
        end
    end

    if count > 0
        features[8] = iso_sum / count
        features[9] = scale_sum / count
    end

    # Current position encoding
    cx, cy, cz = state.current_cell
    rx, ry, rz = grid.resolution
    features[10] = cx / rx
    features[11] = cy / ry
    features[12] = cz / rz

    # Progress
    features[13] = state.depth / n_cells

    # Estimated properties
    estimate_properties!(state)
    features[14] = state.estimated_porosity
    features[15] = state.estimated_pore_size / 500.0  # Normalize
    features[16] = state.estimated_modulus / 20.0

    state.features = features
    return features
end

"""
    estimate_properties!(state)

Estimate scaffold properties from current grid state.
"""
function estimate_properties!(state::ScaffoldState)
    grid = state.grid

    # Count active cells
    active_cells = 0
    iso_sum = 0.0
    scale_sum = 0.0

    for cell in grid.cells
        if cell.tpms_type != NONE
            active_cells += 1
            iso_sum += cell.iso_value
            scale_sum += cell.scale
        end
    end

    if active_cells == 0
        state.estimated_porosity = 1.0  # Empty = 100% porous
        state.estimated_pore_size = 0.0
        state.estimated_modulus = 0.0
        return
    end

    avg_iso = iso_sum / active_cells
    avg_scale = scale_sum / active_cells

    # Porosity estimate (simplified model)
    # Higher iso-value → more material → lower porosity
    state.estimated_porosity = 0.95 - avg_iso * 0.5

    # Pore size estimate
    # Larger scale → larger pores
    state.estimated_pore_size = 100.0 + avg_scale * 200.0

    # Modulus estimate (Gibson-Ashby)
    solid_fraction = 1 - state.estimated_porosity
    state.estimated_modulus = 100.0 * solid_fraction^2  # Simplified
end

"""
    scaffold_env_step(state, action_int, config) -> new_state

Apply action to scaffold state.
"""
function scaffold_env_step(state::ScaffoldState, action_int::Int,
                           config::ScaffoldGFNConfig)
    new_state = deepcopy(state)
    action = int_to_action(action_int)

    cx, cy, cz = new_state.current_cell
    rx, ry, rz = config.grid_resolution

    if action.action_type == :set_tpms
        tpms_type = TPMSType(Int(action.value))
        current_cell = new_state.grid.cells[cx, cy, cz]
        new_state.grid.cells[cx, cy, cz] = GridCell(
            tpms_type,
            current_cell.iso_value,
            current_cell.scale,
            current_cell.rotation
        )

    elseif action.action_type == :set_iso
        current_cell = new_state.grid.cells[cx, cy, cz]
        new_state.grid.cells[cx, cy, cz] = GridCell(
            current_cell.tpms_type,
            action.value,
            current_cell.scale,
            current_cell.rotation
        )

    elseif action.action_type == :set_scale
        current_cell = new_state.grid.cells[cx, cy, cz]
        new_state.grid.cells[cx, cy, cz] = GridCell(
            current_cell.tpms_type,
            current_cell.iso_value,
            action.value,
            current_cell.rotation
        )

    elseif action.action_type == :next_cell
        # Move to next cell in grid
        if cz < rz
            new_state.current_cell = (cx, cy, cz + 1)
        elseif cy < ry
            new_state.current_cell = (cx, cy + 1, 1)
        elseif cx < rx
            new_state.current_cell = (cx + 1, 1, 1)
        else
            # Reached end of grid
            new_state.is_terminal = true
        end

    elseif action.action_type == :finish
        new_state.is_terminal = true
    end

    new_state.depth += 1
    update_features!(new_state)

    return new_state
end

# ============================================================================
# REWARD FUNCTION
# ============================================================================

"""
    reward_scaffold(state, config) -> Float64

Compute reward for a scaffold state.

Reward is based on:
1. Distance from target properties
2. Constraint satisfaction
3. Structural validity

Returns reward in [0, 1] range.
"""
function reward_scaffold(state::ScaffoldState, config::ScaffoldGFNConfig)
    if !state.is_terminal
        return 0.0  # Only terminal states get reward
    end

    # Property matching rewards
    porosity_error = abs(state.estimated_porosity - config.target_porosity)
    porosity_reward = exp(-porosity_error^2 / (2 * config.porosity_tolerance^2))

    pore_size_error = abs(state.estimated_pore_size - config.target_pore_size)
    pore_size_reward = exp(-pore_size_error^2 / (2 * config.pore_size_tolerance^2))

    modulus_error = abs(state.estimated_modulus - config.target_modulus)
    modulus_reward = exp(-modulus_error^2 / (2 * config.modulus_tolerance^2))

    # Structural validity
    valid_cells = count(c -> c.tpms_type != NONE, state.grid.cells)
    total_cells = prod(config.grid_resolution)
    coverage = valid_cells / total_cells

    # Connectivity bonus (simplified)
    connectivity_reward = coverage > 0.5 ? 1.0 : coverage * 2

    # Combined reward
    property_reward = (porosity_reward + pore_size_reward + modulus_reward) / 3
    structural_reward = coverage * connectivity_reward

    # Weighted combination
    total_reward = 0.7 * property_reward + 0.3 * structural_reward

    # Apply sharpness (higher = more peaked around optima)
    total_reward = total_reward^config.reward_sharpness

    return max(total_reward, 1e-8)  # Minimum reward
end

# ============================================================================
# SCAFFOLD GFN TRAINER
# ============================================================================

"""
    ScaffoldGFNTrainer

Complete scaffold-specific GFlowNet trainer.
"""
struct ScaffoldGFNTrainer
    config::ScaffoldGFNConfig
    gfn_trainer::GFlowNetTrainer
end

function ScaffoldGFNTrainer(config::ScaffoldGFNConfig)
    gfn_config = GFlowNetConfig(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        learning_rate=config.learning_rate
    )

    gfn_trainer = GFlowNetTrainer(gfn_config)

    ScaffoldGFNTrainer(config, gfn_trainer)
end

"""
    create_scaffold_env(config)

Create environment step function for scaffold generation.
"""
function create_scaffold_env(config::ScaffoldGFNConfig)
    function env_step!(state::State, action::Int)
        # Convert to scaffold state
        scaffold_state = ScaffoldState(config.grid_resolution;
                                       feature_dim=config.state_dim)
        scaffold_state.features = state.features

        # Apply action
        new_scaffold_state = scaffold_env_step(scaffold_state, action, config)

        # Convert back to generic state
        new_state = State(config.state_dim)
        new_state.features = new_scaffold_state.features
        new_state.is_terminal = new_scaffold_state.is_terminal
        new_state.depth = new_scaffold_state.depth

        return new_state
    end

    return env_step!
end

# ============================================================================
# TRAINING AND GENERATION
# ============================================================================

"""
    train_scaffold_gfn!(trainer; n_episodes=1000, verbose=true)

Train the scaffold GFlowNet.
"""
function train_scaffold_gfn!(trainer::ScaffoldGFNTrainer;
                             n_episodes::Int=1000, verbose::Bool=true)
    config = trainer.config
    env_step! = create_scaffold_env(config)

    losses = Float64[]

    for episode in 1:n_episodes
        # Sample trajectory
        trajectory = sample_trajectory(trainer.gfn_trainer, env_step!)

        # Compute reward for terminal state
        scaffold_state = ScaffoldState(config.grid_resolution;
                                       feature_dim=config.state_dim)
        scaffold_state.features = trajectory.states[end].features
        scaffold_state.is_terminal = true
        estimate_properties!(scaffold_state)

        reward = reward_scaffold(scaffold_state, config)

        # Create trajectory with reward
        traj_with_reward = Trajectory(
            trajectory.states,
            trajectory.actions,
            trajectory.log_forward_probs,
            trajectory.log_backward_probs,
            reward
        )

        # Train
        loss = train_step!(trainer.gfn_trainer, traj_with_reward)
        push!(losses, loss)

        if verbose && episode % 100 == 0
            recent_loss = mean(losses[max(1, end-99):end])
            @info "Episode $episode: Loss = $(round(recent_loss, digits=4))"
        end
    end

    return losses
end

"""
    generate_diverse_scaffolds(trainer; n_samples=100, min_reward=0.5)

Generate diverse scaffolds using trained GFlowNet.

Returns scaffolds proportional to their reward, ensuring diversity.
"""
function generate_diverse_scaffolds(trainer::ScaffoldGFNTrainer;
                                    n_samples::Int=100,
                                    min_reward::Float64=0.5)
    config = trainer.config
    scaffolds = Tuple{ScaffoldState, Float64}[]

    for _ in 1:n_samples
        # Sample from GFlowNet
        state = initial_scaffold_state(config)

        for _ in 1:100  # Max steps
            if state.is_terminal
                break
            end

            # Convert to generic state for policy
            generic_state = State(config.state_dim)
            generic_state.features = state.features
            generic_state.is_terminal = state.is_terminal

            # Get action probabilities
            log_probs, _ = forward_policy(trainer.gfn_trainer, generic_state)
            action = sample_action(log_probs)

            # Apply action
            state = scaffold_env_step(state, action, config)
        end

        # Compute reward
        state.is_terminal = true
        reward = reward_scaffold(state, config)

        if reward >= min_reward
            push!(scaffolds, (state, reward))
        end
    end

    # Sort by reward (highest first)
    sort!(scaffolds, by=x -> -x[2])

    return scaffolds
end

"""
    optimize_scaffold_gfn(trainer; target_properties, n_iterations=500)

Generate scaffolds optimized for specific target properties.
"""
function optimize_scaffold_gfn(trainer::ScaffoldGFNTrainer;
                               target_porosity::Float64=0.85,
                               target_pore_size::Float64=200.0,
                               target_modulus::Float64=5.0,
                               n_iterations::Int=500)
    # Update config targets (would need mutable config in practice)
    # For now, use the trainer's config

    # Train for target
    train_scaffold_gfn!(trainer; n_episodes=n_iterations, verbose=false)

    # Generate samples
    scaffolds = generate_diverse_scaffolds(trainer; n_samples=50, min_reward=0.3)

    # Return best and diverse set
    if isempty(scaffolds)
        return nothing, []
    end

    best = scaffolds[1]

    # Get diverse subset (different TPMS types, property ranges)
    diverse = unique_scaffolds(scaffolds; max_return=10)

    return best, diverse
end

"""
    unique_scaffolds(scaffolds; max_return=10)

Select diverse scaffolds from a set.
"""
function unique_scaffolds(scaffolds::Vector{Tuple{ScaffoldState, Float64}};
                          max_return::Int=10)
    if length(scaffolds) <= max_return
        return scaffolds
    end

    selected = Tuple{ScaffoldState, Float64}[]
    push!(selected, scaffolds[1])  # Always include best

    for (state, reward) in scaffolds[2:end]
        if length(selected) >= max_return
            break
        end

        # Check diversity (simplified - compare properties)
        is_diverse = true
        for (sel_state, _) in selected
            porosity_diff = abs(state.estimated_porosity - sel_state.estimated_porosity)
            pore_diff = abs(state.estimated_pore_size - sel_state.estimated_pore_size)

            if porosity_diff < 0.05 && pore_diff < 30
                is_diverse = false
                break
            end
        end

        if is_diverse
            push!(selected, (state, reward))
        end
    end

    return selected
end

# ============================================================================
# UTILITIES
# ============================================================================

"""
    scaffold_to_volume(state, voxel_resolution)

Convert scaffold state to voxel volume.
"""
function scaffold_to_volume(state::ScaffoldState, voxel_resolution::Int=64)
    # Placeholder - would generate actual TPMS volume
    volume = zeros(Float32, voxel_resolution, voxel_resolution, voxel_resolution)

    grid = state.grid
    rx, ry, rz = grid.resolution
    cell_size_x = voxel_resolution ÷ rx
    cell_size_y = voxel_resolution ÷ ry
    cell_size_z = voxel_resolution ÷ rz

    for ix in 1:rx
        for iy in 1:ry
            for iz in 1:rz
                cell = grid.cells[ix, iy, iz]

                if cell.tpms_type == NONE
                    continue
                end

                # Fill cell region
                x_start = (ix - 1) * cell_size_x + 1
                y_start = (iy - 1) * cell_size_y + 1
                z_start = (iz - 1) * cell_size_z + 1

                for x in x_start:min(x_start + cell_size_x - 1, voxel_resolution)
                    for y in y_start:min(y_start + cell_size_y - 1, voxel_resolution)
                        for z in z_start:min(z_start + cell_size_z - 1, voxel_resolution)
                            # Simplified TPMS evaluation
                            volume[x, y, z] = evaluate_tpms_simple(
                                cell.tpms_type,
                                x / voxel_resolution,
                                y / voxel_resolution,
                                z / voxel_resolution,
                                cell.scale,
                                cell.iso_value
                            )
                        end
                    end
                end
            end
        end
    end

    return volume
end

"""
    evaluate_tpms_simple(tpms_type, x, y, z, scale, iso)

Simple TPMS evaluation for volume generation.
"""
function evaluate_tpms_simple(tpms_type::TPMSType, x::Float64, y::Float64, z::Float64,
                              scale::Float64, iso::Float64)
    xs = x * 2π / scale
    ys = y * 2π / scale
    zs = z * 2π / scale

    value = if tpms_type == GYROID
        sin(xs) * cos(ys) + sin(ys) * cos(zs) + sin(zs) * cos(xs)
    elseif tpms_type == DIAMOND
        sin(xs) * sin(ys) * sin(zs) +
        sin(xs) * cos(ys) * cos(zs) +
        cos(xs) * sin(ys) * cos(zs) +
        cos(xs) * cos(ys) * sin(zs)
    elseif tpms_type == SCHWARZ_P
        cos(xs) + cos(ys) + cos(zs)
    elseif tpms_type == IWP
        2 * (cos(xs) * cos(ys) + cos(ys) * cos(zs) + cos(zs) * cos(xs)) -
        (cos(2xs) + cos(2ys) + cos(2zs))
    else
        cos(xs) + cos(ys) + cos(zs)  # Default
    end

    return value > iso ? 1.0f0 : 0.0f0
end

"""
    summarize_scaffold(state)

Get summary statistics for a scaffold state.
"""
function summarize_scaffold(state::ScaffoldState)
    Dict(
        "porosity" => state.estimated_porosity,
        "pore_size_um" => state.estimated_pore_size,
        "modulus_MPa" => state.estimated_modulus,
        "grid_resolution" => state.grid.resolution,
        "active_cells" => count(c -> c.tpms_type != NONE, state.grid.cells),
        "total_cells" => prod(state.grid.resolution)
    )
end

end # module
