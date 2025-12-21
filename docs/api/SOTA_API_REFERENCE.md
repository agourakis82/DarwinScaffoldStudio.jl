# SOTA+++ API Reference

Complete API documentation for Darwin Scaffold Studio v3.4.0 SOTA+++ modules.

---

## Table of Contents

1. [Uncertainty Quantification](#uncertainty-quantification)
2. [Multi-Task Learning](#multi-task-learning)
3. [Scaffold Foundation Model](#scaffold-foundation-model)
4. [Geometric Laplace Neural Operators](#geometric-laplace-neural-operators)
5. [Active Learning](#active-learning)
6. [Explainable AI](#explainable-ai)

---

## Uncertainty Quantification

**Module**: `DarwinScaffoldStudio.UncertaintyQuantification`

### Types

#### `BayesianNN`
Bayesian Neural Network using variational inference.

**Fields**:
- `mean_net::Chain` - Mean network parameters
- `logvar_net::Chain` - Log-variance network parameters
- `prior_σ::Float32` - Prior standard deviation
- `n_samples::Int` - Number of MC samples for prediction

**Constructor**:
```julia
BayesianNN(input_dim, hidden_dims, output_dim; prior_σ=1.0f0, n_samples=100)
```

**Example**:
```julia
bnn = BayesianNN(10, [64, 32], 1)
```

---

#### `ConformalPredictor`
Conformal prediction for distribution-free uncertainty quantification.

**Fields**:
- `model::Any` - Base prediction model
- `calibration_scores::Vector{Float64}` - Nonconformity scores
- `α::Float64` - Miscoverage level (1-α is coverage probability)

**Constructor**:
```julia
ConformalPredictor(model; α=0.1)
```

**Example**:
```julia
cp = ConformalPredictor(my_model, α=0.1)  # 90% coverage
```

---

#### `UncertaintyDecomposition`
Decompose total uncertainty into aleatoric and epistemic components.

**Fields**:
- `total::Float64` - Total uncertainty (predictive variance)
- `aleatoric::Float64` - Aleatoric uncertainty (data noise)
- `epistemic::Float64` - Epistemic uncertainty (model uncertainty)

---

### Functions

#### `train_bayesian!`
Train Bayesian Neural Network using variational inference.

**Signature**:
```julia
train_bayesian!(bnn, X_train, y_train; epochs=100, lr=0.001)
```

**Arguments**:
- `bnn::BayesianNN` - Bayesian neural network
- `X_train::Matrix` - Training inputs (features × samples)
- `y_train::Matrix` - Training targets (outputs × samples)
- `epochs::Int` - Number of training epochs (default: 100)
- `lr::Float64` - Learning rate (default: 0.001)

**Returns**:
- `losses::Vector{Float64}` - Training losses per epoch

**Example**:
```julia
X_train = randn(Float32, 10, 100)
y_train = randn(Float32, 1, 100)
losses = train_bayesian!(bnn, X_train, y_train, epochs=50)
```

---

#### `predict_with_uncertainty`
Predict with uncertainty using Monte Carlo sampling.

**Signature**:
```julia
predict_with_uncertainty(bnn, X_test)
```

**Arguments**:
- `bnn::BayesianNN` - Trained Bayesian neural network
- `X_test::Matrix` - Test inputs (features × samples)

**Returns**:
- `mean::Vector` - Predictive mean
- `std::Vector` - Predictive standard deviation (total uncertainty)
- `samples::Matrix` - MC samples (for further analysis)

**Example**:
```julia
y_pred, y_std, samples = predict_with_uncertainty(bnn, X_test)
println("Prediction: $(y_pred[1]) ± $(y_std[1])")
```

---

#### `decompose_uncertainty`
Decompose uncertainty into aleatoric and epistemic components.

**Signature**:
```julia
decompose_uncertainty(bnn, X_test)
```

**Arguments**:
- `bnn::BayesianNN` - Bayesian neural network
- `X_test::Matrix` - Test inputs

**Returns**:
- `decompositions::Vector{UncertaintyDecomposition}` - Per-sample decomposition

**Example**:
```julia
decomps = decompose_uncertainty(bnn, X_test)
println("Total: $(decomps[1].total)")
println("Aleatoric: $(decomps[1].aleatoric)")
println("Epistemic: $(decomps[1].epistemic)")
```

---

#### `calibrate_conformal!`
Calibrate conformal predictor on calibration set.

**Signature**:
```julia
calibrate_conformal!(cp, X_cal, y_cal)
```

**Arguments**:
- `cp::ConformalPredictor` - Conformal predictor
- `X_cal::Matrix` - Calibration inputs
- `y_cal::Vector` - Calibration targets

**Example**:
```julia
calibrate_conformal!(cp, X_cal, y_cal)
```

---

#### `predict_conformal`
Predict with conformal prediction intervals.

**Signature**:
```julia
predict_conformal(cp, X_test)
```

**Arguments**:
- `cp::ConformalPredictor` - Calibrated conformal predictor
- `X_test::Matrix` - Test inputs

**Returns**:
- `y_pred::Vector` - Point predictions
- `lower::Vector` - Lower bounds of prediction intervals
- `upper::Vector` - Upper bounds of prediction intervals

**Example**:
```julia
y_pred, lower, upper = predict_conformal(cp, X_test)
println("Prediction: $(y_pred[1]) ∈ [$(lower[1]), $(upper[1])]")
```

---

## Multi-Task Learning

**Module**: `DarwinScaffoldStudio.MultiTaskLearning`

### Types

#### `MultiTaskModel`
Multi-task learning model with shared encoder and task-specific heads.

**Fields**:
- `encoder::SharedEncoder` - Shared feature encoder
- `task_heads::Dict{String, TaskHead}` - Task-specific prediction heads
- `task_names::Vector{String}` - List of task names

---

### Functions

#### `create_scaffold_mtl_model`
Create a multi-task model for scaffold property prediction.

**Signature**:
```julia
create_scaffold_mtl_model(input_dim; encoder_dims=[128, 64], head_dim=32)
```

**Arguments**:
- `input_dim::Int` - Input dimension (scaffold features)
- `encoder_dims::Vector{Int}` - Shared encoder dimensions (default: [128, 64])
- `head_dim::Int` - Task head hidden dimension (default: 32)

**Returns**:
- `model::MultiTaskModel` - Multi-task model

**Example**:
```julia
model = create_scaffold_mtl_model(100)
println("Tasks: $(model.task_names)")
```

---

#### `train_multitask!`
Train multi-task model.

**Signature**:
```julia
train_multitask!(model, X_train, y_train_dict; epochs=100, lr=0.001, batch_size=32)
```

**Arguments**:
- `model::MultiTaskModel` - Multi-task model
- `X_train::Matrix` - Training inputs (features × samples)
- `y_train_dict::Dict{String, Vector}` - Training targets for each task
- `epochs::Int` - Number of training epochs (default: 100)
- `lr::Float64` - Learning rate (default: 0.001)
- `batch_size::Int` - Batch size (default: 32)

**Returns**:
- `history::Dict` - Training history (losses per epoch)

**Example**:
```julia
y_train_dict = Dict(
    "porosity" => y_porosity,
    "pore_size" => y_pore_size,
    "interconnectivity" => y_interconnectivity
)
history = train_multitask!(model, X_train, y_train_dict, epochs=50)
```

---

#### `predict_multitask`
Predict all tasks simultaneously.

**Signature**:
```julia
predict_multitask(model, X_test)
```

**Arguments**:
- `model::MultiTaskModel` - Trained multi-task model
- `X_test::Matrix` - Test inputs

**Returns**:
- `predictions::Dict{String, Vector}` - Predictions for each task

**Example**:
```julia
predictions = predict_multitask(model, X_test)
println("Porosity: $(predictions["porosity"][1])")
println("Pore size: $(predictions["pore_size"][1])")
```

---

#### `evaluate_multitask`
Evaluate multi-task model on test set.

**Signature**:
```julia
evaluate_multitask(model, X_test, y_test_dict)
```

**Arguments**:
- `model::MultiTaskModel` - Multi-task model
- `X_test::Matrix` - Test inputs
- `y_test_dict::Dict{String, Vector}` - Test targets for each task

**Returns**:
- `metrics::Dict{String, Dict}` - Metrics for each task (MSE, MAE, R²)

**Example**:
```julia
metrics = evaluate_multitask(model, X_test, y_test_dict)
println("Porosity R²: $(metrics["porosity"]["R²"])")
```

---

## Scaffold Foundation Model

**Module**: `DarwinScaffoldStudio.ScaffoldFoundationModel`

### Types

#### `ScaffoldFM`
Scaffold Foundation Model - Multi-modal transformer for scaffold analysis.

**Fields**:
- `patch_embed::PatchEmbedding3D` - 3D patch embedding
- `pos_embed::Array{Float32, 3}` - Learnable positional embeddings
- `cls_token::Array{Float32, 3}` - Classification token
- `transformer_blocks::Vector{TransformerBlock}` - Transformer encoder
- `material_encoder::Chain` - Material property encoder
- `fusion_layer::Dense` - Multi-modal fusion
- `decoder_head::Chain` - Reconstruction decoder
- `property_head::Chain` - Property prediction head

---

### Functions

#### `create_scaffold_fm`
Create Scaffold Foundation Model.

**Signature**:
```julia
create_scaffold_fm(;
    scaffold_size=(64,64,64),
    patch_size=(8,8,8),
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    material_dim=50
)
```

**Arguments**:
- `scaffold_size::Tuple{Int,Int,Int}` - Input scaffold dimensions (default: (64,64,64))
- `patch_size::Tuple{Int,Int,Int}` - Patch size for 3D ViT (default: (8,8,8))
- `embed_dim::Int` - Embedding dimension (default: 256)
- `num_heads::Int` - Number of attention heads (default: 8)
- `num_layers::Int` - Number of transformer layers (default: 6)
- `material_dim::Int` - Material property dimension (default: 50)

**Returns**:
- `model::ScaffoldFM` - Scaffold foundation model

**Example**:
```julia
scaffold_fm = create_scaffold_fm(
    scaffold_size=(64, 64, 64),
    embed_dim=256,
    num_heads=8
)
```

---

#### `encode_scaffold`
Encode scaffold into latent representation.

**Signature**:
```julia
encode_scaffold(model, scaffold_voxels, material_props)
```

**Arguments**:
- `model::ScaffoldFM` - Scaffold foundation model
- `scaffold_voxels::Array{Float32, 5}` - Voxel grid (W×H×D×1×B)
- `material_props::Matrix{Float32}` - Material properties (material_dim × B)

**Returns**:
- `latent::Matrix{Float32}` - Latent representation (embed_dim × B)

**Example**:
```julia
voxels = rand(Float32, 64, 64, 64, 1, 10)
materials = randn(Float32, 50, 10)
latent = encode_scaffold(scaffold_fm, voxels, materials)
```

---

#### `predict_properties`
Predict scaffold properties from geometry and material.

**Signature**:
```julia
predict_properties(model, scaffold_voxels, material_props)
```

**Arguments**:
- `model::ScaffoldFM` - Scaffold foundation model
- `scaffold_voxels::Array` - Voxel grids
- `material_props::Matrix` - Material properties

**Returns**:
- `properties::Matrix{Float32}` - Predicted properties (7 × B)
  [porosity, pore_size, interconnectivity, tortuosity, surface_area, permeability, modulus]

**Example**:
```julia
properties = predict_properties(scaffold_fm, voxels, materials)
println("Porosity: $(properties[1, 1])")
println("Pore size: $(properties[2, 1])")
```

---

#### `pretrain_scaffoldfm!`
Pre-train Scaffold Foundation Model on large unlabeled dataset.

**Signature**:
```julia
pretrain_scaffoldfm!(model, scaffold_dataset; epochs=100, lr=0.0001)
```

**Arguments**:
- `model::ScaffoldFM` - Scaffold foundation model
- `scaffold_dataset::Vector` - List of (voxels, materials) tuples
- `epochs::Int` - Number of pre-training epochs (default: 100)
- `lr::Float64` - Learning rate (default: 0.0001)

**Returns**:
- `losses::Vector{Float64}` - Pre-training losses

**Example**:
```julia
dataset = [(voxels_i, materials_i) for i in 1:10000]
losses = pretrain_scaffoldfm!(scaffold_fm, dataset, epochs=100)
```

---

#### `finetune_scaffoldfm!`
Fine-tune pre-trained model on downstream task.

**Signature**:
```julia
finetune_scaffoldfm!(model, X_train, y_train, material_props; epochs=50, lr=0.0001)
```

**Arguments**:
- `model::ScaffoldFM` - Pre-trained scaffold foundation model
- `X_train::Array` - Training scaffold voxels
- `y_train::Matrix` - Training property labels (7 × N)
- `material_props::Matrix` - Material properties
- `epochs::Int` - Fine-tuning epochs (default: 50)
- `lr::Float64` - Learning rate (default: 0.0001)

**Returns**:
- `losses::Vector{Float64}` - Fine-tuning losses

**Example**:
```julia
losses = finetune_scaffoldfm!(scaffold_fm, X_train, y_train, materials, epochs=50)
```

---

## Geometric Laplace Neural Operators

**Module**: `DarwinScaffoldStudio.GeometricLaplaceOperator`

### Types

#### `GeometricLaplaceNO`
Geometric Laplace Neural Operator for learning PDE solutions on scaffolds.

**Fields**:
- `spectral_encoder::Chain` - Encodes spectral features
- `kernel_network::Chain` - Learns integral kernel in spectral space
- `decoder::Chain` - Decodes to physical space
- `k_modes::Int` - Number of spectral modes

---

### Functions

#### `build_laplacian_matrix`
Build discrete Laplacian matrix for scaffold geometry.

**Signature**:
```julia
build_laplacian_matrix(scaffold_voxels, voxel_size)
```

**Arguments**:
- `scaffold_voxels::Array{Bool, 3}` - Binary scaffold (true = solid)
- `voxel_size::Float64` - Physical voxel size (μm)

**Returns**:
- `L::SparseMatrixCSC` - Laplacian matrix (N × N)
- `node_coords::Matrix{Float64}` - Node coordinates (3 × N)
- `node_map::Dict` - Mapping from 3D indices to node IDs

**Example**:
```julia
scaffold = rand(Bool, 32, 32, 32)
L, coords, node_map = build_laplacian_matrix(scaffold, 10.0)
println("Nodes: $(size(L, 1))")
```

---

#### `spectral_embedding`
Compute spectral embedding using Laplacian eigenvectors.

**Signature**:
```julia
spectral_embedding(L, k)
```

**Arguments**:
- `L::SparseMatrixCSC` - Laplacian matrix
- `k::Int` - Number of eigenvectors to use

**Returns**:
- `embedding::Matrix{Float64}` - Spectral embedding (k × N)

**Example**:
```julia
spectral_basis = spectral_embedding(L, 32)
```

---

#### `GeometricLaplaceNO` (Constructor)
Construct Geometric Laplace Neural Operator.

**Signature**:
```julia
GeometricLaplaceNO(input_dim, hidden_dim, output_dim, k_modes)
```

**Arguments**:
- `input_dim::Int` - Input feature dimension
- `hidden_dim::Int` - Hidden layer dimension
- `output_dim::Int` - Output dimension
- `k_modes::Int` - Number of spectral modes

**Example**:
```julia
glno = GeometricLaplaceNO(1, 128, 1, 32)
```

---

#### `train_glno!`
Train Geometric Laplace Neural Operator.

**Signature**:
```julia
train_glno!(glno, training_data, L, spectral_basis; epochs=100, lr=0.001)
```

**Arguments**:
- `glno::GeometricLaplaceNO` - Neural operator
- `training_data::Vector` - List of (u0, u_target) pairs
- `L::SparseMatrixCSC` - Laplacian matrix
- `spectral_basis::Matrix` - Spectral embedding
- `epochs::Int` - Training epochs (default: 100)
- `lr::Float64` - Learning rate (default: 0.001)

**Returns**:
- `losses::Vector{Float64}` - Training losses

**Example**:
```julia
training_data = [(u0_i, u_target_i) for i in 1:100]
losses = train_glno!(glno, training_data, L, spectral_basis, epochs=100)
```

---

#### `solve_pde_on_scaffold`
Solve PDE on scaffold geometry using trained neural operator.

**Signature**:
```julia
solve_pde_on_scaffold(glno, scaffold_voxels, u0, voxel_size)
```

**Arguments**:
- `glno::GeometricLaplaceNO` - Trained neural operator
- `scaffold_voxels::Array{Bool, 3}` - Scaffold geometry
- `u0::Vector` - Initial conditions (per node)
- `voxel_size::Float64` - Voxel size (μm)

**Returns**:
- `u_solution::Vector` - Solution field (per node)
- `node_coords::Matrix` - Node coordinates

**Example**:
```julia
u0 = ones(Float64, n_nodes)
u_solution, coords = solve_pde_on_scaffold(glno, scaffold, u0, 10.0)
```

---

## Active Learning

**Module**: `DarwinScaffoldStudio.ActiveLearning`

### Types

#### `ActiveLearner`
Active learning framework for scaffold optimization.

**Fields**:
- `model::Any` - Surrogate model
- `acquisition::AcquisitionFunction` - Acquisition function
- `X_observed::Matrix` - Observed inputs
- `y_observed::Vector` - Observed outputs
- `f_best::Float64` - Best observed value

---

#### Acquisition Functions

- `ExpectedImprovement(ξ=0.01)` - Expected Improvement
- `UpperConfidenceBound(β=2.0)` - Upper Confidence Bound
- `ProbabilityOfImprovement(ξ=0.01)` - Probability of Improvement
- `ThompsonSampling()` - Thompson Sampling

---

### Functions

#### `ActiveLearner` (Constructor)
Create active learner.

**Signature**:
```julia
ActiveLearner(model, acquisition)
```

**Arguments**:
- `model::Any` - Surrogate model
- `acquisition::AcquisitionFunction` - Acquisition function

**Example**:
```julia
learner = ActiveLearner(my_model, ExpectedImprovement())
```

---

#### `update_model!`
Update active learner with new observations.

**Signature**:
```julia
update_model!(learner, X_new, y_new)
```

**Arguments**:
- `learner::ActiveLearner` - Active learner
- `X_new::Matrix` - New input observations
- `y_new::Vector` - New output observations

**Example**:
```julia
update_model!(learner, X_new, y_new)
```

---

#### `select_next_experiments`
Select next experiments using acquisition function.

**Signature**:
```julia
select_next_experiments(learner, X_candidates; n_select=1)
```

**Arguments**:
- `learner::ActiveLearner` - Active learner
- `X_candidates::Matrix` - Candidate experiments (features × N)
- `n_select::Int` - Number of experiments to select (default: 1)

**Returns**:
- `selected_indices::Vector{Int}` - Indices of selected experiments
- `acquisition_values::Vector{Float64}` - Acquisition values

**Example**:
```julia
selected, acq_vals = select_next_experiments(learner, X_candidates, n_select=5)
```

---

#### `batch_selection`
Select batch of experiments for parallel execution.

**Signature**:
```julia
batch_selection(learner, X_candidates, batch_size; method=:greedy)
```

**Arguments**:
- `learner::ActiveLearner` - Active learner
- `X_candidates::Matrix` - Candidate experiments
- `batch_size::Int` - Number of experiments in batch
- `method::Symbol` - Selection method (:greedy, :diverse, :thompson)

**Returns**:
- `batch_indices::Vector{Int}` - Selected batch indices

**Example**:
```julia
batch = batch_selection(learner, X_candidates, 10, method=:diverse)
```

---

## Explainable AI

**Module**: `DarwinScaffoldStudio.ExplainableAI`

### Functions

#### `compute_shap_values`
Compute SHAP values for a prediction using Kernel SHAP.

**Signature**:
```julia
compute_shap_values(model, x, X_background; n_samples=100)
```

**Arguments**:
- `model::Function` - Prediction model (x -> y)
- `x::Vector` - Instance to explain
- `X_background::Matrix` - Background dataset for baseline
- `n_samples::Int` - Number of samples for approximation (default: 100)

**Returns**:
- `shap_values::Vector{Float64}` - SHAP value for each feature
- `base_value::Float64` - Baseline prediction

**Example**:
```julia
shap_vals, base = compute_shap_values(model, x, X_background, n_samples=100)
```

---

#### `explain_prediction`
Generate human-readable explanation of prediction.

**Signature**:
```julia
explain_prediction(model, x, X_background, feature_names)
```

**Arguments**:
- `model::Function` - Prediction model
- `x::Vector` - Instance to explain
- `X_background::Matrix` - Background dataset
- `feature_names::Vector{String}` - Names of features

**Returns**:
- `explanation::Dict` - Explanation with SHAP values and interpretation

**Example**:
```julia
explanation = explain_prediction(model, x, X_bg, ["porosity", "pore_size", ...])
```

---

#### `feature_importance`
Compute feature importance using permutation importance.

**Signature**:
```julia
feature_importance(model, X_test, y_test; n_repeats=10)
```

**Arguments**:
- `model::Function` - Prediction model
- `X_test::Matrix` - Test inputs
- `y_test::Vector` - Test targets
- `n_repeats::Int` - Number of permutation repeats (default: 10)

**Returns**:
- `importances::Vector{Float64}` - Importance score for each feature
- `importances_std::Vector{Float64}` - Standard deviation of importance

**Example**:
```julia
importances, std = feature_importance(model, X_test, y_test, n_repeats=10)
```

---

#### `visualize_attention`
Visualize attention weights from transformer model.

**Signature**:
```julia
visualize_attention(attention_weights, patch_indices)
```

**Arguments**:
- `attention_weights::Matrix` - Attention weights (num_patches × num_patches)
- `patch_indices::Vector{Tuple}` - 3D indices of patches

**Returns**:
- `attention_map::Dict` - Attention visualization data

**Example**:
```julia
attention_map = visualize_attention(attn_weights, patch_indices)
```

---

#### `counterfactual_explanation`
Generate counterfactual explanation.

**Signature**:
```julia
counterfactual_explanation(model, x, target_value, feature_names; 
                          max_changes=3, lr=0.1, max_iter=100)
```

**Arguments**:
- `model::Function` - Prediction model
- `x::Vector` - Original instance
- `target_value::Float64` - Desired prediction
- `feature_names::Vector{String}` - Feature names
- `max_changes::Int` - Maximum number of features to change (default: 3)
- `lr::Float64` - Learning rate (default: 0.1)
- `max_iter::Int` - Maximum iterations (default: 100)

**Returns**:
- `counterfactual::Vector{Float64}` - Modified instance
- `changes::Dict` - Description of changes

**Example**:
```julia
x_cf, changes = counterfactual_explanation(model, x, 0.9, feature_names, max_changes=3)
```

---

## Common Patterns

### Complete Workflow Example

```julia
using DarwinScaffoldStudio

# 1. Load data
X_train = randn(Float32, 50, 1000)
y_train = randn(Float32, 1, 1000)

# 2. Train with uncertainty quantification
bnn = UncertaintyQuantification.BayesianNN(50, [128, 64], 1)
losses = UncertaintyQuantification.train_bayesian!(bnn, X_train, y_train, epochs=100)

# 3. Predict with uncertainty
X_test = randn(Float32, 50, 100)
y_pred, y_std, samples = UncertaintyQuantification.predict_with_uncertainty(bnn, X_test)

# 4. Decompose uncertainty
decomps = UncertaintyQuantification.decompose_uncertainty(bnn, X_test)

# 5. Explain predictions
feature_names = ["feature_$i" for i in 1:50]
explanation = ExplainableAI.explain_prediction(
    x -> bnn.mean_net(x),
    X_test[:, 1],
    X_train,
    feature_names
)

# 6. Active learning for next experiments
learner = ActiveLearning.ActiveLearner(x -> bnn.mean_net(x), ActiveLearning.ExpectedImprovement())
ActiveLearning.update_model!(learner, X_train, vec(y_train))
X_candidates = randn(Float64, 50, 500)
selected, acq = ActiveLearning.select_next_experiments(learner, X_candidates, n_select=10)
```

---

## Performance Tips

1. **Batch Processing**: Use batch sizes of 32-128 for optimal GPU utilization
2. **MC Samples**: Use 100-1000 samples for uncertainty quantification
3. **Spectral Modes**: Use 16-64 modes for GLNO depending on geometry complexity
4. **Pre-training**: Pre-train ScaffoldFM on 10K+ scaffolds before fine-tuning
5. **Active Learning**: Start with 20-50 initial observations before active selection

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory during training
**Solution**: Reduce batch size or model size

**Issue**: Slow training
**Solution**: Use GPU acceleration (CUDA.jl) or reduce model complexity

**Issue**: Poor uncertainty calibration
**Solution**: Increase training epochs or use more calibration data

**Issue**: GLNO not converging
**Solution**: Increase spectral modes or use more training data

---

## Version History

- **v3.4.0** (2025-12-21): Initial SOTA+++ release
  - Uncertainty Quantification
  - Multi-Task Learning
  - Scaffold Foundation Model
  - Geometric Laplace Neural Operators
  - Active Learning
  - Explainable AI

---

**For more information, see**:
- [SOTA_PLUS_PLUS_PLUS.md](../../SOTA_PLUS_PLUS_PLUS.md) - Feature overview
- [examples/sota_plus_plus_plus_demo.jl](../../examples/sota_plus_plus_plus_demo.jl) - Complete examples
- [GitHub Issues](https://github.com/agourakis82/darwin-scaffold-studio/issues) - Support

---

**Darwin Scaffold Studio v3.4.0 - SOTA+++ API Reference**
