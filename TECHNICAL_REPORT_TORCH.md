# PyTranus PyTorch Implementation - Technical Report

## Overview

The PyTorch implementation (`LcalTorch`) provides an alternative to the NumPy/SciPy-based LCAL calibration. It reformulates the calibration as **end-to-end differentiable optimization**, leveraging PyTorch's automatic differentiation and modern optimizers for improved robustness and GPU acceleration.

## Key Differences from NumPy Implementation

| Aspect | NumPy/SciPy (`Lcal`) | PyTorch (`LcalTorch`) |
|--------|---------------------|----------------------|
| **Dependencies** | numpy, scipy | torch only (no scipy) |
| **Differentiation** | Analytical Jacobians (manual) | Automatic differentiation |
| **Optimizer** | Levenberg-Marquardt (`scipy.leastsq`) | L-BFGS (`torch.optim.LBFGS`) |
| **Linear solver** | `scipy.linalg.solve` | `torch.linalg.solve` |
| **Housing optimization** | Per-zone (parallel processes) | Joint optimization (all zones) |
| **GPU support** | No | Yes (`device='cuda'`) |
| **Jacobian code** | 265 lines in `_math.py` | 0 lines (autograd) |

> **Note**: The PyTorch implementation is completely scipy-free. All optimization uses `torch.optim.LBFGS` and linear algebra uses `torch.linalg`.

## Architecture

### Module Structure

The PyTorch implementation uses `nn.Module` classes for each computational component:

```
LCALModel (nn.Module)
├── DemandFunction          # a(U) = demin + (demax - demin) × exp(-δU)
├── SubstitutionProbability # S(U) via logit over choice set
├── LocationProbability     # Pr(φ) via logit over zones
├── TotalDemand            # D = exog + Σ(a × S × X)
├── ProductionModule       # X = D @ Pr
└── HousingSectorModel     # Combined model for housing sectors
```

Each module:
- Stores parameters as **registered buffers** (non-trainable)
- Implements a differentiable `forward()` method
- Supports batched operations for efficiency

### Tensor Layout

All tensors use consistent shapes:
- Sector-zone matrices: `(n_sectors, n_zones)`
- Demand/substitution: `(n_sectors, n_sectors, n_zones)` for `a[m,n,i]`, `S[m,n,i]`
- Location probabilities: `(n_sectors, n_zones, n_zones)` for `Pr[n,i,j]`
- Transport costs: `(n_sectors, n_zones, n_zones)` for `t[n,i,j]`

## Optimization Approach

### Phase 1: Housing Sectors

**Problem formulation:**
```
minimize   L(h) = Σᵢ Σₙ (Xₙᵢ(h) - Xₙᵢᵒᵇˢ)²
subject to h ∈ ℝⁿ_housing × n_zones
```

**NumPy approach:** Solve n_zones independent problems in parallel:
```python
for i in range(n_zones):  # Parallel via ProcessPoolExecutor
    h_i = leastsq(residual, h0_i, args=(i,))
```

**PyTorch approach:** Single joint optimization over all zones:
```python
h = torch.zeros(n_housing, n_zones, requires_grad=True)
optimizer = torch.optim.LBFGS([h], lr=1.0, line_search_fn="strong_wolfe")

def closure():
    optimizer.zero_grad()
    X_pred = model.housing_production(h)
    loss = F.mse_loss(X_pred, X_target)
    loss.backward()  # Automatic differentiation
    return loss

optimizer.step(closure)
```

**Advantages of joint optimization:**
1. **GPU efficiency**: Single kernel launch vs. n_zones sequential launches
2. **Better conditioning**: Optimizer sees full loss landscape
3. **Simpler code**: No multiprocessing overhead

### Phase 2: Transportable Sectors

**Problem formulation (per sector n):**
```
minimize   Lₙ(φₙ) = Σⱼ (Xₙⱼ(φₙ) - Xₙⱼᵒᵇˢ)²
where      φₙ = λₙ × (pₙ + hₙ)
```

**NumPy approach:** Levenberg-Marquardt with analytical Jacobian:
```python
result = leastsq(
    res_X_n,           # Residual function
    ph0,               # Initial guess
    Dfun=calc_DX_n,    # Analytical Jacobian (265 lines of code)
    ftol=1e-4
)
```

**PyTorch approach:** L-BFGS with automatic differentiation:
```python
phi = torch.zeros(n_zones, requires_grad=True)
optimizer = torch.optim.LBFGS([phi], lr=1.0, line_search_fn="strong_wolfe")

def closure():
    optimizer.zero_grad()
    X_pred = model.transportable_production(phi, n, D_n)
    loss = F.mse_loss(X_pred, X_target)
    loss.backward()  # Gradient computed automatically
    return loss

optimizer.step(closure)
```

### Phase 3: Price Recovery

Both implementations solve the same linear system:
```
(I - Δ) × p = Λ
```

**NumPy:** `scipy.linalg.solve(DELTA, LAMBDA)`
**PyTorch:** `torch.linalg.solve(DELTA, LAMBDA)`

## Automatic Differentiation Details

### Forward Pass

The production computation is fully differentiable:

```python
def transportable_production(phi_n, n, D_n):
    # U_ij = λ × φⱼ + t_ij
    U_nij = lamda[n] * phi_n.unsqueeze(0) + t_nij[n]

    # Pr_ij = softmax_j(log(A_j) - β × U_ij)
    log_weights = safe_log(A_ni[n]).unsqueeze(0) - beta[n] * U_nij
    Pr_n = F.softmax(log_weights, dim=-1)

    # X_j = Σᵢ Dᵢ × Pr_ij
    return torch.einsum("i,ij->j", D_n, Pr_n)
```

### Backward Pass (Automatic)

PyTorch computes gradients via reverse-mode autodiff:

```
∂L/∂φₖ = Σⱼ ∂L/∂Xⱼ × ∂Xⱼ/∂φₖ
```

The chain rule propagates through:
1. `einsum` (production aggregation)
2. `softmax` (location probabilities)
3. Linear operations (utility computation)

This eliminates the need for manual Jacobian derivation:

**Analytical Jacobian (NumPy):**
```python
# From _math.py - 265 lines of manual derivative code
def compute_DX_n(DX, n_sectors, n_zones, beta, lamda, D_n, Pr_n, U_n, logit):
    coef = lamda * beta
    weighted_Pr = Pr_n * D_n[:, np.newaxis]
    DX[:, :] = coef * (weighted_Pr.T @ Pr_n)
    diag_term = coef * np.sum(D_n[:, np.newaxis] * Pr_n * (1 - Pr_n), axis=0)
    np.fill_diagonal(DX, -diag_term)
    return DX
```

**Automatic Jacobian (PyTorch):**
```python
# No code needed - just call loss.backward()
```

## Numerical Considerations

### Softmax Stability

Location probabilities use log-space computation for numerical stability:

```python
# Instead of: Pr = A * exp(-β*U) / sum(A * exp(-β*U))
# We compute: Pr = softmax(log(A) - β*U)

log_weights = safe_log(A_ni[n]) - beta[n] * U_nij
Pr_n = F.softmax(log_weights, dim=-1)
```

This avoids overflow/underflow when `β*U` is large.

### Safe Logarithm

```python
def safe_log(x, eps=1e-10):
    return torch.log(x + eps)
```

Prevents `log(0)` for zero attractors.

### Double Precision

All computations use `torch.float64` to match NumPy precision:

```python
lcal = LcalTorch(config, dtype=torch.float64)
```

## L-BFGS Optimizer

### Why L-BFGS?

| Property | Levenberg-Marquardt | L-BFGS |
|----------|-------------------|--------|
| **Type** | Gauss-Newton (least squares) | Quasi-Newton (general) |
| **Hessian** | J^T J approximation | BFGS approximation |
| **Memory** | O(n²) for Jacobian | O(mn) for m history vectors |
| **Line search** | Trust region | Strong Wolfe conditions |
| **Convergence** | Quadratic near minimum | Superlinear |

L-BFGS is well-suited because:
1. **Memory efficient**: Doesn't store full Hessian
2. **Robust**: Strong Wolfe line search ensures descent
3. **Fast**: Superlinear convergence for smooth problems
4. **Compatible**: Works with PyTorch's autograd

### Configuration

```python
optimizer = torch.optim.LBFGS(
    [h],
    lr=1.0,                        # Step size (scaled by line search)
    max_iter=100,                  # Max iterations per step()
    tolerance_grad=1e-8,           # Gradient tolerance
    tolerance_change=1e-8,         # Parameter change tolerance
    line_search_fn="strong_wolfe"  # Robust line search
)
```

## GPU Acceleration

### When to Use GPU

GPU provides speedup for:
- Large number of zones (n_zones > 50)
- Large number of sectors (n_sectors > 20)
- Batch operations (joint housing optimization)

### Usage

```python
# CPU (default)
lcal = LcalTorch(config)

# GPU (CUDA)
lcal = LcalTorch(config, device='cuda')

# Apple Silicon (MPS)
lcal = LcalTorch(config, device='mps')
```

### Memory Layout

All tensors are allocated on the same device:
```python
self.register_buffer("t_nij", to_tensor(params.t_nij))  # On device
self.register_buffer("A_ni", A_ni)                       # On device
```

## API Comparison

### NumPy Implementation

```python
from pytranus import Lcal, TranusConfig

config = TranusConfig(bin_path, work_dir, project_id, scenario_id)
lcal = Lcal(config, normalize=True)

# Initial guess required
ph0 = np.random.randn(n_sectors, n_zones) * 0.1
p, h, conv, lamda_opt = lcal.compute_shadow_prices(ph0)
```

### PyTorch Implementation

```python
from pytranus import LcalTorch, TranusConfig

config = TranusConfig(bin_path, work_dir, project_id, scenario_id)
lcal = LcalTorch(config, normalize=True, device='cuda')

# Initial guess optional (defaults to zeros)
p, h, conv = lcal.compute_shadow_prices()

# Get goodness-of-fit statistics
stats = lcal.goodness_of_fit()
print(stats['housing']['r_squared'])
```

### Convenience Function

```python
from pytranus import calibrate

p, h, stats = calibrate(config, device='cuda', verbose=True)
```

## Validation

The PyTorch implementation is validated against the NumPy implementation:

| Test | Tolerance | Status |
|------|-----------|--------|
| Jacobian consistency | 1e-10 | ✓ |
| Demand functions | 1e-6 | ✓ |
| Location probabilities | 1e-6 | ✓ |
| Housing production | 1e-6 | ✓ |
| Housing optimization fit | <1% | ✓ |
| Full calibration error | <10% diff | ✓ |

Run tests with:
```bash
pytest tests/test_torch.py -v
```

## Future Extensions

The PyTorch architecture enables:

1. **Joint sector optimization**: Optimize all sectors simultaneously instead of sequentially
2. **Learnable parameters**: Make β, λ, σ trainable via gradient descent
3. **Neural network integration**: Replace logit models with neural networks
4. **Batched scenarios**: Process multiple scenarios in parallel
5. **Second-order methods**: Use exact Hessian via `torch.autograd.functional.hessian`

## File Structure

```
pytranus/
├── torch_utils.py    # Tensor utilities (to_tensor, softmax_logit, etc.)
├── modules.py        # nn.Module implementations
│   ├── DemandFunction
│   ├── SubstitutionProbability
│   ├── LocationProbability
│   ├── ProductionModule
│   ├── TotalDemand
│   ├── HousingSectorModel
│   └── LCALModel
└── lcal_torch.py     # Main LcalTorch class
    ├── _optimize_housing_zone()
    ├── _optimize_housing_all_lbfgs()
    ├── _optimize_sector_lbfgs()
    ├── calc_sp_housing()
    ├── calc_ph()
    ├── calc_p_linear()
    └── compute_shadow_prices()
```

## References

1. Liu, D. C., & Nocedal, J. (1989). "On the limited memory BFGS method for large scale optimization." *Mathematical Programming*, 45(1-3), 503-528.

2. Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*.

3. Nocedal, J., & Wright, S. (2006). *Numerical Optimization*. Springer. (Chapter 7: Large-Scale Unconstrained Optimization)
