# PyTranus LCAL Algorithm - Technical Report

## Overview

The Land Use Calibration (LCAL) module implements a discrete choice model for urban land use allocation. Unlike the original Tranus implementation, PyTranus reformulates calibration as an **optimization problem**, finding shadow prices that minimize the difference between observed and predicted production levels.

## Mathematical Foundation

### The Tranus Land Use Model

Tranus models urban systems using an input-output economic framework combined with discrete choice theory. The economy is divided into:

- **Sectors** (n): Economic activities (employment, housing, retail, etc.)
- **Zones** (i, j): Geographic areas

The model computes:
- **Production** X_n,i: Output of sector n in zone i
- **Prices** p_n,i: Market prices
- **Shadow prices** h_n,i: Calibration adjustments to match observed data

### Core Equations

#### 1. Demand Functions

The demand from sector m for sector n follows an elastic demand function:

```
a^{mn}_i = a^{mn}_min + (a^{mn}_max - a^{mn}_min) × exp(-δ^{mn} × U^n_i)
```

Where:
- `a^{mn}_min`, `a^{mn}_max`: Minimum/maximum demand coefficients
- `δ^{mn}`: Demand elasticity parameter
- `U^n_i = p^n_i + h^n_i`: Total utility (price + shadow price)

#### 2. Substitution Probabilities

When sector m can substitute between alternative sectors, the substitution probability follows a logit model:

```
S^{mn}_i = (W^n_i × exp(-σ^m × Ũ^{mn}_i)) / Σ_k (W^k_i × exp(-σ^m × Ũ^{mk}_i))
```

Where:
- `σ^m`: Substitution dispersion parameter
- `Ũ^{mn}_i = ω^{mn} × a^{mn}_i × U^n_i`: Weighted utility
- `W^n_i`: Attractor for sector n in zone i
- The sum is over all sectors k in the substitution choice set K^m

#### 3. Location Choice (Logit Model)

For transportable sectors, the probability that demand from zone i is satisfied in zone j:

```
Pr^n_{ij} = (A^n_j × exp(-β^n × U^n_{ij})) / Σ_k (A^n_k × exp(-β^n × U^n_{ik}))
```

Where:
- `β^n`: Location dispersion parameter
- `A^n_j`: Attractor for sector n in zone j
- `U^n_{ij} = λ^n × (p^n_j + h^n_j) + t^n_{ij}`: Total disutility including transport cost

The attractor is computed as:
```
A^n_i = W^n_i × (Σ_k b^{kn} × X^k_i)^{α^n}
```

#### 4. Total Demand

Total demand for sector n in zone i:

```
D^n_i = D^{n,exog}_i + Σ_m (a^{mn}_i × S^{mn}_i × X^m_i)
```

#### 5. Production

Production is computed from demand and location probabilities:

```
X^n_j = Σ_i (D^n_i × Pr^n_{ij})
```

#### 6. Price Equilibrium

Prices satisfy the input-output equilibrium:

```
p^m_i = VA^m_i + Σ_n (a^{mn}_i × S^{mn}_i × Σ_j (Pr^n_{ij} × (p^n_j + t^{m,n}_{ij})))
```

Where `VA^m_i` is value added for sector m in zone i.

## Calibration Algorithm

The calibration finds shadow prices h that make predicted production match observed production. The algorithm proceeds in two phases:

### Phase 1: Housing (Non-Transportable) Sectors

For land-use sectors (housing, land), location is fixed (Pr = Identity matrix). The problem becomes:

**For each zone i, find h_i such that:**
```
X^n_i(h_i) = X^{n,observed}_i    for all housing sectors n
```

This is solved using **Levenberg-Marquardt least squares** (`scipy.optimize.leastsq`) independently for each zone, enabling parallel computation.

**Residual function:**
```python
def residual_housing(h_i, zone_i):
    X_predicted = calc_induced_prod_housing(h_i, zone_i)
    return X_predicted - X_observed[housing_sectors, zone_i]
```

### Phase 2: Transportable Sectors

For sectors with location choice (β ≠ 0), we optimize the combined variable φ = p + h:

**For each transportable sector n, find φ^n such that:**
```
X^n(φ^n) = X^{n,observed}
```

This is also solved using Levenberg-Marquardt with analytically computed Jacobians.

**Jacobian computation:**

The derivative of production with respect to φ is:

```
∂X^n_j/∂φ^n_k = Σ_i D^n_i × λ^n × β^n × (Pr^n_{ij} × Pr^n_{ik} - δ_{jk} × Pr^n_{ij})
```

Where δ_{jk} is the Kronecker delta.

### Phase 3: Price Recovery

After finding optimal φ = p + h, recover equilibrium prices by solving the linear system:

```
(I - Δ) × p = Λ
```

Where:
- `Δ_{mi,nj} = a^{mn}_i × S^{mn}_i × Pr^n_{ij}`
- `Λ_{mi} = VA^m_i + Σ_n (a^{mn}_i × S^{mn}_i × Σ_j (Pr^n_{ij} × t^{mn}_{ij}))`

Finally, shadow prices are recovered:
```
h^n = φ^n / λ^n - p^n
```

## Algorithm Flow

```
┌─────────────────────────────────────────────┐
│           LCAL Calibration                  │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  1. Load Parameters from Tranus Files       │
│     - L0E: Production, demand, prices       │
│     - L1E: Sector parameters (β, λ, σ, δ)   │
│     - C1S: Transport costs (t_nij)          │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  2. Phase 1: Housing Sectors                │
│     For each zone i (parallel):             │
│       minimize ||X(h_i) - X_obs||²          │
│       using Levenberg-Marquardt             │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  3. Compute Demand & Substitution           │
│     - Demand functions a^{mn}_i             │
│     - Substitution probabilities S^{mn}_i   │
│     - Total demands D^n_i                   │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  4. Phase 2: Transportable Sectors          │
│     For each sector n with β ≠ 0:           │
│       minimize ||X_n(φ_n) - X_n,obs||²      │
│       using Levenberg-Marquardt + Jacobian  │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  5. Phase 3: Price Recovery                 │
│     Solve linear system: (I - Δ)p = Λ       │
│     Recover: h = φ/λ - p                    │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│  6. Output: p, h, X for all sectors/zones   │
└─────────────────────────────────────────────┘
```

## Key Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `beta` | β^n | Location choice dispersion (higher = more deterministic) |
| `lamda` | λ^n | Marginal utility of price |
| `sigma` | σ^m | Substitution dispersion |
| `delta` | δ^{mn} | Demand elasticity |
| `alpha` | α^n | Attractor exponent |
| `demin/demax` | a_min, a_max | Demand function bounds |
| `omega` | ω^{mn} | Substitution weights |
| `t_nij` | t^n_{ij} | Transport disutility (time + cost) |

## Computational Considerations

1. **Parallelization**: Phase 1 (housing sectors) is embarrassingly parallel across zones, using Python's `ProcessPoolExecutor`.

2. **Analytical Jacobians**: The algorithm computes analytical derivatives for the Levenberg-Marquardt solver, significantly improving convergence speed compared to finite differences.

3. **Numerical Stability**: Input normalization divides transport costs and prices by their order of magnitude to improve optimization conditioning.

4. **Convergence**: The algorithm reports R² goodness-of-fit and residual norms for each sector to assess calibration quality.

## Differences from Original Tranus

| Aspect | Original Tranus | PyTranus |
|--------|-----------------|----------|
| Approach | Iterative fixed-point | Optimization (least squares) |
| Shadow prices | Manual adjustment | Automatic calibration |
| Convergence | May not converge | Guaranteed local minimum |
| Jacobians | Not used | Analytical derivatives |
| Parallelization | Single-threaded | Multi-core support |

## References

1. de la Barra, T. (1989). *Integrated Land Use and Transport Modelling*. Cambridge University Press.

2. Capelle, T., et al. (2017). "A Python implementation of the Tranus land use model calibration." *Computers, Environment and Urban Systems*, 66, 1-15. [DOI](https://doi.org/10.1016/j.compenvurbsys.2017.06.006)

3. [Tranus Mathematical Description](http://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnx0cmFudXNtb2RlbHxneDo3YWQzYTk0OTkxN2RlN2Rj)
