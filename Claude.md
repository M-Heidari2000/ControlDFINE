# ControlDFINE — SIGReg + Balanced Gramian Extension

## Project Context

This is the ControlDFINE repo (branch `lsp`), an extension of DFINE (by Maryam Shanechi) for control.
GitHub: https://github.com/M-Heidari2000/ControlDFINE/tree/lsp

## Architecture

DFINE learns a latent linear state-space model from high-dimensional observations:
- **Encoder**: y → a (nonlinear, MLP)
- **Decoder**: a → ŷ (nonlinear, MLP)  
- **Latent SSM**: x_{t+1} = Ax_t + Bu_t + w_t, a_t = Cx_t + v_t
- **Kalman filter** runs in x-space for state estimation
- Key files: `dfine/models.py` (Dynamics, Encoder, Decoder), `dfine/train.py`, `dfine/utils.py`

## What We're Building

### 1. SIGReg on Encoder Outputs (replacing decoder)

Replace the decoder-based training with a JEPA-style approach:
- Apply SIGReg regularization on encoder outputs `a` to enforce a ~ N(0, I)
- Train with prediction loss directly in a-space: ||â_{t+k} - a_{t+k}||²
- No decoder needed during training

**Why SIGReg:**
- Prevents encoder collapse (same role the decoder currently plays)
- Makes Kalman filter genuinely optimal (Gaussian assumption satisfied)
- Fixes scale of C implicitly (Cov(a) = CΣ_xC^T + N_a = I)
- Based on LeJEPA/LeWorldModel papers (Balestriero & LeCun)

**SIGReg implementation** (from LeJEPA paper):
- Project embeddings onto M random unit-norm directions
- Compare empirical characteristic function against N(0,1) CF using Epps-Pulley test
- Average over directions
- ~50 lines of PyTorch, O(N) complexity, DDP-friendly
- Recommended defaults: M=1024 projections, 17 integration points, range [-5,5], λ=0.1

**Key theoretical point:** Enforcing Cov(a) = I does NOT restrict dynamics learning.
The autocovariance Cov(a_t, a_{t+k}) = C A^k Σ_x C^T is unconstrained — temporal
structure is fully free. Only the snapshot marginal distribution is standardized.

### 2. Normalized B Parameterization

Constrain B norm structurally (not via loss):
```python
B = c * B_raw / B_raw.norm()  # project onto sphere of radius c
```
This fixes the gauge freedom on the B-side that SIGReg doesn't address.
Together with SIGReg (which fixes C-side), the only remaining gauge freedom
is orthogonal transforms in x-space, which don't affect any control-relevant quantity.

### 3. Balanced Gramian Regularization

With both gauges fixed, add: -α * log(σ_min) where σ_i are Hankel singular values.

Computing HSVs:
1. Solve Lyapunov: W_c = A W_c A^T + B B^T, W_o = A^T W_o A + C^T C
2. HSVs = sqrt(eigenvalues(W_c @ W_o))
3. Loss = -log(min(HSVs))

This was previously tried but had gauge issues (optimizer inflated B/C to
trivially increase σ_min). Now with SIGReg + normalized B, it's well-posed.

### 4. Total Loss

```
L = Σ_k ||â_{t+k} - a_{t+k}||² + λ_filter * filter_loss + λ_sig * SIGReg(a) - α * log(σ_min)
```

Where â_{t+k} comes from Kalman filter posterior rolled forward k steps through (A, B, C).

## Key Design Decisions

- **stable_a**: Use Cayley parameterization for A (ensures spectral radius < 1)
- **No decoder**: SIGReg replaces the decoder's anti-collapse role
- **B normalization**: Structural constraint, not a loss term
- **Prediction in a-space**: ||Cx̂ - a||² instead of ||decoder(Cx̂) - y||²

## What NOT to Do

- Don't compare learned noise (Nx, Na) to true environment noise — states are latent,
  scales are arbitrary (only internal consistency matters)
- Don't use KL consistency loss — experiments showed MSE-only learns dynamics fine
  in non-periodic settings; KL without free nats causes posterior collapse
- Don't use stable_a=True (Cayley) when testing if dynamics bypass occurs —
  Cayley forces scaled-rotation form, preventing the A≈I shortcut

## Environment

Primary test: Torus environment (`envs/torus.py`)
- State: 2D angles, Observation: 3D manifold embedding (possibly rotated)
- Parameters: A (rotation+decay), B (control), Ns (process noise), No (obs noise)
- periodic=True triggers Nx inflation pathology; periodic=False does not

## Papers Referenced

- **DFINE**: Shanechi's original model
- **LeJEPA** (Balestriero & LeCun 2025): Proves isotropic Gaussian is optimal embedding
  distribution. Introduces SIGReg (Sketched Isotropic Gaussian Regularization)
- **LeWorldModel** (Maes et al. 2026): Applies LeJEPA to world models for control.
  Uses SIGReg + prediction loss, no decoder, plans via CEM in latent space
- **FCCA** (Kumar et al. 2024): Shows PCA = feedforward controllability,
  introduces feedback controllability analysis. Non-normality determines divergence.