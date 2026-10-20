```@meta
CurrentModule = RxGPs
```

# [Helper utilities](@id helpers-reference)

RxGPs exports several utility functions used throughout the message passing rules and by end users.

## Mean function application

```@docs
apply_mean_fn
```

Applies the scalar mean function `mf` to an input `x`. Handles dispatch for `PointMass`, `Distribution`, scalar, and vector inputs.

## Mean and covariance extraction

```@docs
mean_cov_scalar_matrix
mean_cov_vector_matrix
```

These convenience functions extract `(mean, cov)` from various input types (`PointMass`, `Real`, distribution) and normalise the output shapes:

| Function | Returns |
|:---------|:--------|
| `mean_cov_scalar_matrix` | `(scalar, 1×1 matrix)` |
| `mean_cov_vector_matrix` | `(vector, matrix)` |

## Linear algebra helpers

```@docs
jdotavx
create_blockmatrix
```

## Hyperparameter optimisation

The kernel hyperparameters ``\theta`` are learned outside message passing, by an M-step that alternates with inference: the functions below compute the negative log of the backward messages towards ``\theta``, summed over the data, and its gradient, for a gradient-based optimiser such as [Flux.jl](https://github.com/FluxML/Flux.jl)'s, as in the [usage examples](@ref examples-usage), or [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

```@docs
neg_log_backwardmess_fast
neg_log_backwardmess_uncertain
neg_log_backwardmess_msg
grad_llh_default!
grad_llh_msg!
neg_log_backwardmess_multi
grad_llh_multi
grad_llh_multi!
```
