```@meta
CurrentModule = RxGPs
```

# [Prediction](@id prediction-reference)

After running inference, use [`predict_GP`](@ref) to compute the posterior predictive distribution at new test inputs.

## Predictive distribution

```@docs
predict_GP
```

The posterior predictive at a test point ``x_*`` is:

```math
\mu_{f_*} = \tilde{L}\,m(x_*) + \tilde{K}_{x_*u}\bigl(\mu_v - K_{uu}^{-1}\mathbf{m}_u\bigr)
```

```math
\Sigma_{f_*} = \tilde{K}_{x_*x_*} + \tilde{K}_{x_*u}\bigl(\Sigma_v - K_{uu}^{-1}\bigr)\tilde{K}_{x_*u}^\top
```

where:
- ``\tilde{L}`` is the linear operator encoded in the meta (identity, gradient, or joint)
- ``\mu_v, \Sigma_v`` are the posterior mean and covariance of the inducing variable ``\mathbf{v}``
- ``\mathbf{m}_u`` collects the prior mean at the inducing locations

The output dimensionality ``P`` at each test point depends on the operator:

| Operator | ``P`` | `means[i]` | `covs[i]` |
|:---------|:------|:-----------|:----------|
| `:fn` | 1 | 1-vector | 1×1 matrix |
| `:grad` | ``D`` | ``D``-vector | ``D \times D`` matrix |
| `:joint_fn_grad` | ``1+D`` | ``(1+D)``-vector | ``(1+D) \times (1+D)`` matrix |

### Example

[Getting started](@ref getting-started) runs `predict_GP` on a fitted regression and plots its mean
and ``2\sigma`` band, and the [usage examples](@ref examples-usage) do the same after learning
the hyperparameters. The meta passed to `predict_GP` must be built at the same `θ` as `q_θ`.
