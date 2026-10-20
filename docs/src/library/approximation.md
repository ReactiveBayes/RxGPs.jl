```@meta
CurrentModule = RxGPs
```

# [Approximation methods](@id approximation-reference)

When the input ``x`` to a GP node is itself uncertain (i.e. a distribution rather than a point), kernel expectations of the form ``\mathbb{E}_{q(x)}[k(x, x')]`` must be approximated. RxGPs provides approximation methods for this purpose.

Every method below works on a Gaussian input. Gauss–Hermite cubature, `ghcubature(p)` from
MessagePassingRulesApproximations, is the default of [`get_UniSGPMeta`](@ref).

## Generalised unscented transform

```@docs
GenUnscented
GenUT
GenUnscentedTransform
```

`GenUnscented` (aliases [`GenUT`](@ref), [`GenUnscentedTransform`](@ref)) extends the unscented
transform: its `2D + 1` sigma points and weights match not only the input's mean ``\mu`` and
covariance ``\Sigma`` but also the third and fourth central moments of each component,
``\mathbb{E}[(x_i - \mu_i)^3]`` and ``\mathbb{E}[(x_i - \mu_i)^4]``. For a univariate input they
are read from its skewness and (non-excess) kurtosis; for a multivariate normal they are ``0`` and
``3\Sigma_{ii}^2``.

## Spherical-radial cubature

```@docs
srcubature
SphericalRadialCubature
```

[`srcubature`](@ref) gives the third-degree spherical-radial cubature, exact for polynomials up to
degree three, with `2D + 1` points: the cheapest of the three in many dimensions, and what the
`MultiSGP` examples use.

## Kernel expectation approximation

```@docs
approximate_kernel_expectation
approximate_kernel_expectation!
```

These functions approximate expectations of the form:

```math
\bar{g} = \mathbb{E}_{q(x)}\bigl[g(x)\bigr] \approx \sum_{i} w_i \, g(\sigma_i)
```

where ``\sigma_i`` are the sigma/quadrature points and ``w_i`` the corresponding weights.

The in-place variant [`approximate_kernel_expectation!`](@ref) writes the result into a pre-allocated array, avoiding allocations in hot loops.

### Supported approximation methods

| Method | Type | Description |
|:-------|:-----|:------------|
| [`GenUnscented`](@ref) | `GenUnscented()` | Generalised unscented transform, matching the third and fourth central moments |
| Gauss-Hermite | `ghcubature(p)`, from MessagePassingRulesApproximations | Tensor-product Gauss–Hermite cubature, `p` points per dimension |
| Spherical-radial | [`srcubature`](@ref)`()` | Third-degree cubature with `2D + 1` points |

!!! tip
    The meta's `method` is used only for an uncertain input. When the input is observed, a number,
    a vector or a `PointMass`, the kernel expectations are exact evaluations. Gauss–Hermite
    cubature with `p` points per dimension costs ``p^D`` kernel evaluations; for larger `D`,
    `srcubature()` or `GenUnscented()` cost ``2D + 1``.
