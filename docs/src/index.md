```@meta
CurrentModule = RxGPs
```

# RxGPs.jl

*Sparse Gaussian process factor nodes for variational message passing on factor graphs.*

RxGPs brings variational sparse Gaussian processes (VSGPs) to [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl).
A GP enters a model as one factor node per observation, connected to shared inducing variables, a
noise precision and the kernel hyperparameters, and inference is message passing on the factor
graph, alongside whatever else the model contains.

## Why RxGPs?

Gaussian processes are flexible non-parametric models, but exact inference costs ``O(N^3)`` in the
number of observations. Sparse variational approximations reduce this with a small set of
**inducing points**, yet putting a sparse GP inside a larger probabilistic model, a state-space
model say, usually takes an inference algorithm derived by hand.

RxGPs packages the sparse GP into factor nodes, [`UniSGP`](@ref), [`UniSGP_dID`](@ref) and
[`MultiSGP`](@ref), whose message passing rules are declared with
[MessagePassingRulesBase.jl](https://github.com/ReactiveBayes/MessagePassingRulesBase.jl) and run
by RxInfer's engine. So you can:

- **compose** GP components with any other model element, in [GraphPPL.jl](https://github.com/ReactiveBayes/GraphPPL.jl)'s
  model syntax;
- **infer** the inducing variables, the noise precisions and uncertain inputs jointly by
  variational message passing, with no derivation by hand;
- **observe** function values, gradients, or both, through `UniSGP_dID`'s linear operators.

## Features

- **Three sparse GP factor nodes**, with message passing rules towards every edge:
  - [`UniSGP`](@ref): a scalar function of a D-dimensional input;
  - [`UniSGP_dID`](@ref): a linear operator of the function, such as its gradient, or the
    function and its gradient together;
  - [`MultiSGP`](@ref): a vector-valued function.
- **Kernels** from [`get_simple_kernel_and_params`](@ref): squared exponential (`:SE`), mixtures of
  them (`:SEn`), spectral mixtures (`:SMn`) and their sums (`:SEn_SMn`).
- **Kernel derivatives** for the gradient operators, by automatic differentiation (`:AD`) or in
  closed form for the squared exponential (`:AN`).
- **Uncertain inputs**: kernel expectations by Gauss–Hermite cubature (the default), the
  spherical-radial cubature [`srcubature`](@ref) or the generalised unscented transform
  [`GenUnscented`](@ref).
- **Prediction** at new inputs with [`predict_GP`](@ref).
- **Hyperparameter learning**: the M-step objectives and their gradients,
  [`neg_log_backwardmess_fast`](@ref) and [`grad_llh_default!`](@ref).

## How to get started?

Head to [Getting started](@ref getting-started), then the [usage examples](@ref examples-usage).

## Contents

```@contents
Pages = [
    "manuals/getting-started.md",
    "library/nodes.md",
    "library/meta.md",
    "library/kernels.md",
    "library/approximation.md",
    "library/prediction.md",
    "library/cache.md",
    "library/helpers.md",
    "library/types.md",
    "examples/overview.md",
    "examples/usage.md",
    "contributing/guide.md",
]
Depth = 2
```

## Citing

If you use RxGPs, please cite H. M. H. Nguyen, İ. Şenöz and B. de Vries, "A Factor Graph Approach
to Variational Sparse Gaussian Processes", *IEEE Open Journal of Signal Processing* 6, 815–837
(2025), [doi:10.1109/OJSP.2025.3585440](https://doi.org/10.1109/OJSP.2025.3585440). The repository's
`CITATION.cff` has the BibTeX details.

## The package

```@docs
RxGPs
```

## Index

```@index
```
