# RxGPs.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://reactivebayes.github.io/RxGPs.jl/dev/)
[![CI](https://github.com/ReactiveBayes/RxGPs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ReactiveBayes/RxGPs.jl/actions/workflows/CI.yml?query=branch%3Amain)

**Sparse Gaussian process factor nodes for variational message passing on factor graphs.**

RxGPs brings variational sparse Gaussian processes (VSGPs) to [RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl).
A GP enters a model as one factor node per observation, connected to shared inducing variables,
a noise precision and the kernel hyperparameters. Inference over all of them, and over uncertain
inputs, is then message passing on the factor graph, alongside whatever else the model contains.

## Factor nodes

| Node | Observes | Description |
|:-----|:---------|:------------|
| `UniSGP` | `y ∈ ℝ` | A scalar function of a D-dimensional input |
| `UniSGP_dID` | `ỹ ∈ ℝᴾ` | A linear operator of the function, such as its gradient, or the function and its gradient together (decoupled inter-domain VSGP) |
| `MultiSGP` | `y ∈ ℝᴰ` | A vector-valued function, through the intrinsic coregionalisation model |

Each node takes a meta as its algorithm (`UniSGPMeta`, built with `get_UniSGPMeta`, or
`MultiSGPMeta`), which carries the kernel, the inducing inputs and the approximation method for
uncertain inputs. The nodes and their rules are declared with
[MessagePassingRulesBase.jl](https://github.com/ReactiveBayes/MessagePassingRulesBase.jl).

## Installation

RxGPs needs Julia 1.10 or later and RxInfer 6. It is not yet registered; install it from GitHub,
with RxInfer and the packages the example uses:

```julia
using Pkg
Pkg.add(url = "https://github.com/ReactiveBayes/RxGPs.jl")
Pkg.add(["RxInfer", "KernelFunctions", "Flux"])
```

## Quick start

GP regression of a noisy `sinc`, with the kernel hyperparameters learned by alternating VMP
inference (E-step) and gradient steps on them (M-step):

```julia
using RxInfer, RxGPs, KernelFunctions, Flux, Random

rng = Xoshiro(42)
N, Nu = 30, 20
xtrain = sort(rand(rng, N) .* 8 .- 4)
ytrain = sinc.(xtrain) .+ 0.05 .* randn(rng, N)
Xu = [[x] for x in range(-4, 4; length = Nu)]    # inducing inputs

D = 1
mean_fn = (x) -> 0.0
kernel, θ_init, _ = get_simple_kernel_and_params(D; kernel_spec = :SE)

@model function gp_regression(y, x, θ, mv_prior, gamma_prior)
    v ~ mv_prior
    w ~ gamma_prior
    for i in eachindex(y)
        y[i] ~ UniSGP(x[i], v, w, θ)
    end
end

constraints = @constraints begin
    q(v, w) = q(v)q(w)
end

θ_opt, q_v, q_w = let θ = copy(θ_init),
        q_v = MvNormalWeightedMeanPrecision(zeros(Nu), 50 * diageye(Nu)),
        q_w = GammaShapeRate(1e-2, 1e-2),
        state = Flux.setup(Flux.AdaMax(0.01), θ),
        grad = similar(θ)
    for _ in 1:15
        # E-step: VMP with the meta built at the current θ, from the previous posterior
        meta = get_UniSGPMeta(D; mean_fn = mean_fn, kernel = kernel, Xu = Xu, θ = θ)
        initialization = @initialization begin
            q(v) = q_v
            q(w) = q_w
        end
        result = infer(
            model = gp_regression(θ = θ, mv_prior = q_v, gamma_prior = q_w),
            data = (y = ytrain, x = [[x] for x in xtrain]),
            algorithm = @algorithm(begin UniSGP() -> meta end),
            initialization = initialization,
            constraints = constraints,
            iterations = 2,
            returnvars = KeepLast(),
        )
        q_v, q_w = result.posteriors[:v], result.posteriors[:w]

        # M-step: gradient steps on the kernel hyperparameters
        for _ in 1:10
            grad_llh_default!(grad, θ; y_data = ytrain, x_y_data = [[x] for x in xtrain], x_ω_data = nothing,
                q_v = q_v, q_w = q_w, kernel = kernel, mean_fn = mean_fn, Xu = Xu)
            Flux.update!(state, θ, grad)
        end
    end
    (θ, q_v, q_w)
end

# The predictive mean and variance of the function on a test grid
xtest = [[x] for x in range(-6, 6; length = 200)]
meta = get_UniSGPMeta(D; mean_fn = mean_fn, kernel = kernel, Xu = Xu, θ = θ_opt)
means, covs = predict_GP(m_in = xtest, q_v = q_v, q_θ = θ_opt, meta = meta)
```

`means[i][1]` and `covs[i][1, 1]` are the mean and variance at `xtest[i]`. To observe gradients
as well, add `UniSGP_dID` nodes with `operator = :grad` and a Wishart noise precision; the
[usage examples](https://reactivebayes.github.io/RxGPs.jl/dev/examples/usage/) show both.

## Features

- **Composable**: GP nodes combine with any other node in a [GraphPPL.jl](https://github.com/ReactiveBayes/GraphPPL.jl)
  model, such as state-space dynamics or non-Gaussian likelihoods.
- **Variational inference** over the inducing variables, the noise precision and uncertain inputs;
  the kernel hyperparameters by an M-step, with the gradients the package provides.
- **Inter-domain observations**: gradients, or the function and its gradient together, through
  `UniSGP_dID` and a linear operator, or a custom one.
- **Kernels**: squared exponential, mixtures of them, spectral mixtures and their sums, from
  [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl), with
  derivatives by automatic differentiation or, for the squared exponential, in closed form.
- **Uncertain inputs** through Gauss–Hermite or spherical-radial cubature, or the generalised
  unscented transform.

## Documentation

The documentation is at **[reactivebayes.github.io/RxGPs.jl](https://reactivebayes.github.io/RxGPs.jl/dev/)**:
the nodes, their metas, the kernels and the approximation methods, with runnable examples.

## Examples

The `examples/` directory holds three Jupyter notebooks: GP regression, a GP state-space model
with `MultiSGP`, and regression with function and gradient observations and an arbitrary mean
function. They have an environment of their own; set it up once from the repository root:

```julia
using Pkg
Pkg.activate("examples")
Pkg.develop(path = ".")    # on Julia 1.11 and later, Pkg.instantiate() alone also links it
Pkg.instantiate()
```

## Citing

If you use RxGPs in your work, please cite:

```bibtex
@article{nguyen2025factor,
  author  = {Nguyen, Hoang M. H. and {\c{S}}en{\"o}z, {\.I}smail and de Vries, Bert},
  title   = {A Factor Graph Approach to Variational Sparse {G}aussian Processes},
  journal = {IEEE Open Journal of Signal Processing},
  volume  = {6},
  pages   = {815--837},
  year    = {2025},
  doi     = {10.1109/OJSP.2025.3585440}
}
```

The `UniSGP_dID` node implements A. H. Ledbetter, H. M. H. Nguyen and B. de Vries, "A
Factor-Graph Approach to Decoupled Inter-Domain Variational Sparse Gaussian Processes with
Arbitrary Mean Functions" (under review). `CITATION.cff` has both.

## License

MIT; see [LICENSE](LICENSE).
