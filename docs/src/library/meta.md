```@meta
CurrentModule = RxGPs
```

# [Meta objects](@id meta-reference)

Meta objects carry the configuration and precomputed data that the GP factor nodes need during message passing. They are the nodes' algorithms: a model gives one to a node with `where { algorithm = meta }`, or to every node of a kind with `@algorithm`. They are immutable and hold no working memory, so one meta may be shared by every node of a model.

## UniSGPMeta

[`UniSGPMeta`](@ref) is used with both [`UniSGP`](@ref) and [`UniSGP_dID`](@ref) nodes. It stores:

- The approximation method (e.g. [`GenUnscented`](@ref))
- The mean function, kernel, and inducing point locations
- Precomputed kernel matrices and Cholesky factors
- The operator-specific functions (`Lm_fn`, `Kxu_fn`, `Kxx_fn`) for identity, gradient, or joint operators

```@docs
UniSGPMeta
```

### Construction

The recommended way to create a [`UniSGPMeta`](@ref) is via [`get_UniSGPMeta`](@ref):

```@docs
get_UniSGPMeta
```

**Key arguments:**

| Argument | Default | Description |
|:---------|:--------|:------------|
| `D` | (positional) | Input dimensionality |
| `method` | `nothing` | Approximation method (e.g. `GenUnscented()`) for uncertain inputs |
| `mean_fn` | (required) | Prior mean function `m(x) → ℝ` |
| `kernel` | (required) | Kernel constructor `θ → Kernel` (from KernelFunctions.jl) |
| `kernel_spec` | `:SE` | Kernel family: `:SE`, `:SEn`, `:SMn`, or `:SEn_SMn` |
| `mode` | `:AD` | Differentiation mode: `:AD` (autodiff) or `:AN` (analytic, SE only) |
| `operator` | `:fn` | Linear operator: `:fn` (identity, P=1), `:grad` (gradient, P=D), `:joint_fn_grad` (P=1+D) |
| `Xu` | (required) | Vector of inducing point locations |
| `θ` | (required) | Initial kernel hyperparameters |
| `Lm_fn`, `Kxu_fn`, `Kxx_fn` | `nothing` | Override operator functions for a custom linear operator |

### Accessors

```@docs
getmethod
getMeanFn
getInducingInput
getLm_fn
getKxx_fn
getKxu_fn
getKuuF
getKernel
get_dims_input
get_dims_output
get_dims_theta
```

---

## MultiSGPMeta

[`MultiSGPMeta`](@ref) is used with the [`MultiSGP`](@ref) node.

```@docs
MultiSGPMeta
```

### Accessors

```@docs
getKuuInverse
```

!!! note
    Several accessors (`getInducingInput`, `getKernel`, `getmethod`) are shared between `UniSGPMeta` and `MultiSGPMeta`.
