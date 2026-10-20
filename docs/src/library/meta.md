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
| `method` | `ghcubature(21)` | Approximation method for uncertain inputs: `ghcubature(p)`, `srcubature()` or `GenUnscented()` |
| `mean_fn` | (required) | Prior mean function `m(x) → ℝ` |
| `kernel` | (required) | Kernel constructor `θ → Kernel` (from KernelFunctions.jl) |
| `kernel_spec` | `:SE` | Kernel family: `:SE`, `:SEn`, `:SMn`, or `:SEn_SMn` |
| `mode` | `:AD` | Kernel derivatives for the gradient operators: `:AD` (automatic differentiation) or `:AN` (closed form, for `:SE` and `:SEn`) |
| `independent_SE_lengthscales` | `true` | Whether each input dimension has its own lengthscale |
| `operator` | `:fn` | Linear operator: `:fn` (identity, P=1), `:grad` (gradient, P=D), `:joint_fn_grad` (P=1+D) |
| `Xu` | (required) | The inducing inputs: numbers for `D = 1`, otherwise vectors of length `D` |
| `θ` | (required) | Kernel hyperparameters at which ``K_{uu}`` is factorised; build a new meta when they change |
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
