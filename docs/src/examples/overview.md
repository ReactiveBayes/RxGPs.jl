```@meta
CurrentModule = RxGPs
```

# [Examples](@id examples-overview)

The repository's [`examples/`](https://github.com/ReactiveBayes/RxGPs.jl/tree/main/examples)
directory has three Jupyter notebooks, each an end-to-end workflow. The
[usage examples](@ref examples-usage) on this site run as the documentation builds.

## GP regression

[`examples/GPRegression.ipynb`](https://github.com/ReactiveBayes/RxGPs.jl/blob/main/examples/GPRegression.ipynb):
sparse GP regression of scalar observations.

- Inducing inputs and a squared-exponential kernel.
- A model of [`UniSGP`](@ref) nodes, inferred by variational message passing with RxInfer.
- The kernel hyperparameters learned by alternating inference with gradient steps.
- Prediction with [`predict_GP`](@ref).

## GP state-space model

[`examples/GPSSM.ipynb`](https://github.com/ReactiveBayes/RxGPs.jl/blob/main/examples/GPSSM.ipynb):
a state-space model whose transition is a GP, learned from noisy observations of a pendulum.

- [`MultiSGP`](@ref) nodes for the vector-valued transition, with a [`MultiSGPMeta`](@ref) and
  spherical-radial cubature over the uncertain previous state.
- Smoothing by variational message passing over the whole sequence, and the kernel
  hyperparameters learned with [`grad_llh_multi!`](@ref).
- A comparison with a known transition through RxInfer's unscented transform, and one-step
  predictions with [`MultiSGP`](@ref)'s rule towards `out`.

## GP regression with gradient observations and an arbitrary mean function

[`examples/GP_Regression_AMF_Grads.ipynb`](https://github.com/ReactiveBayes/RxGPs.jl/blob/main/examples/GP_Regression_AMF_Grads.ipynb):
regression from function values and gradients, with an arbitrary mean function (AMF).

- [`UniSGP`](@ref) nodes for the values and [`UniSGP_dID`](@ref) nodes with the `:grad` operator
  for the gradients, sharing the inducing variables.
- Optionally noisy inputs, inferred by message passing.
- The kernel hyperparameters learned by gradient steps, and the predictive error compared.

!!! tip
    The notebooks have an environment of their own. Clone the repository and set it up once from
    its root:
    ```julia
    using Pkg
    Pkg.activate("examples")
    Pkg.develop(path = ".")    # on Julia 1.11 and later, Pkg.instantiate() alone also links it
    Pkg.instantiate()
    ```
    Then open a notebook with Jupyter or VS Code; it activates the environment itself.
