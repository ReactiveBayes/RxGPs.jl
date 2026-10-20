# Changelog

All notable changes to RxGPs.jl are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0]

The first release, as RxGPs (the package was RxGP and unregistered).

### Changed
- RxGPs runs on ReactiveMP 7 and RxInfer 6. Its nodes are declared with MessagePassingRulesBase's
  `@define_factor_node`, and their rules and average energies with `@define_message_update_rule`
  and `@define_average_energy`; it depends on MessagePassingRulesBase,
  MessagePassingRulesApproximations, BayesBase and ExponentialFamily instead of ReactiveMP and
  RxInfer.
- A node's meta is its algorithm: a model gives it with `where { algorithm = meta }` or
  `@algorithm`, and `infer` takes `algorithm =`.
- `UniSGPMeta` and `MultiSGPMeta` are immutable and hold configuration only, so one meta may be
  shared by every node of a model and by models on several threads. `MultiSGPMeta(method, Xu,
  Kuu_inverse, kernel)` replaces the eight-argument constructor.
- Every factorisation of `K_uu` uses one jitter, `1e-8`.
- `ForwardDiff` 1, `Optim` 2, `KernelFunctions` 0.11 and `StatsFuns` 2 are supported, on Julia 1.10
  and later.

### Added
- `srcubature`, the spherical-radial cubature, which ReactiveMP 7 no longer provides.
- `laplace_in_message`, the Laplace approximation of `MultiSGP`'s message towards `in`, which was a
  rule the engine could not select.
- `GenUnscented` accepts a multivariate normal input, and `approximate_kernel_expectation!` has a
  `GenUnscented` method.

### Removed
- `BufferUniSGP` and the metas' kernel-statistics fields and accessors (`getΨx`, …, `getUv`,
  `getcounter`, `getN`, `getGPCache`), the `getcache_lowermatrix` cache, and the unused
  `grad_llh_new!` and `grad_llh_uncertain!`.

### Fixed
- `UniSGP`'s message towards `w` and average energy, and `UniSGP_dID`'s messages towards `Wg` and
  `θ` and average energy, with an uncertain input: `E[k Kuu⁻¹ k']` was computed from `E[k]`.
- `UniSGP_dID`: a univariate Gaussian output's variance was dropped; the message towards `Wg` had
  the wrong degrees of freedom for `:joint_fn_grad`; the message towards `θ` held `K_uu` at the
  meta's `θ`.
- `MultiSGP`'s message towards `θ` no longer adds a jitter to `Ψ₂`; its `out` and `v` messages no
  longer share one buffer between nodes.
- `GenUnscented` takes central moments, so it matches the fourth moment of an input with variance
  other than one, and its multivariate sigma points invert the elementwise powers of the
  covariance factor correctly.
- The `:joint_fn_grad` operator for one-dimensional inputs; `grad_llh_msg!`,
  `get_gradient_Kxu_fn` and `get_gradient_Kxx_fn` without their keyword defaults that referred to
  themselves; an unknown `kernel_spec` is an error.
