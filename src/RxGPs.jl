"""
    RxGPs

Sparse Gaussian process factor nodes for variational message passing with
[RxInfer.jl](https://github.com/ReactiveBayes/RxInfer.jl): a GP enters a model as one factor per
observation, and inference over its inducing variables, noise, hyperparameters and uncertain
inputs is message passing on the factor graph.

- [`UniSGP`](@ref): a scalar function of a D-dimensional input.
- [`UniSGP_dID`](@ref): a linear operator of the function, such as its gradient, observed in ``\\mathbb{R}^P``.
- [`MultiSGP`](@ref): a vector-valued function, through the intrinsic coregionalisation model.
- [`UniSGPMeta`](@ref), built by [`get_UniSGPMeta`](@ref), and [`MultiSGPMeta`](@ref): their
  algorithms, which carry the kernel, the inducing inputs and the approximation method.
- [`predict_GP`](@ref): the posterior predictive distribution; [`grad_llh_default!`](@ref) and its
  siblings: the M-step for the kernel hyperparameters.

The nodes and rules are declared with
[MessagePassingRulesBase.jl](https://github.com/ReactiveBayes/MessagePassingRulesBase.jl).
"""
module RxGPs

using MessagePassingRulesBase, BayesBase, ExponentialFamily, Distributions
using MessagePassingRulesBase: AbstractAlgorithm, diageye
using ExponentialFamily: WishartFast
using MessagePassingRulesApproximations: AbstractApproximationMethod, ghcubature
using FastCholesky: fastcholesky, fastcholesky!, cholinv, cholsqrt
using LoopVectorization, Optim, ForwardDiff, StatsFuns, KernelFunctions, LinearAlgebra

import MessagePassingRulesApproximations: getweights, getpoints

# other
include("types.jl")

# helper functions
include("helper_functions/genUT.jl")
include("helper_functions/sphericalradial.jl")
include("helper_functions/approximate_kernel.jl")
include("helper_functions/gp_cache.jl")
include("helper_functions/other_functions.jl")
include("helper_functions/meta.jl")
include("helper_functions/univariate_prediction_fns.jl")

# node 
include("SparseGaussianProcessnode/univariate_node.jl")
include("SparseGaussianProcessnode/univariate_dID_node.jl")
include("SparseGaussianProcessnode/multivariate_node.jl")

# rule
include("rule/univariate_rules/out.jl")
include("rule/univariate_rules/in.jl")
include("rule/univariate_rules/v.jl")
include("rule/univariate_rules/w.jl")
include("rule/univariate_rules/theta.jl")

include("rule/univariate_dID_rules/out.jl")
include("rule/univariate_dID_rules/in.jl")
include("rule/univariate_dID_rules/v.jl")
include("rule/univariate_dID_rules/Wg.jl")
include("rule/univariate_dID_rules/theta.jl")

include("rule/multivariate_rules/out.jl")
include("rule/multivariate_rules/in.jl")
include("rule/multivariate_rules/v.jl")
include("rule/multivariate_rules/w.jl")
include("rule/multivariate_rules/theta.jl")

# The M-step helpers call the rules.
include("helper_functions/derivative.jl")

end
