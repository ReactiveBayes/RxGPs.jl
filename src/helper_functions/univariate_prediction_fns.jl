export predict_GP

#---- Define prediction functions ----#

"""
    predict_GP(; m_in, q_v, q_θ, meta::UniSGPMeta) -> (means, covs)

The posterior predictive distribution of the GP, under the operator of `meta`, at the test inputs
`m_in`, given the marginal `q_v` of the inducing variables (Eq. 30 of the dID-VSGP paper):

```math
\\mu_* = \\tilde{L}m(x_*) + \\tilde{K}_{*u}\\,(\\mu_v - K_{uu}^{-1} m_u), \\qquad
\\Sigma_* = \\tilde{K}_{**} + \\tilde{K}_{*u}\\,(\\Sigma_v - K_{uu}^{-1})\\,\\tilde{K}_{*u}^\\top.
```

# Keywords
- `m_in`: the test inputs, numbers or vectors of length D.
- `q_v`: the marginal of the inducing variables, a multivariate normal.
- `q_θ`: the kernel hyperparameters: a vector, or a distribution whose mean is used.
- `meta`: the [`UniSGPMeta`](@ref) the model ran with, built at the same `θ`.

# Returns
`means`, a vector with the predictive mean at each test input, and `covs`, its covariance. Each
mean has length P, the output dimension of the operator: 1 for `:fn` (so the function's mean at
input `i` is `means[i][1]`), D for `:grad` and 1 + D for `:joint_fn_grad`; each covariance is P×P.
"""
function predict_GP(; m_in::AbstractVector{<:AbstractVector{<:Real}},
                     q_v::MultivariateNormalDistributionsFamily,
                     q_θ::Any,
                     meta::UniSGPMeta)
    θ = typeof(q_θ) <: Union{MultivariateNormalDistributionsFamily, PointMass} ? mean(q_θ) : q_θ
    μ_v, Σ_v = mean_cov(q_v)

    mf     = getMeanFn(meta)
    Xu     = getInducingInput(meta)
    Lm_fn  = getLm_fn(meta)
    Kxu_fn = getKxu_fn(meta)
    Kxx_fn = getKxx_fn(meta)

    mxu    = apply_mean_fn.(Xu, mf)
    Ku_mxu = meta.KuuF \ mxu        # K_uu⁻¹ mᵤ  (M-vector)
    Kuu_inv = meta.KuuF \ I          # K_uu⁻¹     (M×M)

    means = AbstractVector{Float64}[]
    covs  = Matrix{Float64}[]

    for x in m_in
        Lm  = Lm_fn(x)              # P-vector:  L̃m(x*)
        Kxu = Kxu_fn(x, θ, Xu)     # P×M:       K̃_xu(x*)
        Kxx = Kxx_fn(x, θ)         # P×P:       K̃_xx(x*)

        μ = Lm + Kxu * (μ_v - Ku_mxu)
        Σ = Kxx + Kxu * (Σ_v - Kuu_inv) * transpose(Kxu)

        push!(means, μ)
        push!(covs,  Symmetric(Σ))
    end

    return means, covs
end