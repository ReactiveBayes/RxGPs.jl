# Natural-gradient (NGMP) rule for "in" (univariate dID case).
#
# This rule implements the natural-gradient message-passing projection of
# Lukashchuk et al. (Information Geometry of Message Passing). The factor first
# forms the *exact* BP log-message ℓ_{b→i}(x) toward the `in` edge (identical to
# the message produced by `in.jl`), then projects it onto the tangent space of
# the receiving edge's Gaussian family at the current marginal q_in. By Eq. 36,
#
#     η_{b→i} = T_i^{λ_i}[ℓ_{b→i}] = ∇_{μ_i} E_{q_{λ_i}}[ℓ_{b→i}(x)],
#
# i.e. the gradient of the expected log-message in the *mean* coordinates of the
# receiving family. For a (multivariate) Gaussian edge family with sufficient
# statistics T(x) = (x, x xᵀ), Bonnet's and Price's theorems turn this into a
# closed-form calculation in terms of expectations of the log-message gradient
# and Hessian under q_in:
#
#     g = E_{q_in}[∇_x ℓ],     H = E_{q_in}[∇²_x ℓ],
#     Λ (precision) = -H,      ξ (potential) = g - H·m,
#
# and the projected outgoing message is the Gaussian exp{ξᵀx - ½ xᵀΛ x}
# (Eq. 38). This is the multivariate generalisation of the unary Eq. 42–43.
# The dependence of the projection point on q_in is the defining NGMP feature:
# the message is re-projected at the current receiving marginal.
@rule UniSGP_dID(:in, Marginalisation) (q_out::IN_OUT, q_v::MultivariateNormalDistributionsFamily, q_Wg::NOISE_Wg, q_θ::PointMass, q_in::MultivariateNormalDistributionsFamily, meta::UniSGPMeta) = begin
    θ = mean(q_θ)
    μ_v, Σ_v = mean_cov(q_v)
    μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
    Wg_bar = mean(q_Wg)
    mf = getMeanFn(meta)
    mxu = apply_mean_fn.(meta.Xu, mf)
    Lm_fn = getLm_fn(meta)
    Kxx_fn = getKxx_fn(meta)
    Kxu_fn = getKxu_fn(meta)

    Ku_mxu = meta.KuuF \ mxu
    Lx = (x) -> Lm_fn(x)
    Kx = (x) -> Kxu_fn(x, θ, meta.Xu)
    KxT = (x) -> transpose(Kx(x))
    Qx = (x) -> Kxx_fn(x, θ) - Kx(x) * (meta.KuuF \ transpose(Kx(x)))
    Rv = μ_v * transpose(μ_v) + Σ_v
    WK = (x) -> Wg_bar * Kx(x)
    KTWK = (x) -> KxT(x) * Wg_bar * Kx(x)
    KTWK_Ku_mxu = (x) -> KTWK(x) * Ku_mxu

    part_A = (x) -> tr(Wg_bar * Qx(x))
    part_B = (x) -> - 2 * transpose(μ_ω) * Wg_bar * ( Lx(x) + Kx(x) * (μ_v - Ku_mxu) )
    part_C = (x) -> (
        transpose(Lx(x)) * Wg_bar * Lx(x)
        + tr(Rv * KTWK(x))
        + (transpose(Ku_mxu) - 2 * transpose(μ_v)) * KTWK_Ku_mxu(x)
        + 2 * transpose(Lx(x)) * WK(x) * (μ_v - Ku_mxu)
    )

    # Exact BP log-message ℓ_{b→i}(x) toward the `in` edge (a scalar of the D-vector x).
    log_backwardmess = (x) -> only(-0.5 * (part_A(x) + part_B(x) + part_C(x)))

    # --- Natural-gradient projection at the receiving marginal q_in (Eq. 36) ---
    # By Bonnet's / Price's theorems, ∇_μ E_q[ℓ] is obtained from the q_in-averaged
    # gradient and Hessian of the log-message.
    m_in, _ = mean_cov_vector_matrix(q_in)
    D = length(m_in)

    grad_ℓ = (x) -> ForwardDiff.gradient(log_backwardmess, x)          # ∇_x ℓ   (D)
    hess_ℓ = (x) -> ForwardDiff.hessian(log_backwardmess, x)           # ∇²_x ℓ  (D×D)

    g = approximate_kernel_expectation(meta.method, grad_ℓ, q_in)      # E_{q_in}[∇_x ℓ]
    H = approximate_kernel_expectation(meta.method, hess_ℓ, q_in)      # E_{q_in}[∇²_x ℓ]
    H = 0.5 * (H + transpose(H))                                       # symmetrise

    Λ = -H + 1e-8 * I                                                  # projected precision  (Λ = -H)
    ξ = g - H * m_in                                                   # projected potential  (ξ = g - H·m)

    return D == 1 ? NormalWeightedMeanPrecision(only(ξ), only(Λ)) : MvNormalWeightedMeanPrecision(ξ, Matrix(Λ))
end
