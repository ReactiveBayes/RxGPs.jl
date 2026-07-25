# =============================================================================
# Natural-gradient message-passing (NGMP) projection demo — UniSGP_dID, 2-D `in`
#
# Reproduces the construction of Lukashchuk et al. (Information Geometry of
# Message Passing) for the factor-to-edge message toward the `in` edge:
#
#   1. the factor forms the *exact* BP log-message ℓ_{b→i}(x)   (non-Gaussian),
#   2. NGMP projects it onto the receiving Gaussian family's tangent at the
#      current `in` marginal q_in  (Eq. 36),
#   3. the projected Gaussian message exp{ξᵀx - ½ xᵀΛ x} is sent onward (Eq. 38).
#
# Setup (so everything is well-defined, not arbitrary):
#   * A known function (a parabola) and its gradient define a *real* GP. We
#     condition the inducing variable `v` on that function + gradient over a
#     grid (repeated `:v` rules, then product) to get a well-defined posterior
#     q_v — this is `define_qv_for_exact_GP` inlined.
#   * We then "observe" a gradient ω* = ∇f(x*) at a test input x*. The exact BP
#     message toward `in` answers *which input x produced this gradient*, so it
#     concentrates around x*.
#   * q_in is the receiving marginal (placed near x*); NGMP projects the exact
#     message at q_in.
#
# We contour-plot, side by side:
#   (a) the exact BP message exp{ℓ_{b→i}(x)}   (from the plain `:in` rule),
#   (b) the receiving marginal q_in,
#   (c) the projected outgoing Gaussian message (from the NGMP `:in` rule).
#
# Run from the repo root:
#   using Pkg; Pkg.activate("examples")
#   include("examples/natural_gradient_projection_demo.jl")
# =============================================================================

using RxGP
using RxInfer
using ReactiveMP
using LinearAlgebra
using Distributions
using ForwardDiff
using Plots

import ReactiveMP: GenericProd
import RxGP: get_simple_kernel_and_params, get_UniSGPMeta

# Gauss-Newton projection of (Gaussian × log-pdf) products via ExponentialFamilyProjection.
# Provides `prod(GenericProd(), ::Normal, ::ContinuousMvLogPdf)` used in the benchmark below.
include(joinpath(@__DIR__, "helper_functions", "projection_prods.jl"))

# ----------------------------------------------------------------------------
# Known function defining the GP: a parabola and its gradient.
# ----------------------------------------------------------------------------
unit_parabola(x; radius = 1.5, scale = 2.0, offset = [0.0, 0.0]) =
    scale * ((x[1] - offset[1])^2 + (x[2] - offset[2])^2 - radius^2)
unit_parabola_grad(x; radius = 1.5, scale = 2.0, offset = [0.0, 0.0]) =
    scale * [2 * (x[1] - offset[1]), 2 * (x[2] - offset[2])]

# ----------------------------------------------------------------------------
# GP configuration
# ----------------------------------------------------------------------------
D       = 2
method  = ReactiveMP.ghcubature(21)
kspec   = :SE
indep   = false
kernel, θ_init, _ = get_simple_kernel_and_params(D; kernel_spec = kspec, independent_SE_lengthscales = indep)
mean_fn = (x) -> 0.0                                    # model the whole function (zero prior mean)

gu = range(-2.0, 2.0; length = 5)
Xu = [[a, b] for a in gu for b in gu]                   # 5×5 inducing grid

q_θ  = PointMass(θ_init)
q_w  = PointMass(1.0)                                   # scalar precision for function observations
q_Wg = PointMass(Matrix(1.0I(D)))                       # matrix precision for gradient observations

meta_val  = get_UniSGPMeta(D; method = method, mean_fn = mean_fn, kernel = kernel, kernel_spec = kspec,
                           mode = :AN, operator = :fn,   independent_SE_lengthscales = indep, Xu = Xu, θ = θ_init)
meta_grad = get_UniSGPMeta(D; method = method, mean_fn = mean_fn, kernel = kernel, kernel_spec = kspec,
                           mode = :AN, operator = :grad, independent_SE_lengthscales = indep, Xu = Xu, θ = θ_init)

# ----------------------------------------------------------------------------
# Well-defined q_v: condition the inducing variable on (f, ∇f) over a disk grid.
# (inline `define_qv_for_exact_GP` using @call_rule toward :v + product)
# ----------------------------------------------------------------------------
function build_qv(; radius, density)
    xr  = range(-radius, radius; length = density)
    pts = [[x, y] for x in xr for y in xr if x^2 + y^2 <= radius^2]
    msgs = Any[]
    for p in pts
        push!(msgs, (@call_rule UniSGP_dID(:v, Marginalisation) (
            q_out = unit_parabola(p),      q_in = p, q_Wg = q_w,  q_θ = q_θ, meta = meta_val)).qv)
        push!(msgs, (@call_rule UniSGP_dID(:v, Marginalisation) (
            q_out = unit_parabola_grad(p), q_in = p, q_Wg = q_Wg, q_θ = q_θ, meta = meta_grad)).qv)
    end
    qv = msgs[1]
    for i in 2:length(msgs)
        qv = ReactiveMP.prod(GenericProd(), qv, msgs[i])
    end
    return qv
end

q_v = build_qv(radius = 2.0, density = 11)
@info "Built well-defined q_v" dim = length(q_v)

# ----------------------------------------------------------------------------
# Observe a gradient ω* = ∇f(x*); the `in` message should localise x around x*.
# ----------------------------------------------------------------------------
x_star = [1.0, -0.5]
ω_star = unit_parabola_grad(x_star)
q_out  = MvNormalMeanCovariance(ω_star, Matrix(1e-3 * I(D)))

# Receiving marginal on the `in` edge — the NGMP projection point.
q_in = MvNormalMeanCovariance(x_star .+ [0.1, 0.05], [0.05 0.01; 0.01 0.04])

# (a) Exact BP message (plain `:in` rule, no q_in dependency).
exact_msg = @call_rule UniSGP_dID(:in, Marginalisation) (
    q_out = q_out, q_v = q_v, q_Wg = q_Wg, q_θ = q_θ, meta = meta_grad)

# (c) NGMP-projected outgoing Gaussian message (`:in` rule WITH the q_in marginal).
proj_msg = @call_rule UniSGP_dID(:in, Marginalisation) (
    q_out = q_out, q_v = q_v, q_Wg = q_Wg, q_θ = q_θ, q_in = q_in, meta = meta_grad)

@info "Projected outgoing message" x_star = x_star mean = mean(proj_msg) cov = cov(proj_msg)

# ----------------------------------------------------------------------------
# Benchmark: NGMP (closed-form) vs Gauss-Newton (iterative project_to).
#
# Both approximate the SAME target by a Gaussian — the (unnormalised) product
#     q_in(x) · exp{ℓ_{b→i}(x)}      (prior message × likelihood, the CVI view).
#   * NGMP   : q_in ⊗ (projected NGMP message)        — closed form, one step.
#   * GaussN : project_to(q_in × exp{ℓ})              — iterative Gauss-Newton.
# ----------------------------------------------------------------------------
timed(f) = (f(); ts = [(@elapsed f()) for _ in 1:5]; minimum(ts))   # warm up, then min of 5

ngmp_post() = ReactiveMP.prod(GenericProd(), q_in, proj_msg)        # Gaussian ⊗ Gaussian
gn_post()   = ReactiveMP.prod(GenericProd(), q_in, exact_msg)       # Gaussian ⊗ log-pdf (Gauss-Newton)

# Total NGMP cost must include forming the projected message itself.
ngmp_full() = begin
    pm = @call_rule UniSGP_dID(:in, Marginalisation) (
        q_out = q_out, q_v = q_v, q_Wg = q_Wg, q_θ = q_θ, q_in = q_in, meta = meta_grad)
    ReactiveMP.prod(GenericProd(), q_in, pm)
end

post_ngmp = ngmp_post()
post_gn   = gn_post()

t_ngmp = timed(ngmp_full)
t_gn   = timed(gn_post)

Δmean = norm(mean(post_ngmp) .- mean(post_gn))
Δcov  = norm(cov(post_ngmp) .- cov(post_gn))

println("\n==================== NGMP vs Gauss-Newton ====================")
println("NGMP   posterior  mean = ", round.(mean(post_ngmp); digits = 5), "  cov = ", round.(cov(post_ngmp); digits = 5))
println("GaussN posterior  mean = ", round.(mean(post_gn);   digits = 5), "  cov = ", round.(cov(post_gn);   digits = 5))
println("Δmean = ", round(Δmean; sigdigits = 4), "   Δcov(Frobenius) = ", round(Δcov; sigdigits = 4))
println("time  NGMP (project+combine) = ", round(t_ngmp * 1e3; sigdigits = 4), " ms")
println("time  Gauss-Newton           = ", round(t_gn   * 1e3; sigdigits = 4), " ms")
println("speedup ≈ ", round(t_gn / t_ngmp; sigdigits = 4), "×")
println("==============================================================\n")

# ----------------------------------------------------------------------------
# Contour plots (densities normalised to peak = 1 for shape comparison)
# ----------------------------------------------------------------------------
xs = range(-0.5, 2.5; length = 220)
ys = range(-2.0, 1.0; length = 220)

function normalised_field(logf)
    Z = [logf([x, y]) for y in ys, x in xs]
    Z .-= maximum(Z)
    return exp.(Z)
end

Z_exact = normalised_field(x -> logpdf(exact_msg, x))
Z_in    = normalised_field(x -> logpdf(q_in, x))
Z_proj  = normalised_field(x -> logpdf(proj_msg, x))
Z_ngmp  = normalised_field(x -> logpdf(post_ngmp, x))
Z_gn    = normalised_field(x -> logpdf(post_gn, x))

m_in   = mean(q_in)
m_proj = mean(proj_msg)

mk_plot(Z, title) = contour(xs, ys, Z; fill = true, levels = 12, c = :viridis,
    title = title, xlabel = "x₁", ylabel = "x₂", colorbar = false, aspect_ratio = :equal)

plt_exact = mk_plot(Z_exact, "(a) Exact BP message  exp{ℓ_{b→i}(x)}")
scatter!(plt_exact, [x_star[1]], [x_star[2]]; m = (:diamond, 6, :orange), label = "x*")
scatter!(plt_exact, [m_in[1]], [m_in[2]]; m = (:circle, 5, :white), label = "q_in mean")

plt_in = mk_plot(Z_in, "(b) Receiving marginal  q_in")
scatter!(plt_in, [m_in[1]], [m_in[2]]; m = (:circle, 5, :white), label = "q_in mean")

plt_proj = mk_plot(Z_proj, "(c) NGMP projected message")
scatter!(plt_proj, [x_star[1]], [x_star[2]]; m = (:diamond, 6, :orange), label = "x*")
scatter!(plt_proj, [m_proj[1]], [m_proj[2]]; m = (:star5, 7, :red), label = "proj mean")

# (d) Posterior q_in·exp{ℓ}: NGMP (closed-form) vs Gauss-Newton (iterative) overlaid.
plt_cmp = contour(xs, ys, Z_ngmp; levels = 6, c = :red, lw = 2, colorbar = false,
    aspect_ratio = :equal, title = "(d) Posterior: NGMP vs Gauss-Newton",
    xlabel = "x₁", ylabel = "x₂", label = "")
contour!(plt_cmp, xs, ys, Z_gn; levels = 6, c = :blue, lw = 2, ls = :dash, label = "")
scatter!(plt_cmp, [mean(post_ngmp)[1]], [mean(post_ngmp)[2]]; m = (:star5, 7, :red),   label = "NGMP (closed-form)")
scatter!(plt_cmp, [mean(post_gn)[1]],   [mean(post_gn)[2]];   m = (:cross, 7, :blue),  label = "Gauss-Newton")

plt = plot(plt_exact, plt_in, plt_proj, plt_cmp; layout = (2, 2), size = (1200, 1000),
    plot_title = "NGMP projection of the `in` message (UniSGP_dID, 2-D)")

savefig(plt, joinpath(@__DIR__, "natural_gradient_projection_demo.png"))
@info "Saved figure" path = joinpath(@__DIR__, "natural_gradient_projection_demo.png")
display(plt)
