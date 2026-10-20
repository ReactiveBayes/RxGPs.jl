# The reference the rule tests compare with: each node's log-density, written from its docstring,
# and its expectations under the marginals, computed without the rules' kernel statistics.
#
# A variational message towards `t` is exp(E_q[log f]) as a function of `t`, up to a constant,
# and an average energy is E_q[-log f]. The log-density is quadratic in the output and in `v`, so
# its expectation over them is exact on degree-3 sigma points; it is affine in the noise precision
# and its log (or log-determinant), so their means suffice. The expectation over the input is the
# meta's own Gauss–Hermite rule, the one the rules use, so the two agree to rounding.
@testmodule SGPReference begin
    using RxGPs, RxInfer, KernelFunctions, LinearAlgebra, StableRNGs, Distributions
    using MessagePassingRulesApproximations: getweights, getpoints
    using ExponentialFamily: WishartFast

    # Points and weights exact for polynomials of degree 3 under a Gaussian, or the point mass
    function sigma_points(q)
        q isa PointMass && return ([mean(q)], [1.0])
        if q isa UnivariateDistribution
            m, s = mean(q), std(q)
            return ([m - s, m + s], [0.5, 0.5])
        end
        m, L = mean(q), cholesky(Symmetric(cov(q))).L
        n = length(m)
        points = vcat([m + sqrt(n) * L[:, i] for i in 1:n], [m - sqrt(n) * L[:, i] for i in 1:n])
        return (points, fill(1 / (2n), 2n))
    end

    # The input's points: the point mass, or the meta's Gauss–Hermite rule
    function input_points(q, method)
        q isa PointMass && return ([mean(q)], [1.0])
        m, V = q isa UnivariateDistribution ? (mean(q), var(q)) : (mean(q), cov(q))
        # the points may share one buffer: copy each as it comes
        return ([copy(p) for p in getpoints(method, m, V)], collect(getweights(method, m, V)))
    end

    # The noise precision enters through its mean and the mean of its log(-determinant)
    noise(q::PointMass) = (mean(q), mean(q) isa Number ? log(mean(q)) : logdet(mean(q)))
    noise(q::GammaShapeRate) = (mean(q), mean(log, q))
    noise(q::Wishart) = (mean(q), mean(logdet, q))

    # E over the product of `qs` (a NamedTuple of marginals, or plain values) of `f(values...)`
    function expectation(f, qs::NamedTuple, method)
        names = keys(qs)
        grids = map(names) do name
            q = qs[name]
            q isa Distribution || q isa PointMass || return ([q], [1.0])
            name === :in ? input_points(q, method) : sigma_points(q)
        end
        total = 0.0
        for idx in Iterators.product(map(g -> eachindex(g[1]), grids)...)
            weight = prod(grids[k][2][idx[k]] for k in eachindex(idx))
            values = NamedTuple{names}(Tuple(grids[k][1][idx[k]] for k in eachindex(idx)))
            total += weight * f(values)
        end
        return total
    end

    # UniSGP: exp(-w A / 2) N(y | b, 1/w), b = m(x) + k_xu' (v - Kuu⁻¹ m_u), A = k(x,x) - k_xu' Kuu⁻¹ k_xu
    function unisgp_logf(meta, Kuu; out, in, v, w, logw, θ)
        k = getKernel(meta)(θ)
        Xu, mf = getInducingInput(meta), getMeanFn(meta)
        kxu = vec(kernelmatrix(k, [in], Xu))
        A = kernelmatrix(k, [in], [in])[1] - dot(kxu, Kuu \ kxu)
        b = apply_mean_fn(in, mf) + dot(kxu, v - Kuu \ apply_mean_fn.(Xu, mf))
        return -w * A / 2 + logw / 2 - log(2π) / 2 - w * (only(out) - b)^2 / 2
    end

    # UniSGP_dID: exp(-tr(W Ã) / 2) N(y | b̃, W⁻¹), b̃ = L m(x) + K̃_xu (v - Kuu⁻¹ m_u), Ã = K̃_xx - K̃_xu Kuu⁻¹ K̃_ux
    function did_logf(meta, Kuu; out, in, v, Wg, logdetWg, θ)
        Xu, mf = getInducingInput(meta), getMeanFn(meta)
        Kxu = getKxu_fn(meta)(in, θ, Xu)
        A = getKxx_fn(meta)(in, θ) - Kxu * (Kuu \ Kxu')
        b = getLm_fn(meta)(in) + Kxu * (v - Kuu \ apply_mean_fn.(Xu, mf))
        W = Wg isa Number ? fill(Wg, 1, 1) : Wg
        r = (out isa Number ? [out] : out) - b
        P = length(r)
        return -tr(W * A) / 2 + logdetWg / 2 - P * log(2π) / 2 - dot(r, W * r) / 2
    end

    # MultiSGP: exp(-tr(W A) / 2) N(y | B v, W⁻¹), B = I_D ⊗ k(x, Xu), A = (k(x,x) - k_xu' Kuu⁻¹ k_xu) I_D
    function multisgp_logf(meta, Kuu_inverse; out, in, v, w, logw, θ)
        k = getKernel(meta)(θ)
        Xu = getInducingInput(meta)
        M, D = length(Xu), length(out)
        kxu = vec(kernelmatrix(k, [in], Xu))
        a = kernelmatrix(k, [in], [in])[1] - dot(kxu, Kuu_inverse * kxu)
        r = out - [dot(kxu, view(v, ((d - 1) * M + 1):(d * M))) for d in 1:D]
        return -tr(w) * a / 2 + logw / 2 - D * log(2π) / 2 - dot(r, w * r) / 2
    end

    noisename(::Type{UniSGP_dID}) = :Wg
    noisename(_) = :w

    # E_q[log f] with the target fixed to `t`; the others are integrated
    function expected_logf(node, meta, q::NamedTuple, target, t)
        θ = target === :θ ? t : mean(q.θ)
        K = kernelmatrix(getKernel(meta)(θ), getInducingInput(meta)) + RxGPs.KUU_JITTER * I
        noise_name = noisename(node)
        nbar, lognbar = target === noise_name ? (t, t isa Number ? log(t) : logdet(t)) : noise(q[noise_name])
        rest = NamedTuple(k => (k === target ? t : q[k]) for k in (:out, :in, :v) )
        method = getmethod(meta)
        if node === UniSGP
            Kuu = target === :θ ? K : getKuuF(meta)
            return expectation(x -> unisgp_logf(meta, Kuu; x..., w = nbar, logw = lognbar, θ = θ), rest, method)
        elseif node === UniSGP_dID
            Kuu = target === :θ ? K : getKuuF(meta)
            return expectation(x -> did_logf(meta, Kuu; x..., Wg = nbar, logdetWg = lognbar, θ = θ), rest, method)
        else
            Kuu_inverse = target === :θ ? inv(K) : getKuuInverse(meta)
            return expectation(x -> multisgp_logf(meta, Kuu_inverse; x..., w = nbar, logw = lognbar, θ = θ), rest, method)
        end
    end

    expected_energy(node, meta, q) = -expected_logf(node, meta, q, :none, nothing)

    # The message read off the reference log-density L of the target, exactly for its family; `dim`
    # is the target's dimension for the vector and matrix families
    function expected_message(node, meta, q, target; type, dim = 1)
        L = t -> expected_logf(node, meta, q, target, t)
        if type === :normal                     # L(t) = -λ (t - μ)² / 2
            λ = -(L(1.0) - 2L(0.0) + L(-1.0))
            return NormalMeanPrecision((L(1.0) - L(-1.0)) / 2 / λ, λ)
        elseif type === :gamma                  # L(w) = a log w - b w
            b = (L(2.0) - L(1.0)) - (L(4.0) - L(2.0))
            a = (L(2.0) - L(1.0) + b) / log(2)
            return GammaShapeRate(a + 1, b)
        elseif type === :wishart                # L(W) = logdet(W) / 2 - tr(S W) / 2: a Wishart of dim + 2 degrees
            E = (i, j) -> (A = zeros(dim, dim); A[i, j] = A[j, i] = 1; A)
            L0 = L(Matrix(1.0I, dim, dim))
            S = [i == j ? log(2) - 2 * (L(I + E(i, i)) - L0) : log(0.75) - 2 * (L(I + E(i, j) / 2) - L0) for i in 1:dim, j in 1:dim]
            return WishartFast(dim + 2, S)
        end
        # a quadratic in a vector: its gradient and Hessian at zero, by central differences, which
        # are exact for a quadratic
        e = i -> (z = zeros(dim); z[i] = 1; z)
        g = [(L(e(i)) - L(-e(i))) / 2 for i in 1:dim]
        H = [(L(e(i) + e(j)) - L(e(i) - e(j)) - L(-e(i) + e(j)) + L(-e(i) - e(j))) / 4 for i in 1:dim, j in 1:dim]
        type === :weighted && return MvNormalWeightedMeanPrecision(g, -H)
        # UniSGP_dID's rule gives its precision as a Symmetric view of the upper triangle
        type === :weighted_symmetric && return MvNormalWeightedMeanPrecision(g, Symmetric(-H, :U))
        return MvNormalMeanPrecision(-H \ g, -H)
    end

    # The rule's log-density differences at `points` against the reference's
    function logdensity_gap(message, node, meta, q, target, points)
        rule = [logpdf(message, p) for p in points]
        ref = [expected_logf(node, meta, q, target, p) for p in points]
        return maximum(abs, (rule .- rule[1]) .- (ref .- ref[1]))
    end

    # Seeded fixtures (#36): each node's meta and marginals of every kind its rules take
    function unisgp_fixture(; D = 1, operator = :fn, P = nothing)
        rng = StableRNG(2026 + D)
        Nu = 5
        Xu = D == 1 ? collect(range(-2.0, 2.0; length = Nu)) : [randn(rng, D) for _ in 1:Nu]
        mean_fn = D == 1 ? (x -> 0.3 * x + 0.1) : (x -> 0.3 * sum(x))
        kernel_spec = operator === :fn ? :SE : :SEn
        kernel, θ, _ = get_simple_kernel_and_params(D; kernel_spec = kernel_spec)
        meta = get_UniSGPMeta(D; method = ghcubature(D == 1 ? 15 : 7), mean_fn = mean_fn, kernel = kernel, kernel_spec = kernel_spec,
            mode = :AN, operator = operator, independent_SE_lengthscales = operator !== :fn, Xu = Xu, θ = θ)
        P = get_dims_output(meta)
        q_v = MvNormalMeanCovariance(randn(rng, Nu), (A = randn(rng, Nu, Nu); A * A' / Nu + 0.5I))
        q_in, pm_in = D == 1 ? (NormalMeanVariance(0.3, 0.4), PointMass(0.7)) :
                      (MvNormalMeanCovariance(randn(rng, D), (A = randn(rng, D, D); A * A' / D + 0.3I)), PointMass(randn(rng, D)))
        q_out, pm_out = P == 1 ? (NormalMeanVariance(0.8, 0.5), PointMass(1.1)) :
                        (MvNormalMeanCovariance(randn(rng, P), (A = randn(rng, P, P); A * A' / P + 0.4I)), PointMass(randn(rng, P)))
        q_w = GammaShapeRate(3.0, 2.0)
        q_Wg = P == 1 ? PointMass(1.7) : PointMass((A = randn(rng, P, P); A * A' / P + 0.6I))
        q_Wg_wishart = P == 1 ? nothing : Wishart(P + 3.0, (A = randn(rng, P, P); A * A' / P + 0.3I) / (P + 3))
        in_points = D == 1 ? [-1.0, 0.0, 0.5, 1.5] : [randn(rng, D) for _ in 1:4]
        θ_points = [θ .* s for s in (1.0, 0.8, 1.3, 1.7)]
        return (; D, P, meta, q_v, q_in, pm_in, q_out, pm_out, q_w, q_Wg, q_Wg_wishart, q_θ = PointMass(θ), in_points, θ_points, rng)
    end

    function multisgp_fixture()
        rng = StableRNG(2031)
        θ = [1.0, 1.2]
        Xu = vec([[Float64(i), Float64(j)] for i in 1:3, j in 1:3])
        kernel = θ -> θ[1] * with_lengthscale(SEKernel(), θ[2])
        Kuu_inverse = inv(kernelmatrix(kernel(θ), Xu) + RxGPs.KUU_JITTER * I)
        meta = MultiSGPMeta(ghcubature(7), Xu, Kuu_inverse, kernel)
        M, D = length(Xu), 2
        q_v = MvNormalMeanCovariance(randn(rng, M * D), (A = randn(rng, M * D, M * D); A * A' / (M * D) + 0.5I))
        return (; D, meta, q_v,
            q_in = MvNormalMeanCovariance([1.6, 2.2], [0.3 0.05; 0.05 0.2]), pm_in = PointMass([2.1, 1.4]),
            q_out = MvNormalMeanCovariance([0.4, -0.3], [0.5 0.1; 0.1 0.4]), pm_out = PointMass([0.6, -0.2]),
            q_w = Wishart(5.0, [0.16 0.02; 0.02 0.12]), pm_w = PointMass([2.0 0.3; 0.3 1.5]), q_θ = PointMass(θ),
            in_points = [[1.0, 1.0], [1.5, 2.5], [2.5, 1.5], [3.0, 3.0]], θ_points = [θ .* s for s in (1.0, 0.8, 1.3, 1.7)])
    end
end
