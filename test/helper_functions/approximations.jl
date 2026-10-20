@testitem "helper_functions/approximations/srcubature is exact to degree 3" begin
    using RxGPs, RxInfer, LinearAlgebra
    import RxGPs: approximate_kernel_expectation
    m, V = [0.4, -1.1, 2.0], [1.0 0.3 0.1; 0.3 0.8 -0.2; 0.1 -0.2 0.5]
    q = MvNormalMeanCovariance(m, V)
    # E[x] = m, E[x x'] = V + m m', and odd central moments vanish
    @test approximate_kernel_expectation(srcubature(), x -> x, q) ≈ m
    @test approximate_kernel_expectation(srcubature(), x -> x * x', q) ≈ V + m * m'
    @test approximate_kernel_expectation(srcubature(), x -> [sum((x - m) .^ 3)], q)[1] ≈ 0 atol = 1e-12
    # the points are independent vectors, so they can be collected
    points = collect(RxGPs.getpoints(srcubature(), m, V))
    @test length(unique(points)) == 2 * length(m) + 1
end

@testitem "helper_functions/approximations/GenUnscented in place (#33)" begin
    using RxGPs, RxInfer, KernelFunctions
    import RxGPs: approximate_kernel_expectation, approximate_kernel_expectation!
    k = with_lengthscale(SEKernel(), 0.7)
    Xu = collect(range(-1.0, 1.0; length = 4))
    q = NormalMeanVariance(0.2, 0.3)
    g = x -> kernelmatrix(k, Xu, [x])
    @test approximate_kernel_expectation!(zeros(4, 1), GenUnscented(), g, q) == approximate_kernel_expectation(GenUnscented(), g, q)
    # for a Gaussian, GenUT is exact for polynomials of degree 3
    @test approximate_kernel_expectation(GenUnscented(), x -> [x^3], q)[1] ≈ mean(q)^3 + 3 * mean(q) * var(q)
end

@testitem "helper_functions/approximations/GenUnscented matches the central moments" begin
    using RxGPs, RxInfer, LinearAlgebra
    import RxGPs: approximate_kernel_expectation
    # univariate: mean, variance and the fourth central moment 3σ⁴, for σ² ≠ 1
    q = NormalMeanVariance(0.4, 0.3)
    E = f -> approximate_kernel_expectation(GenUnscented(), x -> [f(x)], q)[1]
    @test E(x -> x) ≈ 0.4
    @test E(x -> (x - 0.4)^2) ≈ 0.3
    @test E(x -> (x - 0.4)^4) ≈ 3 * 0.3^2
    # multivariate normal: mean, covariance and each component's fourth central moment
    m = [0.4, -1.1]
    for V in ([0.5 0.0; 0.0 2.0], [1.0 0.3; 0.3 0.8])
        qm = MvNormalMeanCovariance(m, V)
        Em = f -> approximate_kernel_expectation(GenUnscented(), f, qm)
        @test Em(x -> x) ≈ m
        @test Em(x -> (x - m) * transpose(x - m)) ≈ V
        @test Em(x -> (x - m) .^ 4) ≈ 3 .* diag(V) .^ 2
    end
end

@testitem "helper_functions/approximations/θ rules with nearly co-located inducing points (#35)" begin
    using RxGPs, RxInfer, MessagePassingRulesBase
    kernel, θ, _ = get_simple_kernel_and_params(1; kernel_spec = :SE)
    Xu = [0.0, 1e-9, 1.0]                       # K_uu is singular without the jitter
    meta = get_UniSGPMeta(1; method = ghcubature(11), mean_fn = x -> 0.0, kernel = kernel, operator = :fn, Xu = Xu, θ = θ)
    q = (out = PointMass(0.5), in = PointMass(0.3), v = MvNormalMeanCovariance(zeros(3), [1.0 0 0; 0 1.0 0; 0 0 1.0]), w = GammaShapeRate(2.0, 1.0))
    message = getresult(call_message_update_rule(UniSGP, :θ; q, algorithm = meta))
    @test isfinite(logpdf(message, θ))
    @test isfinite(logpdf(message, 1.2 .* θ))
end
