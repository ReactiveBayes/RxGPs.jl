# MultiSGP's rules and energies against its density (see reference.jl), for every combination of
# input types it has a rule for.

@testitem "MultiSGP: out, v and w" setup = [SGPReference] tags = [:rules] begin
    using RxInfer, LinearAlgebra
    using MessagePassingRulesTestUtils
    R = SGPReference
    m = R.multisgp_fixture()
    full(out, in, w) = (out = out, in = in, v = m.q_v, w = w, θ = m.q_θ)
    @test_message_update_rule(
        node = MultiSGP, target = :out, algorithm = m.meta, atol = 1e-8, check_type_promotion = false,
        cases = [(q = (in = in, v = m.q_v, w = w, θ = m.q_θ),) => R.expected_message(MultiSGP, m.meta, full(m.q_out, in, w), :out; type = :mvnormal, dim = 2)
            for (in, w) in ((m.q_in, m.q_w), (m.q_in, m.pm_w), (m.pm_in, m.q_w))],
    )
    @test_message_update_rule(
        node = MultiSGP, target = :v, algorithm = m.meta, atol = 1e-6, check_type_promotion = false,
        cases = [(q = (out = out, in = in, w = m.q_w, θ = m.q_θ),) => R.expected_message(MultiSGP, m.meta, full(out, in, m.q_w), :v; type = :weighted, dim = 18)
            for (out, in) in ((m.q_out, m.q_in), (m.pm_out, m.q_in), (m.pm_out, m.pm_in))],
    )
    @test_message_update_rule(
        node = MultiSGP, target = :w, algorithm = m.meta, atol = 1e-8, check_type_promotion = false,
        cases = [(q = (out = out, in = in, v = m.q_v, θ = m.q_θ),) => R.expected_message(MultiSGP, m.meta, full(out, in, m.q_w), :w; type = :wishart, dim = 2)
            for (out, in) in ((m.q_out, m.q_in), (m.pm_out, m.q_in), (m.pm_out, m.pm_in))],
    )
end

@testitem "MultiSGP: in and θ" setup = [SGPReference] tags = [:rules] begin
    using RxInfer, LinearAlgebra
    using MessagePassingRulesBase
    R = SGPReference
    m = R.multisgp_fixture()
    for (out, w) in ((m.q_out, m.q_w), (m.pm_out, m.pm_w))
        q = (out = out, in = m.q_in, v = m.q_v, w = w, θ = m.q_θ)
        message = getresult(call_message_update_rule(MultiSGP, :in; q = (out = out, v = m.q_v, w = w, θ = m.q_θ), algorithm = m.meta))
        @test R.logdensity_gap(message, MultiSGP, m.meta, q, :in, m.in_points) < 1e-10
        message = getresult(call_message_update_rule(MultiSGP, :θ; q = (out = out, in = m.q_in, v = m.q_v, w = w), algorithm = m.meta))
        @test R.logdensity_gap(message, MultiSGP, m.meta, q, :θ, m.θ_points) < 1e-10
    end
end

@testitem "MultiSGP: average energy" setup = [SGPReference] tags = [:rules] begin
    using RxInfer, LinearAlgebra
    using MessagePassingRulesTestUtils
    R = SGPReference
    m = R.multisgp_fixture()
    qs = [(out = out, in = in, v = m.q_v, w = w, θ = m.q_θ)
        for (out, in, w) in ((m.q_out, m.q_in, m.q_w), (m.pm_out, m.pm_in, m.q_w), (m.pm_out, m.q_in, m.pm_w), (m.pm_out, m.q_in, m.q_w))]
    @test_average_energy(
        node = MultiSGP, algorithm = m.meta, atol = 1e-8, check_type_promotion = false,
        cases = [(q = q,) => R.expected_energy(MultiSGP, m.meta, q) for q in qs],
    )
end

@testitem "MultiSGP: laplace_in_message" setup = [SGPReference] tags = [:rules] begin
    using RxInfer, LinearAlgebra
    using ForwardDiff, KernelFunctions
    R = SGPReference
    m = R.multisgp_fixture()
    # v such that each output is a bump at x0, f_d(x) ≈ a_d k(x, x0), observed at x0: the log
    # backward message then has its mode near x0
    x0, a = [2.2, 1.8], [1.5, -1.0]
    k = getKernel(m.meta)(mean(m.q_θ))
    Xu = getInducingInput(m.meta)
    v = vcat([a_d * (getKuuInverse(m.meta) * vec(kernelmatrix(k, Xu, [x0]))) for a_d in a]...)
    q = (out = PointMass(a .* kernelmatrix(k, [x0], [x0])[1]), in = MvNormalMeanCovariance([2.0, 2.0], [0.2 0.0; 0.0 0.2]),
        v = MvNormalMeanCovariance(v, 1e-3 * I(length(v))), w = PointMass([20.0 0.0; 0.0 20.0]), θ = m.q_θ)
    message = laplace_in_message(q.out, q.in, q.v, q.w, q.θ, m.meta)
    L = x -> R.expected_logf(MultiSGP, m.meta, q, :in, x)
    mode = mean(message)
    # its mean is the mode of the node's log backward message, its precision the curvature there
    @test norm(mode - x0) < 0.3
    @test norm(ForwardDiff.gradient(L, mode)) < 1e-4
    @test precision(message) ≈ -ForwardDiff.hessian(L, mode) rtol = 1e-4
end
