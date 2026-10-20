# UniSGP's rules and energy against its density (see reference.jl): every combination of a
# Gaussian or point-mass input and output, for a one- and a two-dimensional input. The tables
# check the rules that return a distribution; the rules towards `in` and `θ` return a log-density,
# compared at points. The rules are written for Float64, so the tables do not check promotion.

@testitem "UniSGP: out, v and w" setup = [SGPReference] tags = [:rules] begin
    using RxInfer, LinearAlgebra
    using MessagePassingRulesTestUtils
    R = SGPReference
    for f in (R.unisgp_fixture(D = 1), R.unisgp_fixture(D = 2))
        full(out, in) = (out = out, in = in, v = f.q_v, w = f.q_w, θ = f.q_θ)
        pairs = [(out, in) for out in (f.q_out, f.pm_out) for in in (f.q_in, f.pm_in)]
        @test_message_update_rule(
            node = UniSGP, target = :out, algorithm = f.meta, atol = 1e-8, check_type_promotion = false,
            cases = [(q = (in = in, v = f.q_v, w = f.q_w, θ = f.q_θ),) => R.expected_message(UniSGP, f.meta, full(f.q_out, in), :out; type = :normal) for in in (f.q_in, f.pm_in)],
        )
        # the precision carries the rule's 1e-8 jitter on Ψ₂
        @test_message_update_rule(
            node = UniSGP, target = :v, algorithm = f.meta, atol = 1e-6, check_type_promotion = false,
            cases = [(q = (out = out, in = in, w = f.q_w, θ = f.q_θ),) => R.expected_message(UniSGP, f.meta, full(out, in), :v; type = :weighted, dim = 5) for (out, in) in pairs],
        )
        @test_message_update_rule(
            node = UniSGP, target = :w, algorithm = f.meta, atol = 1e-6, check_type_promotion = false,
            cases = [(q = (out = out, in = in, v = f.q_v, θ = f.q_θ),) => R.expected_message(UniSGP, f.meta, full(out, in), :w; type = :gamma) for (out, in) in pairs],
        )
    end
end

@testitem "UniSGP: in and θ" setup = [SGPReference] tags = [:rules] begin
    using RxInfer, LinearAlgebra
    using MessagePassingRulesBase
    R = SGPReference
    for f in (R.unisgp_fixture(D = 1), R.unisgp_fixture(D = 2))
        for out in (f.q_out, f.pm_out)
            q = (out = out, in = f.q_in, v = f.q_v, w = f.q_w, θ = f.q_θ)
            message = getresult(call_message_update_rule(UniSGP, :in; q = (out = out, v = f.q_v, w = f.q_w, θ = f.q_θ), algorithm = f.meta))
            @test message isa (f.D == 1 ? ContinuousUnivariateLogPdf : ContinuousMultivariateLogPdf)
            @test R.logdensity_gap(message, UniSGP, f.meta, q, :in, f.in_points) < 1e-10
            for in in (f.q_in, f.pm_in)
                q = (out = out, in = in, v = f.q_v, w = f.q_w, θ = f.q_θ)
                message = getresult(call_message_update_rule(UniSGP, :θ; q = (out = out, in = in, v = f.q_v, w = f.q_w), algorithm = f.meta))
                @test R.logdensity_gap(message, UniSGP, f.meta, q, :θ, f.θ_points) < 1e-10
            end
        end
    end
end

@testitem "UniSGP: average energy" setup = [SGPReference] tags = [:rules] begin
    using RxInfer, LinearAlgebra
    using MessagePassingRulesTestUtils
    R = SGPReference
    for f in (R.unisgp_fixture(D = 1), R.unisgp_fixture(D = 2))
        qs = [(out = out, in = in, v = f.q_v, w = f.q_w, θ = f.q_θ) for out in (f.q_out, f.pm_out) for in in (f.q_in, f.pm_in)]
        @test_average_energy(
            node = UniSGP, algorithm = f.meta, atol = 1e-6, check_type_promotion = false,
            cases = [(q = q,) => R.expected_energy(UniSGP, f.meta, q) for q in qs],
        )
    end
end
