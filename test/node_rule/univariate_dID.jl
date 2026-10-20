# UniSGP_dID's rules and energy against its density (see reference.jl), for the gradient operator
# with a one- and a two-dimensional input and the stacked function-and-gradient operator, with a
# point-mass or, for P > 1, a Wishart noise precision.

@testitem "UniSGP_dID: out, v and Wg" setup = [SGPReference] tags = [:rules] begin
    using RxInfer, LinearAlgebra
    using MessagePassingRulesTestUtils
    R = SGPReference
    for f in (R.unisgp_fixture(D = 1, operator = :grad), R.unisgp_fixture(D = 2, operator = :grad), R.unisgp_fixture(D = 1, operator = :joint_fn_grad))
        noises = f.P == 1 ? (f.q_Wg,) : (f.q_Wg, f.q_Wg_wishart)
        full(out, in, Wg) = (out = out, in = in, v = f.q_v, Wg = Wg, θ = f.q_θ)
        pairs = [(out, in) for out in (f.q_out, f.pm_out) for in in (f.q_in, f.pm_in)]
        @test_message_update_rule(
            node = UniSGP_dID, target = :out, algorithm = f.meta, atol = 1e-8, check_type_promotion = false,
            cases = [(q = (in = in, v = f.q_v, Wg = Wg, θ = f.q_θ),) =>
                R.expected_message(UniSGP_dID, f.meta, full(f.q_out, in, Wg), :out; type = f.P == 1 ? :normal : :mvnormal, dim = f.P)
                for in in (f.q_in, f.pm_in) for Wg in noises],
        )
        # the precision carries the rule's 1e-8 jitter
        @test_message_update_rule(
            node = UniSGP_dID, target = :v, algorithm = f.meta, atol = 1e-6, check_type_promotion = false,
            cases = [(q = (out = out, in = in, Wg = Wg, θ = f.q_θ),) => R.expected_message(UniSGP_dID, f.meta, full(out, in, Wg), :v; type = :weighted_symmetric, dim = 5)
                for (out, in) in pairs for Wg in noises],
        )
        @test_message_update_rule(
            node = UniSGP_dID, target = :Wg, algorithm = f.meta, atol = 1e-8, check_type_promotion = false,
            cases = [(q = (out = out, in = in, v = f.q_v, θ = f.q_θ),) => R.expected_message(UniSGP_dID, f.meta, full(out, in, f.q_Wg), :Wg; type = :wishart, dim = f.P)
                for (out, in) in pairs],
        )
    end
end

@testitem "UniSGP_dID: in and θ" setup = [SGPReference] tags = [:rules] begin
    using RxInfer, LinearAlgebra
    using MessagePassingRulesBase
    R = SGPReference
    for f in (R.unisgp_fixture(D = 1, operator = :grad), R.unisgp_fixture(D = 2, operator = :grad), R.unisgp_fixture(D = 1, operator = :joint_fn_grad))
        for out in (f.q_out, f.pm_out)
            q = (out = out, in = f.q_in, v = f.q_v, Wg = f.q_Wg, θ = f.q_θ)
            message = getresult(call_message_update_rule(UniSGP_dID, :in; q = (out = out, v = f.q_v, Wg = f.q_Wg, θ = f.q_θ), algorithm = f.meta))
            @test R.logdensity_gap(message, UniSGP_dID, f.meta, q, :in, f.in_points) < 1e-10
            for in in (f.q_in, f.pm_in)
                q = (out = out, in = in, v = f.q_v, Wg = f.q_Wg, θ = f.q_θ)
                message = getresult(call_message_update_rule(UniSGP_dID, :θ; q = (out = out, in = in, v = f.q_v, Wg = f.q_Wg), algorithm = f.meta))
                @test R.logdensity_gap(message, UniSGP_dID, f.meta, q, :θ, f.θ_points) < 1e-10
            end
        end
    end
end

@testitem "UniSGP_dID: average energy" setup = [SGPReference] tags = [:rules] begin
    using RxInfer, LinearAlgebra
    using MessagePassingRulesTestUtils
    R = SGPReference
    for f in (R.unisgp_fixture(D = 1, operator = :grad), R.unisgp_fixture(D = 2, operator = :grad), R.unisgp_fixture(D = 1, operator = :joint_fn_grad))
        noises = f.P == 1 ? (f.q_Wg,) : (f.q_Wg, f.q_Wg_wishart)
        qs = [(out = out, in = in, v = f.q_v, Wg = Wg, θ = f.q_θ) for out in (f.q_out, f.pm_out) for in in (f.q_in, f.pm_in) for Wg in noises]
        @test_average_energy(
            node = UniSGP_dID, algorithm = f.meta, atol = 1e-8, check_type_promotion = false,
            cases = [(q = q,) => R.expected_energy(UniSGP_dID, f.meta, q) for q in qs],
        )
    end
end
