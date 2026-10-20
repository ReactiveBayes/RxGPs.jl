# Inference with RxInfer, the docs' sparse GP regression: one E-step with fixed hyperparameters.
@testsnippet GPRegression begin
    using RxGP, RxInfer, KernelFunctions, LinearAlgebra, StableRNGs

    function gp_regression_problem()
        rng = StableRNG(42)
        N, Nu = 30, 12
        xtrain = sort(rand(rng, N) .* 8 .- 4)
        ytrain = sinc.(xtrain) .+ 0.05 .* randn(rng, N)
        Xu = [[x] for x in range(-4, 4; length = Nu)]
        kernel, θ, _ = get_simple_kernel_and_params(1; kernel_spec = :SE)
        meta = get_UniSGPMeta(1; mean_fn = (x) -> 0.0, kernel = kernel, operator = :fn, Xu = Xu, θ = θ)
        return (; xtrain, ytrain, Nu, θ, meta)
    end

    @model function gp_regression(y, x, θ, mv_prior, gamma_prior)
        v ~ mv_prior
        w ~ gamma_prior
        for i in eachindex(y)
            y[i] ~ UniSGP(x[i], v, w, θ)
        end
    end

    function run_gp_regression(problem, meta; options = (;))
        q_v = MvNormalWeightedMeanPrecision(zeros(problem.Nu), 50 * diageye(problem.Nu))
        q_w = GammaShapeRate(1e-2, 1e-2)
        init = @initialization begin
            q(v) = q_v
            q(w) = q_w
        end
        constraints = @constraints begin
            q(v, w) = q(v)q(w)
        end
        return infer(
            model = gp_regression(θ = problem.θ, mv_prior = q_v, gamma_prior = q_w),
            data = (y = problem.ytrain, x = [[x] for x in problem.xtrain]),
            algorithm = @algorithm(begin UniSGP() -> meta end),
            initialization = init, constraints = constraints, iterations = 5, free_energy = true, options = options,
        )
    end
end

@testitem "engine/UniSGP regression matches ReactiveMP 6" setup = [GPRegression] tags = [:engine] begin
    problem = gp_regression_problem()
    result = run_gp_regression(problem, problem.meta)
    # recorded with RxGP on ReactiveMP 6.6.0 and RxInfer 5.5.3
    @test result.free_energy ≈ [10.862558488737307, 6.348928526502817, 6.10955343265357, 6.101703667728231, 6.101473116863275] rtol = 1e-10
    @test issorted(result.free_energy; rev = true)
end

@testitem "engine/UniSGP rules are pure under the engine's audit" setup = [GPRegression] tags = [:engine] begin
    problem = gp_regression_problem()
    result = run_gp_regression(problem, problem.meta; options = (diagnostics = (check_everything_pure = true,),))
    @test result.free_energy[end] ≈ 6.101473116863275 rtol = 1e-10
end

@testitem "engine/one meta shared by models on several threads (#34)" setup = [GPRegression] tags = [:engine] begin
    problem = gp_regression_problem()
    @test !ismutabletype(typeof(problem.meta))
    serial = run_gp_regression(problem, problem.meta).free_energy
    parallel = fetch.([Threads.@spawn(run_gp_regression(problem, problem.meta).free_energy) for _ in 1:4])
    @test all(fe -> fe == serial, parallel)
end
