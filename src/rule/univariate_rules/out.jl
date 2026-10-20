# rule for "out" (univariate case)
@define_message_update_rule(
    node = UniSGP, target = :out, algorithm = UniSGPMeta,
    args = (q[:in]::IN_OUT, q[:v]::MultivariateNormalDistributionsFamily, q[:w]::Any, q[:θ]::PointMass,),
    body = (algo, args) -> begin
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_w = args.q[:w]
        q_θ = args.q[:θ]
        meta = algo
        kernel = getKernel(meta)
        θ = mean(q_θ)
        μ_v = mean(q_v)
        Ψ1_trans = Matrix{Float64}(undef, length(meta.Xu), 1)
        mf = getMeanFn(meta)
        mxu = apply_mean_fn.(meta.Xu, mf)
        Ku_mxu = meta.KuuF \ mxu

        if q_in isa Distribution
            Ψx = approximate_kernel_expectation(meta.method, (x) -> [apply_mean_fn(x, mf)], q_in)[1]
            approximate_kernel_expectation!(Ψ1_trans, meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]), q_in)
        else
            μ_in, Σ_in = mean_cov_vector_matrix(q_in)
            Ψx = apply_mean_fn(μ_in, mf)
            kernelmatrix!(Ψ1_trans, kernel(θ), meta.Xu, [μ_in])
        end

        μ_y = Ψx + jdotavx(Ψ1_trans, μ_v) - jdotavx(Ψ1_trans, Ku_mxu)

        return NormalMeanPrecision(μ_y, mean(q_w))
    end,
)