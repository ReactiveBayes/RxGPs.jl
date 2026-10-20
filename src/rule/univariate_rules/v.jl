# rule for "v" edge (univariate case)
@define_message_update_rule(
    node = UniSGP, target = :v, algorithm = UniSGPMeta,
    args = (q[:out]::IN_OUT, q[:in]::IN_OUT, q[:w]::NOISE_w, q[:θ]::PointMass,),
    body = (algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_w = args.q[:w]
        q_θ = args.q[:θ]
        meta = algo
        w_bar = mean(q_w)
        θ = mean(q_θ)
        μ_y, Σ_y = mean_cov_scalar_matrix(q_out)
        kernel = getKernel(meta)
        mf = getMeanFn(meta)
        mxu = apply_mean_fn.(meta.Xu, mf)
        Ku_mxu = meta.KuuF \ mxu

        if q_in isa Distribution
            Ψ1_trans = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]), q_in)
            Ψ2 = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]) * kernelmatrix(kernel(θ), [x], meta.Xu), q_in) + 1e-8*I
            Ψ3 = approximate_kernel_expectation(meta.method, (x) -> apply_mean_fn(x, mf) * kernelmatrix(kernel(θ), [x], meta.Xu), q_in)
        else
            μ_in, Σ_in = mean_cov_vector_matrix(q_in)
            mx = apply_mean_fn(μ_in, mf)
            Ψ1_trans = kernelmatrix(kernel(θ), meta.Xu, [μ_in])
            Ψ2 = kernelmatrix(kernel(θ), meta.Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], meta.Xu) + 1e-8*I
            Ψ3 = mx * kernelmatrix(kernel(θ), [μ_in], meta.Xu)
        end

        W_v = w_bar * Ψ2
        ξ_v = vec(w_bar * (μ_y * Ψ1_trans - transpose(Ψ3)) + w_bar * Ψ2 * Ku_mxu)

        return MvNormalWeightedMeanPrecision(ξ_v, W_v)
    end,
)