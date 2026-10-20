# rule for "θ" edge (univariate dID case)
@define_message_update_rule(
    node = UniSGP_dID, target = :θ, algorithm = UniSGPMeta,
    args = (q[:out]::IN_OUT, q[:in]::IN_OUT, q[:v]::MultivariateNormalDistributionsFamily, q[:Wg]::NOISE_Wg,),
    body = (algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_Wg = args.q[:Wg]
        meta = algo
        μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
        μ_in, Σ_in = mean_cov_vector_matrix(q_in)
        μ_v, Σ_v = mean_cov(q_v)
        Wg_bar = mean(q_Wg)
        mf = getMeanFn(meta)
        mxu = apply_mean_fn.(meta.Xu, mf)
        # K_uu depends on θ, as in the node's b̃ and Ã
        kernel = getKernel(meta)
        KuuF = (θ) -> fastcholesky(kernelmatrix(kernel(θ), meta.Xu) + KUU_JITTER * I)
        Ku_mxu = (θ) -> KuuF(θ) \ mxu
        Rv = μ_v * transpose(μ_v) + Σ_v
        Lm_fn = getLm_fn(meta)
        Kxx_fn = getKxx_fn(meta)
        Kxu_fn = getKxu_fn(meta)

        if q_in isa Distribution
            Ω0 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> Kxx_fn(x, θ), q_in)
            Ω1 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu), q_in)
            ΩK = (θ) -> (F = KuuF(θ); approximate_kernel_expectation(meta.method, (x) -> (K = Kxu_fn(x, θ, meta.Xu); K * (F \ transpose(K))), q_in))
            Ω3 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> transpose(Kxu_fn(x, θ, meta.Xu))*Wg_bar*Kxu_fn(x, θ, meta.Xu), q_in)
            Ω4 = (θ) -> approximate_kernel_expectation(meta.method, (x) -> transpose(Lm_fn(x))*Wg_bar*Kxu_fn(x, θ, meta.Xu), q_in)
        else
            Ω0 = (θ) -> Kxx_fn(μ_in, θ)
            Ω1 = (θ) -> Kxu_fn(μ_in, θ, meta.Xu)
            ΩK = (θ) -> Ω1(θ) * (KuuF(θ) \ transpose(Ω1(θ)))
            Ω3 = (θ) -> transpose(Ω1(θ)) * Wg_bar * Ω1(θ)
            Ω4 = (θ) -> transpose(Lm_fn(μ_in)) * Wg_bar * Ω1(θ)
        end


        # G1 = E[K̃_xx - K̃_xu Kuu⁻¹ K̃_ux]
        G1 = (θ) -> Ω0(θ) - ΩK(θ)
        part_A = (θ) -> (c = Ku_mxu(θ); 2 * dot(Ω4(θ), (μ_v - c)) + dot((transpose(c) - 2*transpose(μ_v)), Ω3(θ)*c) + tr(Ω3(θ) * Rv))
        part_B = (θ) -> 2 * dot(transpose(μ_ω), Wg_bar * Ω1(θ) * (μ_v - Ku_mxu(θ)))

        log_backwardmess = (θ) -> -0.5 * tr( Wg_bar * G1(θ) ) - 0.5 * (part_A(θ) - part_B(θ))

        return get_dims_theta(meta) == 1 ? ContinuousUnivariateLogPdf(log_backwardmess) : ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
    end,
)