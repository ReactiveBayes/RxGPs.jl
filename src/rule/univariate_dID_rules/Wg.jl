# rule for "Wg" edge (univariate dID case)
@define_message_update_rule(
    node = UniSGP_dID, target = :Wg, algorithm = UniSGPMeta,
    args = (q[:out]::IN_OUT, q[:in]::IN_OUT, q[:v]::MultivariateNormalDistributionsFamily, q[:θ]::PointMass,),
    body = (algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_θ = args.q[:θ]
        meta = algo
        θ = mean(q_θ)
        μ_ω, Σ_ω = mean_cov_vector_matrix(q_out)
        μ_in, Σ_in = mean_cov_vector_matrix(q_in)
        μ_v, Σ_v = mean_cov(q_v)
        mf = getMeanFn(meta)
        mxu = apply_mean_fn.(meta.Xu, mf)
        Ku_mxu = meta.KuuF \ mxu
        Rv = μ_v * transpose(μ_v) + Σ_v
        Lm_fn = getLm_fn(meta)
        Kxx_fn = getKxx_fn(meta)
        Kxu_fn = getKxu_fn(meta)

        if q_in isa Distribution
            Ωx = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x), q_in)
            Ω0 = approximate_kernel_expectation(meta.method, (x) -> Kxx_fn(x, θ), q_in)
            Ω1 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu), q_in)
            ΩK = approximate_kernel_expectation(meta.method, (x) -> (K = Kxu_fn(x, θ, meta.Xu); K * (meta.KuuF \ transpose(K))), q_in)
            Ω5 = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x) * transpose(Lm_fn(x)), q_in)
            Ω6 = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x) * transpose(μ_v) * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
            Ω7 = approximate_kernel_expectation(meta.method, (x) -> Lm_fn(x) * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
            Ω8 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * μ_v * transpose(Lm_fn(x)), q_in)
            Ω9 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * Rv * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
            Ω10 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * μ_v * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
            Ω11 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * Ku_mxu * transpose(Lm_fn(x)), q_in)
            Ω12 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * Ku_mxu * transpose(μ_v) * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
            Ω13 = approximate_kernel_expectation(meta.method, (x) -> Kxu_fn(x, θ, meta.Xu) * Ku_mxu * transpose(Ku_mxu) * transpose(Kxu_fn(x, θ, meta.Xu)), q_in)
        else
            Ωx = Lm_fn(μ_in)
            Ω0 = Kxx_fn(μ_in, θ)
            Ω1 = Kxu_fn(μ_in, θ, meta.Xu)
            ΩK = Ω1 * (meta.KuuF \ transpose(Ω1))
            Ω5 = Ωx * transpose(Ωx)
            Ω6 = Ωx * transpose(μ_v) * transpose(Ω1)
            Ω7 = Ωx * transpose(Ku_mxu) * transpose(Ω1)
            Ω8 = Ω1 * μ_v * transpose(Ωx)
            Ω9 = Ω1 * Rv * transpose(Ω1)
            Ω10 = Ω1 * μ_v * transpose(Ku_mxu) * transpose(Ω1)
            Ω11 = Ω1 * Ku_mxu * transpose(Ωx)
            Ω12 = Ω1 * Ku_mxu * transpose(μ_v) * transpose(Ω1)
            Ω13 = Ω1 * Ku_mxu * transpose(Ku_mxu) * transpose(Ω1)
        end


        # G1 = E[K̃_xx - K̃_xu Kuu⁻¹ K̃_ux]
        G1 = Ω0 - ΩK
        A_G2 = q_out isa Distribution ? μ_ω * transpose(μ_ω) + Σ_ω : μ_ω * transpose(μ_ω)
        B_G2 = μ_ω * ( transpose(Ωx) + ( transpose(μ_v) - transpose(Ku_mxu) ) * transpose(Ω1) )
        C_G2 = (Ωx + Ω1 * ( μ_v - Ku_mxu )) * transpose(μ_ω)
        D_G2 = Ω5 + Ω6 - Ω7 + Ω8 + Ω9 - Ω10 - Ω11 - Ω12 + Ω13
        G2 = A_G2 - B_G2 - C_G2 + D_G2

        # Wg is P×P, P the output dimension under the operator (D for :grad, 1 + D for :joint_fn_grad)
        n_Wg = get_dims_output(meta) + 2
        inv_V_Wg = G1 + G2

        return WishartFast(n_Wg, inv_V_Wg)
    end,
)