# rule for "in" (univariate case)
@define_message_update_rule(
    node = UniSGP, target = :in, algorithm = UniSGPMeta,
    args = (q[:out]::IN_OUT, q[:v]::MultivariateNormalDistributionsFamily, q[:w]::NOISE_w, q[:θ]::PointMass,),
    body = (algo, args) -> begin
        q_out = args.q[:out]
        q_v = args.q[:v]
        q_w = args.q[:w]
        q_θ = args.q[:θ]
        meta = algo
        w_bar = mean(q_w)
        kernel = getKernel(meta)
        θ = mean(q_θ)
        μ_y, Σ_y = mean_cov_scalar_matrix(q_out)
        μ_v, Σ_v = mean_cov(q_v)
        mf = getMeanFn(meta)
        mx = (x) -> apply_mean_fn(x, mf)
        mxu = apply_mean_fn.(meta.Xu, mf)
        Uv = fastcholesky(Σ_v + μ_v * μ_v').U

        B_trans = (x) -> kernelmatrix(kernel(θ), meta.Xu,[x])
        α = (x) -> meta.KuuF.L \ B_trans(x)
        A = (x) -> kernelmatrix(kernel(θ),[x]) .- dot(α(x),α(x))
        β = (x) -> Uv * B_trans(x)
        Ku_mxu = meta.KuuF \ mxu
        mxuT_KuuInvT_BT = (x) -> dot(Ku_mxu, B_trans(x))
        B_KuuInv_mxu = (x) -> dot(B_trans(x), Ku_mxu)

        partA = (x) -> -0.5 * w_bar * A(x)[1]
        partB = (x) -> w_bar * μ_y * ( 
            mx(x) 
            + dot(B_trans(x), μ_v) 
            - B_KuuInv_mxu(x) 
            )
        partC = (x) -> -0.5 * w_bar * ( 
            mx(x)^2 
            + dot(β(x), β(x)) 
            + mxuT_KuuInvT_BT(x)*B_KuuInv_mxu(x) 
            + 2*mx(x)*dot(B_trans(x),μ_v) 
            - 2*mx(x)*B_KuuInv_mxu(x) 
            - 2*dot(B_trans(x),μ_v)*B_KuuInv_mxu(x)
            )

        log_backwardmess = (x) -> partA(x) + partB(x) + partC(x)

        return get_dims_input(meta) < 2 ? ContinuousUnivariateLogPdf(log_backwardmess) : ContinuousMultivariateLogPdf(UnspecifiedDomain(), log_backwardmess)
    end,
)