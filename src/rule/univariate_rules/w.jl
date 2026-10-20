# rule for "w" edge (univariate case)
@define_message_update_rule(
    node = UniSGP, target = :w, algorithm = UniSGPMeta,
    args = (q[:out]::IN_OUT, q[:in]::IN_OUT, q[:v]::MultivariateNormalDistributionsFamily, q[:θ]::PointMass,),
    body = (algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_θ = args.q[:θ]
        meta = algo
        μ_y, Σ_y = mean_cov_scalar_matrix(q_out)
        θ = mean(q_θ)
        kernel = getKernel(meta)
        μ_v, Σ_v = mean_cov_vector_matrix(q_v)
        mf = getMeanFn(meta)
        mxu = apply_mean_fn.(meta.Xu, mf)
        Ku_mxu = meta.KuuF \ mxu
        Rv = Σ_v + μ_v * μ_v'

        if q_in isa Distribution
            Ψx = approximate_kernel_expectation(meta.method, (x) -> [apply_mean_fn(x, mf)], q_in)[1]
            Ψxx = approximate_kernel_expectation(meta.method, (x) -> [apply_mean_fn(x, mf) * apply_mean_fn(x, mf)], q_in)[1]
            Ψ0 = approximate_kernel_expectation(meta.method,(x) -> kernelmatrix(kernel(θ), [x], [x]), q_in)[1]
            Ψ1_trans = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]), q_in)
            Ψ2 = approximate_kernel_expectation(meta.method, (x) -> kernelmatrix(kernel(θ), meta.Xu, [x]) * kernelmatrix(kernel(θ), [x], meta.Xu), q_in) + 1e-8*I
            Ψ3 = approximate_kernel_expectation(meta.method, (x) -> apply_mean_fn(x, mf) * kernelmatrix(kernel(θ), [x], meta.Xu), q_in)
        else
            μ_in, Σ_in = mean_cov_vector_matrix(q_in)
            Ψx = apply_mean_fn(μ_in, mf)
            Ψxx = Ψx^2
            Ψ0 = kernelmatrix(kernel(θ), [μ_in], [μ_in])[1]
            Ψ1_trans = kernelmatrix(kernel(θ), meta.Xu, [μ_in])
            Ψ2 = kernelmatrix(kernel(θ), meta.Xu, [μ_in]) * kernelmatrix(kernel(θ), [μ_in], meta.Xu) + 1e-8*I
            Ψ3 = Ψx * kernelmatrix(kernel(θ), [μ_in], meta.Xu)
        end

        Ψ2_Ku_mxu = Ψ2 * Ku_mxu

        # I1 = Ψ0 - tr(Kuu⁻¹ Ψ2) = Ψ0 - E[k_xu' Kuu⁻¹ k_xu], without the jitter Ψ2 carries; Ψ1' Kuu⁻¹ Ψ1
        # equals it only for a point-mass input
        if q_in isa Distribution
            I1 = Ψ0 - approximate_kernel_expectation(meta.method, (x) -> (α = meta.KuuF.L \ kernelmatrix(kernel(θ), meta.Xu, [x]); [dot(α, α)]), q_in)[1]
        else
            α = meta.KuuF.L \ Ψ1_trans
            I1 = Ψ0 - jdotavx(α, α)
        end

        I4 = (
            μ_y^2 
            + Σ_y[1,1]
            - 2*μ_y*( Ψx + jdotavx(Ψ1_trans, (μ_v - Ku_mxu)) )
            + Ψxx 
            + tr(Rv * Ψ2)
            + jdotavx(Ku_mxu, Ψ2_Ku_mxu)
            + 2*jdotavx(Ψ3, (μ_v - Ku_mxu)) 
            - 2*jdotavx(μ_v, Ψ2_Ku_mxu)
        )
        I4 = clamp(I4, 1e-12, 1e12)
    
        return GammaShapeRate(1.5, 0.5*(I1 + I4))
    end,
)