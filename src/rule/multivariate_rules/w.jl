# rule for w 
@define_message_update_rule(
    node = MultiSGP, target = :w, algorithm = MultiSGPMeta,
    args = (q[:out]::MultivariateNormalDistributionsFamily, q[:in]::MultivariateNormalDistributionsFamily, q[:v]::MultivariateNormalDistributionsFamily, q[:θ]::PointMass,),
    scratch = (algo) -> multisgp_scratch(algo),
    body = (scratch, algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_θ = args.q[:θ]
        meta = algo
        μ_y, Σ_y = mean_cov(q_out)
        μ_v, Σ_v = mean_cov(q_v)
        θ = mean(q_θ)
        Xu = getInducingInput(meta)
        kernel = getKernel(meta)
        cache = scratch.cache
        M = length(Xu) #number of inducing points
        D = length(μ_y) #dimension
        C = diageye(D)
        Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D)
        R_v = create_blockmatrix(Σ_v,D,M)
        μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
        Kuu_inverse = getKuuInverse(meta)
        method = getmethod(meta)

        Ψ0 = scratch.Ψ0
        Ψ1_trans = scratch.Ψ1_trans 
        Ψ2 = scratch.Ψ2 

        approximate_kernel_expectation!(Ψ0,method,(x) -> kernelmatrix!(similar(Ψ0),kernel(θ), [x], [x]),q_in)
        approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu,[x]),q_in) 
        approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in) 

        Ψ0 .-= tr(Kuu_inverse * Ψ2)
        I1 = map(x -> Ψ0[1] * x, C) # kron(C, getindex(Ψ0,1) - trace_A)

        E = getcache(cache,(:E,D))
        map!(yi -> jdotavx(Ψ1_trans, yi),E, μ_v)
        Ψ_4 = getcache(cache,(:Ψ_4,(D,D)))
        map!(Rv_i -> sum(Rv_i .* Ψ2'),Ψ_4, R_v)
        tmp = mul_A_B!(cache,μ_y, E',D)
        tmp += tmp'
        Σ_y += mul_A_B!(cache, μ_y, μ_y',D)
        Ψ_4 += Σ_y
        Ψ_4 -= tmp #this is I2
        Ψ_4 += I1
        return WishartFast(D+2, Ψ_4)
    end,
)

@define_message_update_rule(
    node = MultiSGP, target = :w, algorithm = MultiSGPMeta,
    args = (q[:out]::PointMass, q[:in]::MultivariateNormalDistributionsFamily, q[:v]::MultivariateNormalDistributionsFamily, q[:θ]::PointMass,),
    scratch = (algo) -> multisgp_scratch(algo),
    body = (scratch, algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_θ = args.q[:θ]
        meta = algo
        μ_y = mean(q_out)
        μ_v, Σ_v = mean_cov(q_v)
        θ = mean(q_θ)
        Xu = getInducingInput(meta)
        kernel = getKernel(meta)
        cache = scratch.cache
        M = length(Xu) #number of inducing points
        D = length(μ_y) #dimension
        C = diageye(D)
        Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D)
        R_v = create_blockmatrix(Σ_v,D,M)
        μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
        Kuu_inverse = getKuuInverse(meta)
        method = getmethod(meta)

        Ψ0 = scratch.Ψ0
        Ψ1_trans = scratch.Ψ1_trans 
        Ψ2 = scratch.Ψ2 

        approximate_kernel_expectation!(Ψ0,method,(x) -> kernelmatrix!(similar(Ψ0),kernel(θ), [x], [x]),q_in)
        approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu,[x]),q_in) 
        approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu), q_in) 

        Ψ0 .-= tr(Kuu_inverse * Ψ2)
        I1 = map(x -> Ψ0[1] * x, C) # kron(C, getindex(Ψ0,1) - trace_A)

        E = getcache(cache,(:E,D))
        map!(yi -> jdotavx(Ψ1_trans, yi),E, μ_v)
        Ψ_4 = getcache(cache,(:Ψ_4,(D,D)))
        map!(Rv_i -> sum(Rv_i .* Ψ2'),Ψ_4, R_v)
        tmp = mul_A_B!(cache,μ_y, E',D)
        tmp += tmp'
        Ψ_4 += mul_A_B!(cache, μ_y, μ_y',D)
        Ψ_4 -= tmp #this is I2
        Ψ_4 += I1
        return WishartFast(D+2, Ψ_4)
    end,
)

@define_message_update_rule(
    node = MultiSGP, target = :w, algorithm = MultiSGPMeta,
    args = (q[:out]::PointMass, q[:in]::PointMass, q[:v]::MultivariateNormalDistributionsFamily, q[:θ]::PointMass,),
    scratch = (algo) -> multisgp_scratch(algo),
    body = (scratch, algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_θ = args.q[:θ]
        meta = algo
        μ_y = mean(q_out)
        μ_v, Σ_v = mean_cov(q_v)
        θ = mean(q_θ)
        Xu = getInducingInput(meta)
        kernel = getKernel(meta)
        cache = scratch.cache
        M = length(Xu) #number of inducing points
        D = length(μ_y) #dimension
        C = diageye(D)
        Σ_v += mul_A_B!(cache,μ_v,μ_v',M*D)
        R_v = create_blockmatrix(Σ_v,D,M)
        μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D]  
        Kuu_inverse = getKuuInverse(meta)
        method = getmethod(meta)

        Ψ0 = scratch.Ψ0
        Ψ1_trans = scratch.Ψ1_trans 
        Ψ2 = scratch.Ψ2 

        kernelmatrix!(Ψ0,kernel(θ), [mean(q_in)], [mean(q_in)])
        kernelmatrix!(Ψ1_trans,kernel(θ), Xu,[mean(q_in)])
        Ψ2 = kernelmatrix(kernel(θ), Xu, [mean(q_in)]) * kernelmatrix(kernel(θ), [mean(q_in)], Xu)

        Ψ0 .-= tr(Kuu_inverse * Ψ2)
        I1 = map(x -> Ψ0[1] * x, C) # kron(C, getindex(Ψ0,1) - trace_A)

        E = getcache(cache,(:E,D))
        map!(yi -> jdotavx(Ψ1_trans, yi),E, μ_v)
        Ψ_4 = getcache(cache,(:Ψ_4,(D,D)))
        map!(Rv_i -> sum(Rv_i .* Ψ2'),Ψ_4, R_v)
        tmp = mul_A_B!(cache,μ_y, E',D)
        tmp += tmp'
        Ψ_4 += mul_A_B!(cache, μ_y, μ_y',D)
        Ψ_4 -= tmp #this is I2
        Ψ_4 += I1
        return WishartFast(D+2, Ψ_4)
    end,
)
