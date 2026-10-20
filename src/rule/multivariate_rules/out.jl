# rule for "out" edge (multivariate case)

@define_message_update_rule(
    node = MultiSGP, target = :out, algorithm = MultiSGPMeta,
    args = (q[:in]::MultivariateNormalDistributionsFamily, q[:v]::MultivariateNormalDistributionsFamily, q[:w]::Wishart, q[:θ]::PointMass,),
    scratch = (algo) -> multisgp_scratch(algo),
    body = (scratch, algo, args) -> begin
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_w = args.q[:w]
        q_θ = args.q[:θ]
        meta = algo
        μ_v = mean(q_v)
        W = mean(q_w) 
        θ = mean(q_θ)
        Xu = getInducingInput(meta)
        kernel = getKernel(meta)
        cache = scratch.cache
        Ψ1_trans = scratch.Ψ1_trans
        M = length(Xu) #number of inducing points
        D = size(W,1)
        method = getmethod(meta)
        @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
        approximate_kernel_expectation!(Ψ1_trans,method, (x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]), q_in)
        return MvNormalMeanPrecision(map(yi -> jdotavx(Ψ1_trans, yi), μ_v), W)
    end,
)

@define_message_update_rule(
    node = MultiSGP, target = :out, algorithm = MultiSGPMeta,
    args = (q[:in]::MultivariateNormalDistributionsFamily, q[:v]::MultivariateNormalDistributionsFamily, q[:w]::PointMass, q[:θ]::PointMass,),
    scratch = (algo) -> multisgp_scratch(algo),
    body = (scratch, algo, args) -> begin
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_w = args.q[:w]
        q_θ = args.q[:θ]
        meta = algo
        μ_v = mean(q_v)
        W = mean(q_w) 
        θ = mean(q_θ)
        Xu = getInducingInput(meta)
        kernel = getKernel(meta)
        cache = scratch.cache
        Ψ1_trans = scratch.Ψ1_trans
        M = length(Xu) #number of inducing points
        D = size(W,1)
        method = getmethod(meta)
        @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
        approximate_kernel_expectation!(Ψ1_trans,method, (x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ),Xu, [x]), q_in) 
        return MvNormalMeanPrecision(map(yi -> jdotavx(Ψ1_trans, yi), μ_v), W)
    end,
)

@define_message_update_rule(
    node = MultiSGP, target = :out, algorithm = MultiSGPMeta,
    args = (q[:in]::PointMass, q[:v]::MultivariateNormalDistributionsFamily, q[:w]::Wishart, q[:θ]::PointMass,),
    scratch = (algo) -> multisgp_scratch(algo),
    body = (scratch, algo, args) -> begin
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_w = args.q[:w]
        q_θ = args.q[:θ]
        meta = algo
        μ_v = mean(q_v)
        W = mean(q_w) 
        θ = mean(q_θ)
        Xu = getInducingInput(meta)
        kernel = getKernel(meta)
        cache = scratch.cache
        Ψ1_trans = scratch.Ψ1_trans
        M = length(Xu) #number of inducing points
        D = size(W,1)
        method = getmethod(meta)
        @inbounds μ_v = [view(μ_v,i:i+M-1) for i=1:M:M*D] 
        kernelmatrix!(Ψ1_trans, kernel(θ),Xu, [mean(q_in)])
        return MvNormalMeanPrecision(map(yi -> jdotavx(Ψ1_trans, yi), μ_v), W)
    end,
)

