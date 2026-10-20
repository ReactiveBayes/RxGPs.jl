# rule for "v" edge (multivariate case)
@define_message_update_rule(
    node = MultiSGP, target = :v, algorithm = MultiSGPMeta,
    args = (q[:out]::MultivariateGaussianDistributionsFamily, q[:in]::MultivariateGaussianDistributionsFamily, q[:w]::Any, q[:θ]::PointMass,),
    scratch = (algo) -> multisgp_scratch(algo),
    body = (scratch, algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_w = args.q[:w]
        q_θ = args.q[:θ]
        meta = algo
        W = mean(q_w)
        μ_y = mean(q_out)
        θ = mean(q_θ)
        cache = scratch.cache
        Xu = getInducingInput(meta)
        kernel = getKernel(meta)
        M = length(Xu) #number of inducing points
        D = length(μ_y) #dimension
        method = getmethod(meta)
        Ψ1_trans = scratch.Ψ1_trans 
        Ψ2 = scratch.Ψ2 
        approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ),Xu, [x]),q_in) 
        approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu) + 1e-12*I, q_in)

        W_v = kron(W, Ψ2) # precision matrix 
        return MvNormalWeightedMeanPrecision(vcat(Ψ1_trans .* mul_A_B!(cache,μ_y',W,1,D)...), W_v)
    end,
)

@define_message_update_rule(
    node = MultiSGP, target = :v, algorithm = MultiSGPMeta,
    args = (q[:out]::PointMass, q[:in]::MultivariateGaussianDistributionsFamily, q[:w]::Any, q[:θ]::PointMass,),
    scratch = (algo) -> multisgp_scratch(algo),
    body = (scratch, algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_w = args.q[:w]
        q_θ = args.q[:θ]
        meta = algo
        W = mean(q_w)
        μ_y = mean(q_out)
        θ = mean(q_θ)
        cache = scratch.cache
        Xu = getInducingInput(meta)
        kernel = getKernel(meta)
        M = length(Xu) #number of inducing points
        D = length(μ_y) #dimension
        method = getmethod(meta)
        Ψ1_trans = scratch.Ψ1_trans 
        Ψ2 = scratch.Ψ2 
        approximate_kernel_expectation!(Ψ1_trans,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu,[x]),q_in) 
        approximate_kernel_expectation!(Ψ2,method,(x) -> kernelmatrix!(similar(Ψ1_trans),kernel(θ), Xu, [x]) * kernelmatrix!(similar(Ψ1_trans'),kernel(θ), [x], Xu)+ 1e-12*I, q_in)

        W_v = kron(W, Ψ2)
        return MvNormalWeightedMeanPrecision(vcat(Ψ1_trans .* mul_A_B!(cache,μ_y',W,1,D)...), W_v)
    end,
)

@define_message_update_rule(
    node = MultiSGP, target = :v, algorithm = MultiSGPMeta,
    args = (q[:out]::PointMass, q[:in]::PointMass, q[:w]::Any, q[:θ]::PointMass,),
    scratch = (algo) -> multisgp_scratch(algo),
    body = (scratch, algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_w = args.q[:w]
        q_θ = args.q[:θ]
        meta = algo
        W = mean(q_w)
        μ_y = mean(q_out)
        θ = mean(q_θ)
        cache = scratch.cache
        Xu = getInducingInput(meta)
        kernel = getKernel(meta)
        M = length(Xu) #number of inducing points
        D = length(μ_y) #dimension
        method = getmethod(meta)
        Ψ1_trans = scratch.Ψ1_trans 
        Ψ2 = scratch.Ψ2 

        kernelmatrix!(Ψ1_trans,kernel(θ),Xu, [mean(q_in)])
        Ψ2 =  kernelmatrix(kernel(θ), Xu, [mean(q_in)]) * kernelmatrix(kernel(θ), [mean(q_in)], Xu) + 1e-12*I

        W_v = kron(W, Ψ2) # precision matrix 
        return MvNormalWeightedMeanPrecision(vcat(Ψ1_trans .* mul_A_B!(cache,μ_y',W,1,D)...), W_v)
    end,
)
