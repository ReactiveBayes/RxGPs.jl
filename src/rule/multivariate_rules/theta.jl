#rule for "θ" edge (multivariate rule)

@define_message_update_rule(
    node = MultiSGP, target = :θ, algorithm = MultiSGPMeta,
    args = (q[:out]::Any, q[:in]::MultivariateGaussianDistributionsFamily, q[:v]::MultivariateGaussianDistributionsFamily, q[:w]::Any,),
    body = (algo, args) -> begin
        q_out = args.q[:out]
        q_in = args.q[:in]
        q_v = args.q[:v]
        q_w = args.q[:w]
        meta = algo
        μ_y = mean(q_out)
        μ_v, Σ_v = mean_cov(q_v)
        R_v = Σ_v + μ_v * μ_v'
        W_bar = mean(q_w)
        kernel = getKernel(meta)
        Xu = getInducingInput(meta)
        C = diageye(length(μ_y))

        Kuu_inverse = (θ) -> cholinv(kernelmatrix(kernel(θ),Xu))
        Ψ_0 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], [x]),q_in)[1]
        Ψ_1 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), [x], Xu),q_in)
        Ψ_2 = (θ) -> approximate_kernel_expectation(getmethod(meta),(x) -> kernelmatrix(kernel(θ), Xu, [x]) * kernelmatrix(kernel(θ), [x], Xu), q_in) + 1e-7*I 

        I1 = (θ) -> kron(C, Ψ_0(θ) - tr(Kuu_inverse(θ) * Ψ_2(θ)))
        Ψ_1_tilde = (θ) -> kron(C, Ψ_1(θ))
        Ψ_3 = (θ) -> kron(W_bar, Ψ_2(θ))
        log_backwardmess = (θ) -> -0.5 * tr(W_bar * I1(θ)) + μ_y' * W_bar * Ψ_1_tilde(θ) * μ_v - 0.5 * tr(Ψ_3(θ) * R_v)
        return ContinuousMultivariateLogPdf(UnspecifiedDomain(),log_backwardmess)
    end,
)