export UniSGPMeta, MultiSGPMeta
export getmethod, getMeanFn, getInducingInput, getLm_fn, getKxx_fn, getKxu_fn, getKuuF, getKernel, get_dims_input, get_dims_output, get_dims_theta
export getKuuInverse

#---- Define metas -----# 

"""
    UniSGPMeta

The algorithm of the [`UniSGP`](@ref) and [`UniSGP_dID`](@ref) factor nodes, which a model gives
them with `where { algorithm = meta }` or `@algorithm`: the approximation method, the kernel
configuration, the inducing point locations, the factorised ``K_{uu}`` and the operator functions.

It is immutable and holds no working memory, so one instance may be shared by every node of a
model, and by models running on several threads. Construct it with [`get_UniSGPMeta`](@ref).
"""
struct UniSGPMeta{M,F,I,E,D,C,CF,K} <: AbstractAlgorithm
    method          :: M
    mean_fn         :: F
    Xu              :: I
    Lm_fn           :: E        # L*m(x):       operated mean function
    Kxx_fn          :: D        # L1*L2*k(x,x): operated auto-kernel
    Kxu_fn          :: C        # L1*k(x,Xu):   operated cross-kernel
    KuuF            :: CF
    kernel          :: K
    dims_input      :: Int      # D: input dimension
    dims_output     :: Int      # P: output dimension under the linear operator
    dims_theta      :: Int
end

"""Return the approximation method."""
getmethod(meta::UniSGPMeta) = meta.method
"""Return the prior mean function."""
getMeanFn(meta::UniSGPMeta) = meta.mean_fn
"""Return the inducing input locations ``\\mathbf{X}_u``."""
getInducingInput(meta::UniSGPMeta) = meta.Xu
"""Return the operated mean function ``\\tilde{L}m(x)``."""
getLm_fn(meta::UniSGPMeta) = meta.Lm_fn
"""Return the operated auto-kernel function ``\\tilde{K}_{xx}(x,\\theta)``."""
getKxx_fn(meta::UniSGPMeta) = meta.Kxx_fn
"""Return the operated cross-kernel function ``\\tilde{K}_{xu}(x,\\theta,X_u)``."""
getKxu_fn(meta::UniSGPMeta) = meta.Kxu_fn
"""Return the Cholesky factorisation of ``K_{uu}``."""
getKuuF(meta::UniSGPMeta) = meta.KuuF
"""Return the kernel constructor ``\\theta \\mapsto k``."""
getKernel(meta::UniSGPMeta) = meta.kernel
"""Return the output dimension ``P`` under the linear operator."""
get_dims_output(meta::UniSGPMeta) = meta.dims_output
"""Return the input dimension ``D``."""
get_dims_input(meta::UniSGPMeta) = meta.dims_input
"""Return the number of kernel hyperparameters."""
get_dims_theta(meta::UniSGPMeta) = meta.dims_theta

"""
    MultiSGPMeta(method, Xu, Kuu_inverse, kernel)

The algorithm of the [`MultiSGP`](@ref) factor node: the approximation method, the inducing points,
the precomputed ``K_{uu}^{-1}`` and the kernel constructor ``\\theta \\mapsto k``.

It is immutable: the kernel expectations the rules compute in place live in each rule's scratch,
which the engine keeps per node, so one instance may be shared by every node of a model.
"""
struct MultiSGPMeta{M,I,KI,K} <: AbstractAlgorithm
    method      :: M
    Xu          :: I    # inducing inputs
    Kuu_inverse :: KI
    kernel      :: K
end
"""Return the inducing input locations ``\\mathbf{X}_u``."""
getInducingInput(meta::MultiSGPMeta) = meta.Xu
"""Return the precomputed ``K_{uu}^{-1}`` matrix."""
getKuuInverse(meta::MultiSGPMeta) = meta.Kuu_inverse
"""Return the kernel constructor ``\\theta \\mapsto k``."""
getKernel(meta::MultiSGPMeta) = meta.kernel
"""Return the approximation method."""
getmethod(meta::MultiSGPMeta) = meta.method

# The working memory of a MultiSGP rule: the kernel expectations Ψ₀ (1×1), Ψ₁ᵀ (M×1) and Ψ₂
# (M×M), and a GPCache for the products. Write-before-read: every rule fills what it reads.
multisgp_scratch(algo::MultiSGPMeta) = (
    Ψ0 = Matrix{Float64}(undef, 1, 1),
    Ψ1_trans = Matrix{Float64}(undef, length(algo.Xu), 1),
    Ψ2 = Matrix{Float64}(undef, length(algo.Xu), length(algo.Xu)),
    cache = GPCache(),
)
