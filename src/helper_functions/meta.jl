export UniSGPMeta, MultiSGPMeta
export getmethod, getMeanFn, getInducingInput, getLm_fn, getKxx_fn, getKxu_fn, getKuuF, getKernel, get_dims_input, get_dims_output, get_dims_theta
export getKuuInverse

#---- Define metas -----# 

"""
    UniSGPMeta

The algorithm of the [`UniSGP`](@ref) and [`UniSGP_dID`](@ref) nodes, which a model gives them with
`where { algorithm = meta }` or `@algorithm`. Construct it with [`get_UniSGPMeta`](@ref).

It is immutable and holds no working memory, so one instance may be shared by every node of a
model, and by models running on several threads.

# Fields
- `method`: the approximation method for the kernel expectations over an uncertain input.
- `mean_fn`: the prior mean function ``m(x)``.
- `Xu`: the M inducing inputs.
- `Lm_fn`: `x -> ` the operator applied to the mean, a vector of length P.
- `Kxx_fn`: `(x, θ) -> ` the operator applied to both arguments of the kernel at `(x, x)`, P×P.
- `Kxu_fn`: `(x, θ, Xu) -> ` the operator applied to the cross-kernel, P×M.
- `KuuF`: the Cholesky factorisation of ``K_{uu}`` at the `θ` the meta was built with.
- `kernel`: the kernel constructor `θ -> k`.
- `dims_input`, `dims_output`, `dims_theta`: D, P and the number of hyperparameters.

!!! note
    The rules towards `out`, `v` and `w` (or `Wg`) use `KuuF`, fixed at the meta's `θ`, with the
    kernel at the mean of `q(θ)`. Keep the two equal: build a new meta when `θ` changes.
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

The algorithm of the [`MultiSGP`](@ref) node, which a model gives it with
`where { algorithm = meta }` or `@algorithm`.

# Arguments
- `method`: the approximation method for the kernel expectations over the input, such as
  [`srcubature`](@ref)`()` or `ghcubature(p)`.
- `Xu`: the M inducing inputs, vectors of the input dimension.
- `Kuu_inverse`: ``K_{uu}^{-1}``, M×M, for the kernel at the current hyperparameters.
- `kernel`: the kernel constructor `θ -> k`.

It is immutable: the kernel expectations the rules compute in place live in each rule's scratch,
which the engine keeps per node, so one instance may be shared by every node of a model. As for
[`UniSGPMeta`](@ref), build a new one when the hyperparameters change.

# Example
```julia
kernel = θ -> θ[1] * with_lengthscale(SEKernel(), θ[2])
θ = [1.0, 1.0]
Xu = [[x, y] for x in 1.0:3.0 for y in 1.0:3.0]
meta = MultiSGPMeta(srcubature(), Xu, inv(kernelmatrix(kernel(θ), Xu) + 1e-8I), kernel)
```
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
