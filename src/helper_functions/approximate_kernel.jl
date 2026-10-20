export approximate_kernel_expectation, approximate_kernel_expectation!

"""
    approximate_kernel_expectation(method, g, q)
    approximate_kernel_expectation(method, g, m, V)

The expectation ``\\mathbb{E}_{q(x)}[g(x)]`` of a function `g` returning a number or an array, under
a Gaussian `q`, or its mean `m` and (co)variance `V`, as a weighted sum of `g` at the points of the
approximation `method`: `ghcubature(p)`, [`srcubature`](@ref)`()` or [`GenUnscented`](@ref)`()` (which
takes `q` itself). The rules compute their kernel statistics, such as
``\\Psi_1 = \\mathbb{E}[k(X_u, x)]``, with it.

# Example
```julia
q = NormalMeanVariance(0.0, 1.0)
approximate_kernel_expectation(ghcubature(21), x -> [x^2], q)   # ≈ [1.0]
```
"""
function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, distribution::D) where {D <: NormalDistributionsFamily}
    return approximate_kernel_expectation(method, g, mean(distribution), cov(distribution))
end

function approximate_kernel_expectation(method::AbstractApproximationMethod, g::Function, m::Union{T, AbstractVector{T}}, P::Union{T, AbstractMatrix{T}}) where {T <: Real}
    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)
    gbar = zero(g(m))
    foreach(zip(weights, points)) do (weight, point)
        gbar = gbar .+ weight .* g(point)
    end
    return gbar
end

"""
    approximate_kernel_expectation!(gbar, method, g, q)
    approximate_kernel_expectation!(gbar, method, g, m, V)

[`approximate_kernel_expectation`](@ref) written into the array `gbar`, of the shape `g` returns,
which is overwritten and returned.
"""
function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, distribution::D) where {K <: Array, D <: NormalDistributionsFamily}
    return approximate_kernel_expectation!(gbar, method, g, mean(distribution), cov(distribution))
end

function approximate_kernel_expectation!(gbar::K, method::AbstractApproximationMethod, g::Function, m::Union{T, AbstractVector{T}}, P::Union{T, AbstractMatrix{T}}) where {K <: Array, T <: Real}
    weights = getweights(method, m, P)
    points  = getpoints(method, m, P)

    gbar .= 0
    foreach(zip(weights, points)) do (weight, point)
        axpy!(weight, g(point), gbar) # gbar = gbar + weight * g(point)
    end
    return gbar
end

function approximate_kernel_expectation(method::GenUnscented, g::Function, q::D) where {D <: NormalDistributionsFamily}
    return approximate_expectation(method, q, g)
end

# The in-place counterpart, on the same GenUT sigma points.
function approximate_kernel_expectation!(gbar::K, method::GenUnscented, g::Function, q::D) where {K <: Array, D <: NormalDistributionsFamily}
    gbar .= approximate_expectation(method, q, g)
    return gbar
end