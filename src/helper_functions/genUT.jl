export GenUT, GenUnscented, GenUnscentedTransform

"""
    GenUnscented()

The generalized unscented transform (Ebeigbe et al., 2021), an approximation method for
[`approximate_kernel_expectation`](@ref) and the metas' `method`. Its `2D + 1` sigma points and
weights match the input's mean, covariance and the third and fourth central moments of each
component, so it is exact for polynomials of degree three, and of degree four in each component.

It accepts a univariate distribution, with its skewness and kurtosis, or a multivariate normal,
whose components have no skewness and a fourth central moment of `3σᵢ⁴`.

See also [`GenUT`](@ref), [`GenUnscentedTransform`](@ref), [`srcubature`](@ref).
"""
struct GenUnscented <: AbstractApproximationMethod end

"""An alias for the [`GenUnscented`](@ref) approximation method."""
const GenUT = GenUnscented

"""An alias for the [`GenUnscented`](@ref) approximation method."""
const GenUnscentedTransform = GenUnscented

# The sigma points take the third and fourth central moments, E[(x - m)³] and E[(x - m)⁴];
# Distributions' skewness and kurtosis are standardised.
function approximate_expectation(method::GenUnscented, q::UnivariateDistribution, f::F) where {F}
    m, V = mean(q), var(q)
    S = skewness(q) * V^(3 / 2)
    K = kurtosis(q, false) * V^2
    sigma_points, weights = sigma_points_weights(method, m, V, S, K)
    return sum(weights .* f.(sigma_points))
end

function approximate_expectation(method::GenUnscented, q::MultivariateNormalDistributionsFamily, f::F) where {F}
    m, V = mean_cov(q)
    S = zeros(eltype(m), length(m))
    K = 3 .* diag(V) .^ 2
    sigma_points, weights = sigma_points_weights(method, m, V, S, K)
    return sum(weights .* f.(sigma_points))
end

# Univariate sigma points and weights, for the central moments V, S and K
function sigma_points_weights(::GenUnscented, m::Real, V::Real, S::Real, K::Real)
    L     = sqrt(V)
    invL3 = inv(L^3)
    u     = (1/2)*(-S*invL3 + (1/V)*sqrt(4*K - 3*(S^2)/V))
    v     = u + S*invL3
    auxiliary    = inv(v*(u+v))
    sigma_points = (m, m - u*L, m + v*L)
    weights      = (1-auxiliary*(v/u +1)  , (v/u)*auxiliary, auxiliary)

    return (sigma_points, weights)
end

# Multivariate sigma points and weights: L is the lower Cholesky factor of V, and S and K hold the
# components' third and fourth central moments. L∘³ and L∘⁴ are elementwise powers, inverted as
# matrices.
function sigma_points_weights(::GenUnscented, m::AbstractVector, V::AbstractMatrix, S::AbstractVector, K::AbstractVector)
    d     = length(m)
    T     = promote_type(eltype(m), eltype(V), eltype(S), eltype(K))
    L     = cholsqrt(V)
    invL3 = inv(L .^ 3)
    invL4 = inv(L .^ 4)
    discriminant = 4*invL4*K - 3*(invL3*S).^2
    u     = (1/2)*(-invL3*S + sqrt.(discriminant))
    v     = u + invL3*S
    sigma_points = Vector{Vector{T}}(undef, 2 * d + 1)
    weights      = Vector{T}(undef, 2 * d + 1)
    sigma_points[1] = m
    @inbounds for i in 1:d
        @views sigma_points[i+1]  = m - L[:, i]*u[i]
        @views sigma_points[i+d+1] = m + L[:, i]*v[i]
    end
    @inbounds weights[d+2:end] .=  1 ./ v ./ ( u + v) 
    @inbounds weights[2:d+1]   .= weights[d+2:end] .* (v ./ u)
    weights[1] = 1 - sum(weights[2:end])

    return (sigma_points, weights)
end
