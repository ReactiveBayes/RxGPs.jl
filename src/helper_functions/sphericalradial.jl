export srcubature, SphericalRadialCubature

"""
    SphericalRadialCubature

The third-degree spherical-radial cubature: `2d + 1` points for a `d`-dimensional Gaussian, the
mean and `m ± √(d + 1) Lᵢ` for the columns `Lᵢ` of the covariance's square root, weighted
`1 / (d + 1)` and `1 / (2(d + 1))`. Construct it with [`srcubature`](@ref).
"""
struct SphericalRadialCubature <: AbstractApproximationMethod end

"""
    srcubature()

The [`SphericalRadialCubature`](@ref) approximation method.
"""
srcubature() = SphericalRadialCubature()

function getweights(::SphericalRadialCubature, mean::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T <: Real}
    d = length(mean)
    return Base.Generator(1:(2d + 1)) do i
        return i === (2d + 1) ? 1.0 / (d + 1) : 1.0 / (2.0(d + 1))
    end
end

# The points are new vectors, so they can be collected or kept.
function getpoints(::SphericalRadialCubature, mean::AbstractVector{T}, covariance::AbstractMatrix{T}) where {T <: Real}
    d = length(mean)
    L = cholsqrt(covariance)
    return Base.Generator(1:(2d + 1)) do i
        i === (2d + 1) && return copy(mean)
        unit = zeros(T, d)
        unit[rem(i - 1, d) + 1] = sqrt(d + 1) * (-1)^div(i - 1, d)
        return mean + L * unit
    end
end
