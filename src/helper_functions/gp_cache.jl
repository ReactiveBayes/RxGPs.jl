export GPCache
export getcache
export mul_A_B!, mul_A_B_A!, mul_A_B_At!, mul_A_v!
#---- Define cache ----#

"""
    GPCache()

Working memory for the in-place products below: matrices and vectors keyed by a label and their
size, allocated on first use and reused after. A [`MultiSGP`](@ref) rule keeps one in its scratch,
which the engine holds per node.

!!! warning
    A buffer is reused by every later call with the same label and size, so a result read from the
    cache stays valid only until the next such call. Copy it to keep it.
"""
struct GPCache
    cache_matrices::Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}
    cache_vectors::Dict{Tuple{Symbol, Int}, Vector{Float64}}
end

GPCache() = GPCache(Dict{Tuple{Symbol, Tuple{Int, Int}}, Matrix{Float64}}(), Dict{Tuple{Symbol, Int}, Vector{Float64}}())

"""
    getcache(cache::GPCache, label::Tuple{Symbol, Tuple{Int, Int}})
    getcache(cache::GPCache, label::Tuple{Symbol, Int})

The matrix of size `label[2]`, or the vector of length `label[2]`, stored in `cache` under `label`,
allocated (uninitialised) on first use.
"""
function getcache(cache::GPCache, label::Tuple{Symbol, Tuple{Int, Int}})
    return get!(() -> Matrix{Float64}(undef, label[2]), cache.cache_matrices, label)
end

function getcache(cache::GPCache, label::Tuple{Symbol, Int})
    return get!(() -> Vector{Float64}(undef, label[2]), cache.cache_vectors, label)
end

"""
    mul_A_B!(cache::GPCache, A, B, sizeA1, sizeB2)
    mul_A_B!(cache::GPCache, A, B, size1)

`A * B` written into a buffer of `cache` and returned: of size `sizeA1 × sizeB2`, or, for square
`A` and `B`, `size1 × size1`. The buffer is shared with later calls of the same size (see
[`GPCache`](@ref)).
"""
function mul_A_B!(cache::GPCache, A::AbstractArray, B, sizeA1::Int, sizeB2::Int)
    AB = getcache(cache, (:ABdiff, (sizeA1,sizeB2)))
    return mul!(AB, A, B)
end

function mul_A_B!(cache::GPCache, A::Array, B, size1::Int)
    #multiply 2 matrices with the same size
    AB = getcache(cache, (:AB, (size1, size1)))
    return mul!(AB, A, B)
end

"""
    mul_A_B_A!(cache::GPCache, A, B, size1)

`A * B * A` for square `A` and `B` of size `size1`, written into a buffer of `cache` (see
[`GPCache`](@ref)).
"""
function mul_A_B_A!(cache::GPCache, A::Matrix, B::Matrix, size1::Int)
    #A, B are square matrices with a same size
    AB = getcache(cache, (:AB, (size1, size1)))
    ABA = getcache(cache, (:ABA, (size1, size1)))
    mul!(AB, A, B)

    return mul!(ABA, AB, A)
end

"""
    mul_A_B_At!(cache::GPCache, A, B, sizeA1, sizeB1)

`A * B * A'` for `A` of size `sizeA1 × sizeB1` and square `B`, written into a buffer of `cache`
(see [`GPCache`](@ref)).
"""
function mul_A_B_At!(cache::GPCache, A::Matrix, B::Matrix, sizeA1::Int, sizeB1::Int)
    # A: matrix with size (sizeA1,sizeB1)
    # B: matrix with size (sizeB1,sizeB1)
    AB = getcache(cache, (:AB, (sizeA1, sizeB1)))
    ABAt = getcache(cache, (:ABA, (sizeA1, sizeA1)))
    mul!(AB, A, B)
    return mul!(ABAt, AB, A')
end
"""
    mul_A_v!(cache::GPCache, A, v, sizeA1)

`A * v` written into a buffer of `cache` of length `sizeA1` (see [`GPCache`](@ref)).
"""
function mul_A_v!(cache::GPCache, A::Matrix, v::Vector, sizeA1::Int)
    Av = getcache(cache, (:Av, sizeA1))
    return mul!(Av, A, v)
end