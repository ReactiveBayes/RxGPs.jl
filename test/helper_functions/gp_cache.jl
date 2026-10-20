using RxGP
using Random
using LinearAlgebra
using Test
using TestItemRunner

@testitem "helper_functions/gp_cache/TestGPCache" begin
    using StableRNGs
    rng = StableRNG(100)
    using RxGP, Random, LinearAlgebra, Test
    gpcache = GPCache()
    A = rand(rng, 4,4)
    B = rand(rng, 3,4)
    D = rand(rng, 4,4)
    a = rand(rng, 4)
    b = rand(rng, 4)

    # test gpcache
    @test typeof(gpcache.cache_matrices) <: Dict            # get matrix
    @test typeof(gpcache.cache_vectors) <: Dict             # get vector

    @test typeof(getcache(gpcache,(:A, (3,3)))) <: Matrix
    @test typeof(getcache(gpcache,(:a,3))) <: Vector

    # test in-place multiplication
    @test mul_A_B!(gpcache, B, A, size(B,1), size(A,2)) == B * A    # A and B have different size
    @test mul_A_B!(gpcache, A,D,size(A,1)) == A * D                 # A and D have the same size
    @test mul_A_B_A!(gpcache, A, D, size(A,1)) == A * D * A
    @test mul_A_B_At!(gpcache,B,A,size(B,1), size(A,1)) == B * A * B'   # A and B have different size
    @test mul_A_B_At!(gpcache,A,D,size(A,1), size(D,1)) == A * D * A'   # A and D have the same size
    @test mul_A_v!(gpcache,A,a,size(A,1)) == A * a
    @test mul_A_v!(gpcache,A,b,size(A,1)) == A * b
end