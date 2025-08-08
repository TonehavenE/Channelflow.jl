using Channelflow
using Test

@testset "ChannelFlow.jl" begin
    include("helmholtz_tests.jl")
    include("banded_tridiag_tests.jl")
    include("chebyshev_tests.jl")
end
