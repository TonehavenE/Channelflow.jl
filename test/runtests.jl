using Channelflow
using Test

@testset "ChannelFlow.jl" begin
    # Write your tests here.
    include("chebyshev_tests.jl")
    include("fourier_tests.jl")
    include("differentiation_operator_tests.jl")
    include("helmholtz_tests.jl")
end
