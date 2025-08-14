using Channelflow
using Test

@testset "ChannelFlow.jl" begin
    include("helmholtz_tests.jl")
    include("banded_tridiag_tests.jl")
    include("chebyshev_tests.jl")
    include("flowfield_tests.jl")
    include("flowfielddomain_tests.jl")
    include("tensor_flowfield_tests.jl")
    include("metrics/metric_tests.jl")
    include("tausolver_tests.jl")
end
