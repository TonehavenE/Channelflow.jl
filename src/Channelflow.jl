module Channelflow

using Reexport

include("ChebyCoeffs.jl")
include("BandedTridiag.jl")
include("HelmholtzSolver.jl")
include("FlowField.jl")
include("Metrics.jl")

@reexport using .ChebyCoeffs
@reexport using .BandedTridiags
@reexport using .HelmholtzSolver
@reexport using .FlowFields
@reexport using .Metrics

end
