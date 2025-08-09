module Channelflow

using Reexport

include("Coefficients.jl")
include("BandedTridiag.jl")
include("HelmholtzSolver.jl")
include("FlowField.jl")

@reexport using .ChebyCoeffs
@reexport using .BandedTridiags
@reexport using .HelmholtzSolver
@reexport using .FlowFields

end
