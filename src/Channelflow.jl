module Channelflow

using Reexport

include("ChebyCoeffs.jl")
include("BandedTridiag.jl")
include("HelmholtzSolver.jl")
include("FlowField.jl")
include("Metrics.jl")
include("BasisFuncs.jl")
include("TauSolvers.jl")

@reexport using .ChebyCoeffs
@reexport using .BandedTridiags
@reexport using .HelmholtzSolver
@reexport using .FlowFields
@reexport using .Metrics
@reexport using .BasisFuncs
@reexport using .TauSolvers

end
