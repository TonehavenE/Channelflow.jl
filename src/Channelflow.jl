module Channelflow

using Reexport

include("ChebyCoeffs.jl")
include("BandedTridiag.jl")
include("HelmholtzSolver.jl")
include("FlowField.jl")
include("Metrics.jl")
include("BasisFuncs.jl")
include("TauSolvers.jl")
include("DNS/DNSSettings.jl")
include("Equations/Equations.jl")
include("DNS/algorithms/DNSAlgorithms.jl")

@reexport using .ChebyCoeffs
@reexport using .BandedTridiags
@reexport using .HelmholtzSolver
@reexport using .FlowFields
@reexport using .Metrics
@reexport using .BasisFuncs
@reexport using .TauSolvers
@reexport using .DNSSettings
@reexport using .Equations
@reexport using .DNSAlgorithms

end
