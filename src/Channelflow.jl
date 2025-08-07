module Channelflow

include("Coefficients.jl")
include("BandedTridiag.jl")
include("HelmholtzSolver.jl")

using .ChebyCoeffs
using .BandedTridiags
using .HelmholtzSolver

end
