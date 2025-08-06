module Channeflow

include("Coefficients.jl")
include("HelmholtzSolver.jl")
include("BandedTridiag.jl")

using BandedTridiags
using ChebyCoeffs
using HelmholtzSolver

end
