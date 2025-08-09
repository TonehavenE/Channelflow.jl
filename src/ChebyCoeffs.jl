module ChebyCoeffs

#=
This module provides a set of functions and types for working with Chebyshev coefficients.
=#

include("ChebyCoeffs/types_and_constructors.jl")
include("ChebyCoeffs/base_ops.jl")
include("ChebyCoeffs/transforms.jl")
include("ChebyCoeffs/evaluation.jl")
include("ChebyCoeffs/calculus.jl")
include("ChebyCoeffs/utilities.jl")
include("ChebyCoeffs/norms.jl")

end
