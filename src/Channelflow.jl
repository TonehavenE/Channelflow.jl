module Channelflow

include("Coefficients.jl")
include("BandedTridiag.jl")
include("HelmholtzSolver.jl")

using .ChebyCoeffs

export ChebyTransform, ChebyCoeff
export makeSpectral!, makePhysical!, makeState!, setToZero!, setState!
export chebyfft!, ichebyfft!
export L2Norm2, L2Norm, L2InnerProduct, LinfNorm, L1Norm, mean_value, evaluate
export evaluate, eval_a, eval_b, slope_a, slope_b
export chebyNorm2, chebyInnerProduct, chebyNorm
export chebypoints
export bounds, domain_length, num_modes, state
export integrate, derivative, derivative2
export legendre_polynomial, chebyshev_polynomial
export FieldState, Physical, Spectral, BC, Diri, Neumann, Parity, Even, Odd, NormType, Uniform, Cheby

using .BandedTridiags

export BandedTridiag, UL_decompose!, UL_solve!, UL_solve_strided!, multiply_strided!, multiply!, multiply, multiply_strided, to_dense, extract_UL_matrices
export set_diag!, set_band!, set_updiag!, set_lodiag!, band, updiag, lodiag, diag

using .HelmholtzSolver
export HelmholtzProblem, solve!, test_helmholtz, solve

end
