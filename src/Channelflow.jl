module Channelflow

include("Coefficients.jl")
include("BandedTridiag.jl")
include("HelmholtzSolver.jl")
include("FlowField/FlowField.jl")

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
export set_main_diag!, set_first_row!, set_upper_diag!, set_lower_diag!, first_row, upper_diag, lower_diag, main_diag

using .HelmholtzSolver
export HelmholtzProblem, solve!, test_helmholtz, solve

using .FlowFields

# FlowFieldDomain exports
export FlowFieldDomain, x_coord, y_coord, z_coord,
    x_gridpoints, y_gridpoints, z_gridpoints,
    kx_range, kz_range, kx_to_mx, mx_to_kx,
    kz_to_mz, mz_to_kz, kx_max_dealiased,
    kz_max_dealiased, is_aliased,
    geom_congruent, congruent

export FlowField, FlowFieldDomain, FlowFieldTransforms
export _current_data, _ensure_data_allocated!
export cmplx, set_cmplx!
export num_x_gridpoints, num_y_gridpoints, num_z_gridpoints, num_gridpoints
export num_x_modes, num_y_modes, num_z_modes, num_modes
export vector_dim, xz_state, y_state
export Lx, Ly, Lz, domain_a, domain_b
export x, y, z, x_gridpoints, y_gridpoints, z_gridpoints
export kx_to_mx, mx_to_kx, kz_to_mz, mz_to_kz
export kx_max_dealiased, kz_max_dealiased, is_aliased
export geom_congruent, congruent
export make_physical!, make_spectral!, make_state!, make_physical_xz!, make_spectral_xz!, make_physical_y!, make_spectral_y!
export scale!, add!, subtract!, add!, set_to_zero!
export swap!, zero_padded_modes!
export resize!, rescale!

end
