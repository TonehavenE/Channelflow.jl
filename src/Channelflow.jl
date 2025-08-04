module Channelflow

# Write your package code here.
include("domains/Collocation.jl")
include("DifferentialOperators.jl")
include("HelmholtzSolver.jl")
include("TimeSteppers.jl")

using .Collocation
using .DifferentialOperators
using .HelmholtzSolver
using .TimeSteppers

export ChebyshevGrid, FourierGrid, fourier_derivative_matrix, differentiation_operator, differentiation_operators, solve_helmholtz, timestep_CNAB2, SimulationParameters

end
