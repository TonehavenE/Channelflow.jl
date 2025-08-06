module Channelflow

# Write your package code here.
include("domains/Collocation.jl")
include("DifferentialOperators.jl")
include("HelmholtzSolver.jl")
include("TimeSteppers.jl")
include("FlowFields.jl")

using .Collocation
using .DifferentialOperators
using .HelmholtzSolver
using .TimeSteppers
using .FlowFields

export ChebyshevGrid, FourierGrid, fourier_derivative_matrix, differentiation_operator, differentiation_operators, solve_helmholtz, timestep_CNAB2, SimulationParameters, get_derivative_matrix, FlowField, nonlinear_rhs

end
