module Equations

using ..FlowFields
using ..DNSSettings
import ..HelmholtzSolver: solve!

#=
Generalized code for defining *equations* for DNS simulations.
=#

export Equation, nonlinear!, linear!, solve!, create_RHS, reset_lambda!

"""
    Equation

Defines an abstract type for equations used in DNS simulations.
"""
abstract type Equation end

"""
    nonlinear!(eqn, infields, outfields)
Calculates the nonlinear terms of the equation.
"""
function nonlinear!(eqn::Equation, infields::Vector{FlowField}, outfields::Vector{FlowField})
    error("Nonlinear function not implemented for $(typeof(eqn))")
end

"""
    linear!(eqn, infields, outfields)
Calculates the linear terms of the equation.
"""
function linear!(eqn::Equation, infields::Vector{FlowField}, outfields::Vector{FlowField})
    error("Linear function not implemented for $(typeof(eqn))")
end

"""
    solve!(eqn, infields, outfields, i)

Solves the equation for a specific index.
"""
function solve!(eqn::Equation, infields::Vector{FlowField}, outfields::Vector{FlowField}, i::Int)
    error("Solve function not implemented for $(typeof(eqn)) at index $i")
end

"""
    create_RHS(eqn, fields)

Creates a vector of right-hand side terms for the equations.
"""
function create_RHS(eqn::Equation, fields::Vector{FlowField})
    error("create_RHS function not implemented for $(typeof(eqn))")
end

function reset_lambda!(eqn::Equation, lambda_t::Vector{Real}, flags::DNSFlags)
    error("reset_lambda function not implemented for $(typeof(eqn))")
end

include("navier_stokes.jl")

end