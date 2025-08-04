module TimeSteppers

using ..HelmholtzSolver
using ..Collocation

export SimulationParameters, timestep_CNAB2

struct SimulationParameters
    viscosity::Float64
    # Add other parameters here as needed, e.g. density, forcing, etc.
end


"""
    timestep_CNAB2(u_prev, u_current, dt, grid, params, nonlinear_rhs)

Advance one timestep with CNAB2:
- u_prev: velocity at previous timestep (n-1)
- u_current: velocity at current timestep (n)
- dt: timestep size
- grid: spatial discretization object (for Helmholtz solver etc.)
- params: parameters (e.g. viscosity)
- nonlinear_rhs: function (u, grid, params) -> nonlinear term vector
Returns: u_next (velocity at timestep n+1)
"""
function timestep_CNAB2(
    u_current::AbstractVector{Float64},
    dt::Real,
    grid::AbstractGrid,
    params::SimulationParameters,
    nonlinear_rhs::Function;
    u_prev::AbstractVector{Float64},
)::AbstractVector{Float64}
    ν = params.viscosity

    # Compute nonlinear terms at previous two steps
    N_current = nonlinear_rhs(u_current, grid, params)
    N_prev = nonlinear_rhs(u_prev, grid, params)

    # Left hand side operator: (I - ν dt/2 Δ)
    α = 1.0
    β = ν * dt / 2

    # Right hand side vector:
    rhs = (α + β) .* u_current .- dt .* (1.5 .* N_current .- 0.5 .* N_prev)

    # Solve Helmholtz: (α I - β Δ) u_next = rhs
    u_next = solve_helmholtz(grid, α, β, rhs)

    return u_next
end


"""
    timestep_RK3(u_current, dt, grid, params, nonlinear_rhs)

Advance one timestep with explicit RK3:
- u_current: velocity at current timestep (n)
- dt: timestep size
- grid: spatial discretization object
- params: parameters
- nonlinear_rhs: function (u, grid, params) -> nonlinear term vector
Returns: u_next (velocity at timestep n+1)
"""
function timestep_RK3(
    u_current::AbstractVector{Float64},
    dt::Real,
    grid::AbstractGrid,
    params::SimulationParameters,
    nonlinear_rhs::Function
)::AbstractVector{Float64}
    # Implementation as before...
end

end
