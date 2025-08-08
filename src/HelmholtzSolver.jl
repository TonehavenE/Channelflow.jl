"""
Implements a solver for 1d Helmholtz Equation of the form

```math
nu u'' - lambda u = f on [a, b]
```

with Dirichlet or Neumann boundary conditions at `a` and `b`.
"""
module HelmholtzSolver

using ..ChebyCoeffs
using ..BandedTridiags
using Printf

export HelmholtzProblem, solve!, test_helmholtz, solve

struct CoeffVariables
    lambda::Real
    nu::Real
    c::Function
    beta::Function
end

# Coefficient functions from CHQZ1 (5.1.24) on page 130
function left_lower_coeff(n::Int, v::CoeffVariables)
    -(v.c(n - 2) * v.lambda) / (4 * n * (n - 1))
end

function left_diag_coeff(n::Int, v::CoeffVariables)
    v.nu + (v.beta(n) * v.lambda) / (2 * (n^2 - 1))
end

function left_upper_coeff(n::Int, v::CoeffVariables)
    -(v.beta(n + 2) * v.lambda) / (4 * n * (n + 1))
end

function left_coeffs(n::Int, v::CoeffVariables)
    (left_lower_coeff(n, v), left_diag_coeff(n, v), left_upper_coeff(n, v))
end

function right_lower_coeff(n::Int, v::CoeffVariables)
    v.c(n - 2) / (4 * n * (n - 1))
end

function right_diag_coeff(n::Int, v::CoeffVariables)
    -v.beta(n) / (2 * (n^2 - 1))
end

function right_upper_coeff(n::Int, v::CoeffVariables)
    v.beta(n + 2) / (4 * n * (n + 1))
end

function right_coeffs(n::Int, v::CoeffVariables)
    (right_lower_coeff(n, v), right_diag_coeff(n, v), right_upper_coeff(n, v))
end

function build_left_tridiag(v::CoeffVariables, numModes::Int, parity::Parity)::BandedTridiag
    A = BandedTridiag(numModes)

    # Set first row to all ones
    for i in 1:numModes
        A[1, i] = 1.0
    end

    # Fill tridiagonal rows
    for i in 2:numModes
        n = 2 * (i - 1)
        if parity == Odd
            n += 1
        end

        lower_coeff, diag_coeff, upper_coeff = left_coeffs(n, v)

        set_lodiag!(A, i, lower_coeff)
        set_diag!(A, i, diag_coeff)

        if upper_coeff != 0.0
            set_updiag!(A, i, upper_coeff)
        end
    end

    return A
end

function build_right_tridiag(v::CoeffVariables, numModes::Int, parity::Parity)::BandedTridiag
    B = BandedTridiag(numModes)

    # First row
    B[1, 1] = 1.0

    # Fill tridiagonal rows
    for i in 2:numModes
        n = 2 * (i - 1)
        if parity == Odd
            n += 1
        end

        lower_coeff, diag_coeff, upper_coeff = right_coeffs(n, v)

        set_lodiag!(B, i, lower_coeff)
        set_diag!(B, i, diag_coeff)
        if upper_coeff != 0.0
            set_updiag!(B, i, upper_coeff)
        end
    end

    return B
end

struct HelmholtzProblem
    number_modes::Int
    a::Real
    b::Real
    lambda::Real
    nu::Real

    # Internal data
    N::Int
    n_even_modes::Int
    n_odd_modes::Int
    c::Function
    β::Function
    A_even::BandedTridiag
    A_odd::BandedTridiag
    B_even::BandedTridiag
    B_odd::BandedTridiag

    function HelmholtzProblem(
        number_modes::Int,
        a::Real,
        b::Real,
        lambda::Real,
        nu::Real=1.0,
    )
        @assert number_modes % 2 == 1 "Number of modes must be odd"
        @assert number_modes > 2 "Must have at least three modes"

        N = number_modes - 1
        n_even_modes = div(N, 2) + 1
        n_odd_modes = div(N, 2)

        # Coefficient functions
        c(n) = (n == 0 || n == N) ? 2 : 1
        β(n) = (n > N - 2) ? 0 : 1

        # Scale nu by domain size
        nuscaled = nu / (((b - a) / 2)^2)
        v = CoeffVariables(lambda, nuscaled, c, β)

        # Build matrices
        Ae = build_left_tridiag(v, n_even_modes, Even)
        Ao = build_left_tridiag(v, n_odd_modes, Odd)
        Be = build_right_tridiag(v, n_even_modes, Even)
        Bo = build_right_tridiag(v, n_odd_modes, Odd)

        # Perform UL decomposition on left-hand-side matrices
        UL_decompose!(Ae)
        UL_decompose!(Ao)

        new(number_modes, a, b, lambda, nu, N, n_even_modes, n_odd_modes, c, β, Ae, Ao, Be, Bo)
    end
end

"""
Solve the Helmholtz equation with Dirichlet boundary conditions.

# Arguments
- `h::HelmholtzProblem`: The problem setup
- `u::ChebyCoeff`: Output solution coefficients (modified in place)
- `f::ChebyCoeff`: Right-hand side forcing function (spectral coefficients)
- `ua::Real`: Boundary value at left endpoint
- `ub::Real`: Boundary value at right endpoint
"""
function solve!(h::HelmholtzProblem, u::ChebyCoeff, f::ChebyCoeff, ua::Real, ub::Real)
    @assert f.state == Spectral "RHS must be in spectral form"
    # Apply right-hand-side operators B_even and B_odd to f, storing results in u
    # This computes the transformed RHS for the separated even/odd system
    multiply_strided!(f, h.B_even, u, 0, 2)  # Even modes: offset=0, stride=2
    multiply_strided!(f, h.B_odd, u, 1, 2)   # Odd modes: offset=1, stride=2

    # Set boundary condition coefficients
    # u[1] corresponds to (ub + ua)/2, u[2] to (ub - ua)/2
    u[1] = (ub + ua) / 2
    u[2] = (ub - ua) / 2

    # Solve the separated systems
    UL_solve_strided!(h.A_even, u, 0, 2)  # Even modes
    UL_solve_strided!(h.A_odd, u, 1, 2)   # Odd modes

    setState!(u, Spectral)

    return u
end

function solve(h::HelmholtzProblem, f::ChebyCoeff, ua::Real, ub::Real)
    u = ChebyCoeff(h.number_modes, h.a, h.b, Physical)
    solve!(h, u, f, ua, ub)
    return u
end


"""
Extended solver that handles both forcing and mean constraint.

Let u = uf + um.
Then solve:
1. nu uf'' - lambda uf = f  with uf(±1) = ua, ub
2. nu um'' - lambda um = mu with um(±1) = 0, 0

Where mu is chosen so that the total solution has the desired mean value.
"""
function solve!(h::HelmholtzProblem, u::ChebyCoeff, f::ChebyCoeff, umean::Real, ua::Real, ub::Real)
    @assert f.state == Spectral "RHS must be in spectral form"

    N = h.number_modes
    u_temp = ChebyCoeff(N, h.a, h.b, Spectral)
    rhs_temp = ChebyCoeff(f)  # Copy f

    # Step 1: Solve with given forcing and boundary conditions
    solve!(h, u_temp, rhs_temp, ua, ub)
    uamean = mean_value(u_temp)

    # Step 2: Solve homogeneous BCs with constant forcing to find response
    setToZero!(rhs_temp)
    rhs_temp[1] = h.nu  # Constant forcing
    solve!(h, u_temp, rhs_temp, 0.0, 0.0)
    ucmean = mean_value(u_temp)

    # Step 3: Compute correction factor and solve final system
    mu = h.nu * (umean - uamean) / ucmean
    rhs_temp = ChebyCoeff(f)  # Reset to original f
    rhs_temp[1] += mu         # Add mean correction
    solve!(h, u, rhs_temp, ua, ub)

    return u
end

function solve(h::HelmholtzProblem, f::ChebyCoeff, umean::Real, ua::Real, ub::Real)
    u = ChebyCoeff(h.number_modes, h.a, h.b, Physical)
    solve!(h, u, f, umean, ua, ub)
    return u
end

"""
Test the Helmholtz solver with a known analytical solution.
"""
function test_helmholtz()
    println("Testing Helmholtz solver...")

    # Problem setup
    N = 63
    lambda = 1.0
    nu = 1.0
    a = 0.0
    b = 2 * pi
    ua = 0.0
    ub = 0.0

    # Analytical solution: u(x) = sin(3x), so u''(x) = -9*sin(3x)
    # For nu*u'' - lambda*u = f, we have f = -9*sin(3x) - sin(3x) = -10*sin(3x)
    f_func(x) = -10 * sin(3 * x)
    u_exact(x) = sin(3 * x)  # This would give -9*sin(3x) - sin(3x) = -10*sin(3x) ✓

    # Generate forcing function
    x_points = chebypoints(N, a, b)
    f_data = [f_func(x) for x in x_points]

    println("Setting up RHS function...")
    rhs = ChebyCoeff(f_data, a, b, Physical)
    makeSpectral!(rhs)

    # Initialize solution
    u = ChebyCoeff(N, a, b, Spectral)

    # Create Helmholtz problem and solve
    println("Creating Helmholtz problem...")
    h = HelmholtzProblem(N, a, b, lambda, nu)

    println("Solving...")
    solve!(h, u, rhs, ua, ub)

    # Compare with analytical solution
    makePhysical!(u)
    u_analytical = [u_exact(x) for x in x_points]

    max_error = maximum(abs.(u.data - u_analytical))
    println("Maximum error: $(max_error)")

    if max_error < 1e-10
        println("✓ Test PASSED - Solution matches analytical result within tolerance")
    else
        println("✗ Test FAILED - Error too large")
        println("First few values:")
        for i = 1:min(10, length(u.data))
            @printf "  x=%.3f: computed=%.6f, exact=%.6f, error=%.2e\n" x_points[i] u.data[i] u_analytical[i] abs(u.data[i] - u_analytical[i])
        end
    end

    return max_error < 1e-10
end

end
