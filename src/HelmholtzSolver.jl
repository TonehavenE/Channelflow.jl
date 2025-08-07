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

export HelmholtzProblem, solve!, test_helmholtz

struct CoeffVariables
    lambda::Real
    nu::Real
    c::Function
    beta::Function
end


# see CHQZ1 (5.1.24) on page 130
function left_lower_coeff(n::Int, v::CoeffVariables)
    -(v.c(n - 2) * v.lambda) / (4 * n * (n - 1))
end

function left_diag_coeff(n::Int, v::CoeffVariables)
    (v.nu + (v.beta(n) * v.lambda) / (2 * (n^2 - 1)))
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
    for i in 1:numModes
        A[1, i] = 1.0
    end

    for i in 2:numModes-1
        n = 2 * (i - 1)
        if parity == Odd
            n += 1
        end

        lower_coeff, diag_coeff, upper_coeff = left_coeffs(n, v)
        A.lower[i] = lower_coeff
        A.diag[i] = diag_coeff
        if upper_coeff != 0.0
            A.upper[i] = upper_coeff
        end
    end
    A
end


function build_right_tridiag(
    v::CoeffVariables,
    numModes::Int,
    parity::Parity,
)::BandedTridiag
    B = BandedTridiag(numModes)

    # Assign first row
    B.diag[1] = 1.0

    # Assign other rows
    for i in 2:numModes-1
        n = 2 * (i - 1)
        if parity == Odd
            n += 1
        end

        lower_coeff, diag_coeff, upper_coeff = right_coeffs(n, v)
        B.lower[i] = lower_coeff
        B.diag[i] = diag_coeff
        if upper_coeff != 0.0
            B.upper[i] = upper_coeff
        end
    end

    B
end
struct HelmholtzProblem
    number_modes::Int
    a::Real
    b::Real
    lambda::Real
    nu::Real

    # "private"
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
        @assert number_modes % 2 == 1 "must be odd"
        @assert number_modes > 2 "must have at least three modes"

        N = number_modes - 1
        n_even_modes = div(N, 2) + 1
        n_odd_modes = div(N, 2)
        c(n) = (n == 0 || n == N) ? 2 : 1
        β(n) = (n > N - 2) ? 0 : 1
        nuscaled = nu / (((b - a) / 2)^2)
        v = CoeffVariables(lambda, nuscaled, c, β)
        Ae = build_left_tridiag(v, n_even_modes, Even)
        Ao = build_left_tridiag(v, n_odd_modes, Odd)
        Be = build_right_tridiag(v, n_even_modes, Even)
        Bo = build_right_tridiag(v, n_odd_modes, Odd)
        UL_decompose!(Ae)
        UL_decompose!(Ao)

        new(number_modes, a, b, lambda, nu, N, n_even_modes, n_odd_modes, c, β, Ae, Ao, Be, Bo)
    end
end

function solve!(h::HelmholtzProblem, u::ChebyCoeff, f::ChebyCoeff, ua::Real, ub::Real)
    @assert f.state == Spectral

    multiply_strided!(f, h.B_even, u, 0, 2)
    println("after first multiply_strided $(u.data)")
    multiply_strided!(f, h.B_odd, u, 1, 2)
    println("after second multiply_strided $(u.data)")


    u[1] = (ub + ua) / 2
    u[2] = (ub - ua) / 2

    UL_solve_strided!(h.A_even, u, 0, 2)
    println("after first solve_strided $(u.data)")
    UL_solve_strided!(h.A_odd, u, 1, 2)
    println("after second solve_strided $(u.data)")

    setState!(u, Spectral)
end

"""
Let u = uf + um.
Then solve:
1. nu uf'' - lambda uf = f  with uf(±1) = ua, ub
2. nu um'' - lambda um = mu with um(±1) = 0, 0
"""
function solve!(h::HelmholtzProblem, u::ChebyCoeff, f::ChebyCoeff, umean::Real, ua::Real, ub::Real)
    @assert f.state == Spectral "must be spectral RHS"

    N = h.number_modes
    u_temp = ChebyCoeff(N, h.a, h.b, Spectral)
    rhs_temp = ChebyCoeff(f)

    solve!(h, u_temp, rhs_temp, ua, ub)

    uamean = mean_value(u_temp)

    setToZero!(rhs_temp)
    rhs_temp[1] = h.nu
    solve!(h, u_temp, rhs_temp, 0.0, 0.0)

    ucmean = mean_value(u_temp)

    mu = h.nu * (umean - uamean) / ucmean
    rhs_temp = f
    rhs_temp[1] += mu
    solve!(h, u, rhs_temp, ua, ub)
end

function verify(u::ChebyCoeff, f::ChebyCoeff, umean::Real, ua::Real, ub::Real) end

function test_helmholtz()
    N = 63
    lambda = 1
    nu = 1
    a = 0
    b = pi
    ua = 0
    ub = 0
    # analytic solution
    f(x) = -8 * sin(3 * x)

    # need to get discrete points to generate f
    data = [f(x) for x in LinRange(a, b, N)]
    println("Start: data is $data\n\n")
    rhs = ChebyCoeff(data, a, b, Physical)
    makeSpectral!(rhs)
    println("\n\nrhs is: $rhs\n\n")

    u = ChebyCoeff(N, a, b, Spectral)

    println(u.data)

    h = HelmholtzProblem(N, a, b, lambda, nu)

    solve!(h, u, rhs, ua, ub)

    println(u.data)
    # println(data)
    # @assert isapprox(u.data, data; atol=1e-5)
end

end
