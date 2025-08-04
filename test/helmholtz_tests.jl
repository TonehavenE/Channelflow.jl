using Test
using Channelflow

@testset "Helmholtz Tests" begin
    # Parameters for tests
    N = 64
    L = 2π
    α = 1.0
    β = 1.0

    # --- FourierGrid: Solve (α I - β D2) u = f for sine wave modes ---
    grid_f = FourierGrid(N, L)
    x_f = grid_f.x
    k_f = grid_f.wave_numbers

    for mode_idx in 2:5
        k_wave = k_f[mode_idx]
        u_true = sin.(k_wave .* x_f)
        f = (α + β * k_wave^2) .* u_true
        u_sol = solve_helmholtz(grid_f, α, β, f)
        @test isapprox(u_sol, u_true; atol=1e-6)
    end

    # --- ChebyshevGrid: Solve with Dirichlet BCs on (α I - β D2) u = f ---
    grid_c = ChebyshevGrid(N, (-1.0, 1.0))
    x_c = grid_c.x

    # Test function: u(x) = 1 - x^2, which satisfies u(±1)=0 (Dirichlet BCs)
    u_true = 1 .- x_c .^ 2
    # Compute f = α u - β u''; second derivative u'' = -2
    f = α .* u_true .- β .* (-2 .* ones(N))

    # Define Dirichlet BCs at boundaries u(-1)=0, u(1)=0
    bc = Dict(
        :left => (:Dirichlet, 0.0),
        :right => (:Dirichlet, 0.0),
    )

    u_sol = solve_helmholtz(grid_c, α, β, f; bc=bc)
    @test isapprox(u_sol, u_true; atol=1e-5)

    # --- ChebyshevGrid: Solve with Neumann BCs (derivative at boundary) ---
    # u(x) = x^3, so u' = 3x^2; at x=±1, u' = 3
    u_true = x_c .^ 3
    u_prime_left = 3 * (-1)^2
    u_prime_right = 3 * (1)^2
    # u'' = 6x
    f = α .* u_true .- β .* (6 .* x_c)

    bc = Dict(
        :left => (:Neumann, u_prime_left),
        :right => (:Neumann, u_prime_right),
    )

    u_sol = solve_helmholtz(grid_c, α, β, f; bc=bc)
    @test isapprox(u_sol, u_true; atol=1e-4)  # Slightly looser tol due to BC enforcement

    # --- Error test: FourierGrid should error if BCs provided ---
    @test_throws ErrorException solve_helmholtz(grid_f, α, β, f; bc=bc)

    # --- Error test: Negative degree derivative or invalid operator parameters ---
    # You can add any other domain-specific edge cases here as needed
end
