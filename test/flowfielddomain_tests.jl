using Test
using Channelflow

# ===========================
# FlowFieldDomain Tests
# ===========================

@testset "FlowFieldDomain" begin
    @testset "Constructor and Basic Properties" begin
        domain = FlowFieldDomain(32, 17, 64, 3, 4.0, 2.0, -1.0, 1.0)

        @test domain.Nx == 32
        @test domain.Ny == 17
        @test domain.Nz == 64
        @test domain.num_dimensions == 3
        @test domain.Lx == 4.0
        @test domain.Lz == 2.0
        @test domain.a == -1.0
        @test domain.b == 1.0

        # Test derived quantities
        @test domain.Mx == 32
        @test domain.My == 17
        @test domain.Mz == 33  # Nz/2 + 1 = 64/2 + 1 = 33
    end

    @testset "Constructor Validation" begin
        @test_throws AssertionError FlowFieldDomain(0, 17, 64, 3, 4.0, 2.0, -1.0, 1.0)  # Nx <= 0
        @test_throws AssertionError FlowFieldDomain(32, 0, 64, 3, 4.0, 2.0, -1.0, 1.0)  # Ny <= 0
        @test_throws AssertionError FlowFieldDomain(32, 17, 0, 3, 4.0, 2.0, -1.0, 1.0)  # Nz <= 0
        @test_throws AssertionError FlowFieldDomain(32, 17, 64, 0, 4.0, 2.0, -1.0, 1.0) # num_dimensions <= 0
        @test_throws AssertionError FlowFieldDomain(32, 17, 64, 3, 0.0, 2.0, -1.0, 1.0) # Lx <= 0
        @test_throws AssertionError FlowFieldDomain(32, 17, 64, 3, 4.0, 0.0, -1.0, 1.0) # Lz <= 0
        @test_throws AssertionError FlowFieldDomain(32, 17, 64, 3, 4.0, 2.0, 1.0, -1.0) # a >= b
    end

    @testset "Coordinate Functions" begin
        domain = FlowFieldDomain(8, 5, 16, 3, 2π, Float64(π), -1.0, 1.0)

        # Test x coordinates (uniform grid)
        @test x_coord(domain, 1) ≈ 0.0
        @test x_coord(domain, 2) ≈ π / 4
        @test x_coord(domain, 8) ≈ 7π / 4

        # Test z coordinates (uniform grid)  
        @test z_coord(domain, 1) ≈ 0.0
        @test z_coord(domain, 2) ≈ π / 16
        @test z_coord(domain, 16) ≈ 15π / 16

        # Test y coordinates (Chebyshev points)
        @test y_coord(domain, 1) ≈ 1.0   # cos(0) = 1
        @test isapprox(y_coord(domain, 3), 0.0; atol = 1e-15)
        @test y_coord(domain, 5) ≈ -1.0  # cos(π) = -1

        # Test grid point functions
        x_pts = x_gridpoints(domain)
        y_pts = y_gridpoints(domain)
        z_pts = z_gridpoints(domain)

        @test length(x_pts) == 8
        @test length(y_pts) == 5
        @test length(z_pts) == 16
        @test x_pts[1] ≈ x_coord(domain, 1)
        @test y_pts[3] ≈ y_coord(domain, 3)
        @test z_pts[5] ≈ z_coord(domain, 5)
    end

    @testset "Wave Number Functions" begin
        domain = FlowFieldDomain(8, 5, 16, 3, 2π, Float64(π), -1.0, 1.0)

        # Test kx range: [-3, -2, -1, 0, 1, 2, 3, 4] for Nx=8
        @test kx_to_mx(domain, 0) == 1
        @test kx_to_mx(domain, 1) == 2
        @test kx_to_mx(domain, 4) == 5
        @test kx_to_mx(domain, -1) == 8
        @test kx_to_mx(domain, -3) == 6

        @test mx_to_kx(domain, 1) == 0
        @test mx_to_kx(domain, 2) == 1
        @test mx_to_kx(domain, 5) == 4
        @test mx_to_kx(domain, 8) == -1
        @test mx_to_kx(domain, 6) == -3

        # Test kz range: [0, 1, 2, ..., 8] for Mz=9
        @test kz_to_mz(domain, 0) == 1
        @test kz_to_mz(domain, 5) == 6
        @test kz_to_mz(domain, 8) == 9

        @test mz_to_kz(domain, 1) == 0
        @test mz_to_kz(domain, 6) == 5
        @test mz_to_kz(domain, 9) == 8
    end

    @testset "Dealiasing Functions" begin
        domain = FlowFieldDomain(12, 5, 18, 3, 2π, Float64(π), -1.0, 1.0)

        # 2/3 rule: kx_max = 12/3 - 1 = 3, kz_max = 18/3 - 1 = 5
        @test kx_max_dealiased(domain) == 3
        @test kz_max_dealiased(domain) == 5

        @test !is_aliased(domain, 0, 0)
        @test !is_aliased(domain, 3, 5)
        @test !is_aliased(domain, -3, 0)
        @test is_aliased(domain, 4, 0)
        @test is_aliased(domain, 0, 6)
        @test is_aliased(domain, -4, 3)
    end

    @testset "Domain Congruence" begin
        d1 = FlowFieldDomain(8, 5, 16, 3, 2π, 1.0 * π, -1.0, 1.0)
        d2 = FlowFieldDomain(8, 5, 16, 3, 2π, 1.0 * π, -1.0, 1.0)
        d3 = FlowFieldDomain(8, 5, 16, 2, 2π, 1.0 * π, -1.0, 1.0)  # Different num_dimensions
        d4 = FlowFieldDomain(8, 5, 16, 3, 4π, 1.0 * π, -1.0, 1.0)  # Different Lx

        @test d1 == d2
        @test congruent(d1, d2)
        @test geom_congruent(d1, d3)
        @test !congruent(d1, d3)
        @test !geom_congruent(d1, d4)
    end
end
