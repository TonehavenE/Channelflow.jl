using Test
using Channelflow

@testset "FlowField" begin

    # ===========================
    # FlowField Constructor Tests  
    # ===========================
    @testset "FlowField Constructors" begin
        @testset "Basic Constructor" begin
            ff = FlowField(8, 5, 16, 3, 2π, 1π, -1.0, 1.0)

            @test ff.domain.Nx == 8
            @test ff.domain.Ny == 5
            @test ff.domain.Nz == 16
            @test ff.domain.num_dimensions == 3
            @test ff.xz_state == Physical
            @test ff.y_state == Physical
            @test !ff.padded
            @test ff.physical_data !== nothing
            @test ff.spectral_data === nothing
            @test size(ff.physical_data) == (8, 5, 16, 3)
        end

        @testset "Constructor with Spectral State" begin
            ff = FlowField(8, 5, 16, 3, 2π, 1π, -1.0, 1.0; xz_state=Spectral)

            @test ff.xz_state == Spectral
            @test ff.y_state == Physical
            @test ff.physical_data === nothing
            @test ff.spectral_data !== nothing
            @test size(ff.spectral_data) == (8, 5, 9, 3)  # Mz = 16/2 + 1 = 9
        end

        @testset "Domain Constructor" begin
            domain = FlowFieldDomain(8, 5, 16, 3, 2π, 1π, -1.0, 1.0)
            ff = FlowField(domain)

            @test ff.domain == domain
            @test ff.physical_data !== nothing
            @test size(ff.physical_data) == (8, 5, 16, 3)
        end

        @testset "Copy Constructor" begin
            ff1 = FlowField(8, 5, 16, 3, 2π, 1π, -1.0, 1.0)
            ff1[1, 1, 1, 1] = 42.0

            ff2 = FlowField(ff1)

            @test ff2.domain == ff1.domain
            @test ff2.xz_state == ff1.xz_state
            @test ff2.y_state == ff1.y_state
            @test ff2[1, 1, 1, 1] == 42.0

            # Test deep copy
            ff1[1, 1, 1, 1] = 99.0
            @test ff2[1, 1, 1, 1] == 42.0
        end
    end

    # ===========================
    # FlowField Accessor Tests
    # ===========================

    @testset "FlowField Accessors" begin
        ff = FlowField(8, 5, 16, 3, 2π, 1π, -1.0, 1.0)

        @test num_x_gridpoints(ff) == 8
        @test num_y_gridpoints(ff) == 5
        @test num_z_gridpoints(ff) == 16
        @test num_gridpoints(ff) == (8, 5, 16)

        @test num_x_modes(ff) == 8
        @test num_y_modes(ff) == 5
        @test num_z_modes(ff) == 9
        @test num_modes(ff) == (8, 5, 9)

        @test vector_dim(ff) == 3
        @test xz_state(ff) == Physical
        @test y_state(ff) == Physical
        @test Lx(ff) == 2π
        @test Ly(ff) == 2.0
        @test Lz(ff) ≈ π
        @test domain_a(ff) == -1.0
        @test domain_b(ff) == 1.0

        # Test coordinate accessors
        @test x(ff, 1) ≈ 0.0
        @test y(ff, 1) ≈ 1.0
        @test z(ff, 1) ≈ 0.0

        @test length(x_gridpoints(ff)) == 8
        @test length(y_gridpoints(ff)) == 5
        @test length(z_gridpoints(ff)) == 16
    end

    # ===========================
    # Element Access Tests
    # ===========================

    @testset "Element Access" begin
        @testset "Physical Access" begin
            ff = FlowField(4, 3, 8, 2, 2π, 1π, -1.0, 1.0)

            # Test setindex! and getindex
            ff[1, 1, 1, 1] = 42.0
            ff[4, 3, 8, 2] = -2.7

            @test ff[1, 1, 1, 1] == 42.0  # Data should be preserved
            @test ff[4, 3, 8, 2] == -2.7
        end
    end

    # ===========================
    # Congruence Tests
    # ===========================

    @testset "Congruence Tests" begin
        ff1 = FlowField(8, 5, 16, 3, 2π, 1π, -1.0, 1.0)
        ff2 = FlowField(8, 5, 16, 3, 2π, 1π, -1.0, 1.0)
        ff3 = FlowField(8, 5, 16, 2, 2π, 1π, -1.0, 1.0)  # Different dimensions
        ff4 = FlowField(8, 5, 16, 3, 4π, 1π, -1.0, 1.0)  # Different Lx

        @test congruent(ff1, ff2)
        @test geom_congruent(ff1, ff3)
        @test !congruent(ff1, ff3)
        @test !geom_congruent(ff1, ff4)
    end
    # ===========================
    # Integration Tests
    # ===========================

    @testset "Integration Tests" begin
        @testset "Chebyshev Polynomial Representation" begin
            # Test that Chebyshev polynomials are represented correctly
            ff = FlowField(1, 9, 1, 1, 2π, 1π, -1.0, 1.0)  # Only y variation

            # Set to T_4(y) = 8y^4 - 8y^2 + 1 (4th Chebyshev polynomial)
            for ny in 1:9
                y_val = y_coord(ff.domain, ny)
                ff[1, ny, 1, 1] = 8 * y_val^4 - 8 * y_val^2 + 1
            end

            # Transform to spectral in y
            make_spectral_y!(ff)

            # In spectral space, only the 4th mode should be non-zero
            for ny in 1:9
                val = ff.physical_data[1, ny, 1, 1]  # Still stored in physical_data after y transform
                if ny == 5  # 5th coefficient corresponds to T_4
                    @test abs(val) > 1e-10
                else
                    @test abs(val) < 1e-12
                end
            end
        end

        @testset "Mixed State Operations" begin
            # Test operations between fields in different states
            ff1 = FlowField(8, 5, 16, 2, 2π, 1π, -1.0, 1.0)
            ff2 = FlowField(8, 5, 16, 2, 2π, 1π, -1.0, 1.0)

            # Set same data in both
            for i in 1:2, nz in 1:16, ny in 1:5, nx in 1:8
                val = sin(π * (nx - 1) / 8) * cos(π * (nz - 1) / 16) * (ny + i)
                ff1[nx, ny, nz, i] = val
                ff2[nx, ny, nz, i] = val
            end

            # Transform one to spectral
            make_spectral_xz!(ff2)

            # Transform back and compare
            make_physical_xz!(ff2)

            @test maximum(abs.(ff1.physical_data .- ff2.physical_data)) < 1e-12
        end
    end



    # ===========================
    # Edge Case Tests
    # ===========================

    @testset "Edge Cases" begin
        @testset "Minimal Grid Sizes" begin
            # Test smallest possible grids
            ff = FlowField(2, 2, 2, 1, 1π, 1π / 2, -1.0, 1.0)

            @test ff.domain.Nx == 2
            @test ff.domain.Ny == 2
            @test ff.domain.Nz == 2
            @test ff.domain.Mz == 2  # 2/2 + 1 = 2

            ff[1, 1, 1, 1] = 1.0
            ff[2, 2, 2, 1] = -1.0

            # Transforms should work
            make_spectral_xz!(ff)
            @test ff.xz_state == Spectral

            make_physical_xz!(ff)
            @test ff[1, 1, 1, 1] ≈ 1.0
            @test ff[2, 2, 2, 1] ≈ -1.0
        end

        @testset "Single Dimension Fields" begin
            # Test 1D-like fields
            ff_1d_x = FlowField(16, 1, 1, 1, 2π, 1π, 0.0, 1.0)
            ff_1d_y = FlowField(1, 9, 1, 1, 2π, 1π, -1.0, 1.0)
            ff_1d_z = FlowField(1, 1, 16, 1, 2π, 1π, 0.0, 1.0)

            @test size(ff_1d_x.physical_data) == (16, 1, 1, 1)
            @test size(ff_1d_y.physical_data) == (1, 9, 1, 1)
            @test size(ff_1d_z.physical_data) == (1, 1, 16, 1)

            # Basic operations should work
            ff_1d_x[8, 1, 1, 1] = 5.0
            @test ff_1d_x[8, 1, 1, 1] == 5.0

            make_spectral_xz!(ff_1d_x)
            make_physical_xz!(ff_1d_x)
            @test ff_1d_x[8, 1, 1, 1] ≈ 5.0
        end

        @testset "Zero Field Operations" begin
            ff1 = FlowField(8, 5, 16, 2, 2π, 1π, -1.0, 1.0)
            ff2 = FlowField(8, 5, 16, 2, 2π, 1π, -1.0, 1.0)

            # Both start as zero fields
            @test all(ff1.physical_data .== 0)
            @test all(ff2.physical_data .== 0)

            # Operations with zeros
            ff3 = ff1 + ff2
            @test all(ff3.physical_data .== 0)

            ff4 = ff1 * 5.0
            @test all(ff4.physical_data .== 0)

            # Transform zero field
            make_spectral_xz!(ff1)
            @test all(ff1.spectral_data .== 0)

            make_physical_xz!(ff1)
            @test all(ff1.physical_data .== 0)
        end
    end

end