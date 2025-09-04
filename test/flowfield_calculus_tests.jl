using Test
using LinearAlgebra

# Test cases for curl! and cross! methods in FlowField

@testset "FlowField curl! and cross! Tests" begin

    # Test parameters
    Nx, Ny, Nz = 8, 9, 8
    Lx, Lz = 2π, 2π
    a, b = -1.0, 1.0

    @testset "curl! Tests" begin

        @testset "Constant field curl is zero" begin
            # Create a constant vector field
            f = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            curlf = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            # Set constant field: f = (1, 2, 3) everywhere
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                f[nx, ny, nz, 1] = 1.0
                f[nx, ny, nz, 2] = 2.0
                f[nx, ny, nz, 3] = 3.0
            end

            curl!(f, curlf)
            make_physical!(curlf)

            # Curl of constant field should be zero (within numerical precision)
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz, i = 1:3
                @test abs(curlf[nx, ny, nz, i]) < 1e-12
            end
        end

        @testset "Linear field test" begin
            # Test curl of field f = (0, 0, x) should give curl = (0, -1, 0) * 2π/Lx
            f = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            curlf = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                x = nx_to_x(f, nx)
                f[nx, ny, nz, 1] = 0.0
                f[nx, ny, nz, 2] = 0.0
                f[nx, ny, nz, 3] = x
            end

            curl!(f, curlf)
            make_physical!(curlf)

            expected_curl_y = -2π / Lx

            # Check curl components
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                @test abs(curlf[nx, ny, nz, 1]) < 1e-10  # curl_x should be 0
                @test abs(curlf[nx, ny, nz, 2] - expected_curl_y) < 1e-10  # curl_y should be -2π/Lx
                @test abs(curlf[nx, ny, nz, 3]) < 1e-10  # curl_z should be 0
            end
        end

        @testset "Sinusoidal field test" begin
            # Test curl of f = (0, 0, sin(2πx/Lx)) should give curl = (0, -2π/Lx * cos(2πx/Lx), 0)
            f = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            curlf = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                x = nx_to_x(f, nx)
                f[nx, ny, nz, 1] = 0.0
                f[nx, ny, nz, 2] = 0.0
                f[nx, ny, nz, 3] = sin(2π * x / Lx)
            end

            curl!(f, curlf)
            make_physical!(curlf)

            # Check curl components
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                x = nx_to_x(f, nx)
                expected_curl_y = -(2π / Lx) * cos(2π * x / Lx)

                @test abs(curlf[nx, ny, nz, 1]) < 1e-10  # curl_x should be 0
                @test abs(curlf[nx, ny, nz, 2] - expected_curl_y) < 1e-10
                @test abs(curlf[nx, ny, nz, 3]) < 1e-10  # curl_z should be 0
            end
        end

        @testset "Chebyshev polynomial field test" begin
            # Test field that varies only in y: f = (0, 0, T₁(y)) where T₁(y) = y
            # Curl should be (-∂f₃/∂y, 0, 0) = (-1, 0, 0)
            f = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            curlf = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                y = ny_to_y(f, ny)
                f[nx, ny, nz, 1] = 0.0
                f[nx, ny, nz, 2] = 0.0
                f[nx, ny, nz, 3] = y  # Linear in y (first Chebyshev polynomial T₁)
            end

            curl!(f, curlf)
            make_physical!(curlf)

            # Check curl components
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                @test abs(curlf[nx, ny, nz, 1] - (-1.0)) < 1e-10  # curl_x = -∂f₃/∂y = -1
                @test abs(curlf[nx, ny, nz, 2]) < 1e-10           # curl_y should be 0
                @test abs(curlf[nx, ny, nz, 3]) < 1e-10           # curl_z should be 0
            end
        end

        @testset "Identity: curl(curl(f)) = ∇(∇·f) - ∇²f for solenoidal f" begin
            # For a solenoidal (divergence-free) field, curl(curl(f)) = -∇²f
            # Test with f = (sin(2πx/Lx), 0, 0), which is solenoidal
            f = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            curlf = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            curl_curlf = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            # Set f = (sin(2πx/Lx), 0, 0)
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                x = nx_to_x(f, nx)
                f[nx, ny, nz, 1] = sin(2π * x / Lx)
                f[nx, ny, nz, 2] = 0.0
                f[nx, ny, nz, 3] = 0.0
            end

            curl!(f, curlf)
            curl!(curlf, curl_curlf)
            make_physical!(curl_curlf)

            # curl(curl(f)) should equal -∇²f = -(-4π²/Lx²)f = (4π²/Lx²)f
            laplacian_coeff = 4pi^2 / Lx^2

            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                x = nx_to_x(f, nx)
                expected_x = laplacian_coeff * sin(2π * x / Lx)

                @test abs(curl_curlf[nx, ny, nz, 1] - expected_x) < 1e-8
                @test abs(curl_curlf[nx, ny, nz, 2]) < 1e-10
                @test abs(curl_curlf[nx, ny, nz, 3]) < 1e-10
            end
        end
    end

    @testset "cross! Tests" begin

        @testset "Cross product with itself is zero" begin
            f = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            fcf = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            # Set non-zero field
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                f[nx, ny, nz, 1] = 1.0
                f[nx, ny, nz, 2] = 2.0
                f[nx, ny, nz, 3] = 3.0
            end

            cross!(f, f, fcf, Physical)

            # f × f should be zero
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz, i = 1:3
                @test abs(fcf[nx, ny, nz, i]) < 1e-12
            end
        end

        @testset "Standard basis vectors cross product" begin
            f = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            g = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            fcg = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            # Test e₁ × e₂ = e₃
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                f[nx, ny, nz, 1] = 1.0  # e₁
                f[nx, ny, nz, 2] = 0.0
                f[nx, ny, nz, 3] = 0.0

                g[nx, ny, nz, 1] = 0.0  # e₂
                g[nx, ny, nz, 2] = 1.0
                g[nx, ny, nz, 3] = 0.0
            end

            cross!(f, g, fcg, Physical)

            # e₁ × e₂ = e₃
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                @test abs(fcg[nx, ny, nz, 1]) < 1e-12
                @test abs(fcg[nx, ny, nz, 2]) < 1e-12
                @test abs(fcg[nx, ny, nz, 3] - 1.0) < 1e-12
            end

            # Test e₂ × e₃ = e₁
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                f[nx, ny, nz, 1] = 0.0  # e₂
                f[nx, ny, nz, 2] = 1.0
                f[nx, ny, nz, 3] = 0.0

                g[nx, ny, nz, 1] = 0.0  # e₃
                g[nx, ny, nz, 2] = 0.0
                g[nx, ny, nz, 3] = 1.0
            end

            cross!(f, g, fcg, Physical)

            # e₂ × e₃ = e₁
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                @test abs(fcg[nx, ny, nz, 1] - 1.0) < 1e-12
                @test abs(fcg[nx, ny, nz, 2]) < 1e-12
                @test abs(fcg[nx, ny, nz, 3]) < 1e-12
            end

            # Test e₃ × e₁ = e₂
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                f[nx, ny, nz, 1] = 0.0  # e₃
                f[nx, ny, nz, 2] = 0.0
                f[nx, ny, nz, 3] = 1.0

                g[nx, ny, nz, 1] = 1.0  # e₁
                g[nx, ny, nz, 2] = 0.0
                g[nx, ny, nz, 3] = 0.0
            end

            cross!(f, g, fcg, Physical)

            # e₃ × e₁ = e₂
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                @test abs(fcg[nx, ny, nz, 1]) < 1e-12
                @test abs(fcg[nx, ny, nz, 2] - 1.0) < 1e-12
                @test abs(fcg[nx, ny, nz, 3]) < 1e-12
            end
        end

        @testset "Anti-commutativity: f × g = -(g × f)" begin
            f = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            g = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            fcg = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            gcf = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            # Set arbitrary fields
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                x = nx_to_x(f, nx)
                y = ny_to_y(f, ny)
                z = nz_to_z(f, nz)

                f[nx, ny, nz, 1] = x
                f[nx, ny, nz, 2] = y
                f[nx, ny, nz, 3] = z

                g[nx, ny, nz, 1] = sin(x)
                g[nx, ny, nz, 2] = cos(y)
                g[nx, ny, nz, 3] = x * y
            end

            cross!(f, g, fcg, Physical)
            cross!(g, f, gcf, Physical)

            # Check anti-commutativity
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz, i = 1:3
                @test abs(fcg[nx, ny, nz, i] + gcf[nx, ny, nz, i]) < 1e-10
            end
        end

        @testset "Jacobi identity: a × (b × c) + b × (c × a) + c × (a × b) = 0" begin
            avec = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            bvec = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            cvec = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            # Intermediate results
            bc = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            ca = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            ab = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            # Final results
            a_bc = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            b_ca = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            c_ab = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            # Set test fields
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                avec[nx, ny, nz, 1] = 1.0
                avec[nx, ny, nz, 2] = 0.0
                avec[nx, ny, nz, 3] = 0.0

                bvec[nx, ny, nz, 1] = 0.0
                bvec[nx, ny, nz, 2] = 1.0
                bvec[nx, ny, nz, 3] = 0.0

                cvec[nx, ny, nz, 1] = 0.0
                cvec[nx, ny, nz, 2] = 0.0
                cvec[nx, ny, nz, 3] = 1.0
            end

            # Compute cross products
            cross!(bvec, cvec, bc, Physical)      # b × c
            cross!(cvec, avec, ca, Physical)      # c × a  
            cross!(avec, bvec, ab, Physical)      # a × b

            cross!(avec, bc, a_bc, Physical)      # a × (b × c)
            cross!(bvec, ca, b_ca, Physical)      # b × (c × a)
            cross!(cvec, ab, c_ab, Physical)      # c × (a × b)

            # Check Jacobi identity
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz, i = 1:3
                jacobi_sum = a_bc[nx, ny, nz, i] + b_ca[nx, ny, nz, i] + c_ab[nx, ny, nz, i]
                @test abs(jacobi_sum) < 1e-12
            end
        end

        @testset "Magnitude relationship: |a × b| = |a||b|sin(θ)" begin
            # Test with perpendicular unit vectors (θ = π/2, sin(θ) = 1)
            avec = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            bvec = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            ab = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                avec[nx, ny, nz, 1] = 2.0  # magnitude = 2
                avec[nx, ny, nz, 2] = 0.0
                avec[nx, ny, nz, 3] = 0.0

                bvec[nx, ny, nz, 1] = 0.0
                bvec[nx, ny, nz, 2] = 3.0  # magnitude = 3
                bvec[nx, ny, nz, 3] = 0.0
            end

            cross!(avec, bvec, ab, Physical)

            # |a × b| should equal |a| × |b| × sin(π/2) = 2 × 3 × 1 = 6
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                magnitude = sqrt(ab[nx, ny, nz, 1]^2 + ab[nx, ny, nz, 2]^2 + ab[nx, ny, nz, 3]^2)
                @test abs(magnitude - 6.0) < 1e-12
            end
        end
    end

    @testset "State preservation tests" begin

        @testset "curl! preserves original field state" begin
            # Test that curl! restores the original field's state
            f = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)
            curlf = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Spectral, y_state=Spectral)

            # Set some field values
            for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
                f[nx, ny, nz, 1] = sin(2π * nx_to_x(f, nx) / Lx)
                f[nx, ny, nz, 2] = 0.0
                f[nx, ny, nz, 3] = 0.0
            end

            original_xz_state = xz_state(f)
            original_y_state = y_state(f)

            curl!(f, curlf)

            # Check that original field state is preserved
            @test xz_state(f) == original_xz_state
            @test y_state(f) == original_y_state
        end

        @testset "cross! preserves original field states" begin
            f = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Spectral)
            g = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Spectral, y_state=Physical)
            fcg = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b; xz_state=Physical, y_state=Physical)

            original_f_xz_state = xz_state(f)
            original_f_y_state = y_state(f)
            original_g_xz_state = xz_state(g)
            original_g_y_state = y_state(g)

            cross!(f, g, fcg, Spectral)

            # Check that original field states are preserved
            @test xz_state(f) == original_f_xz_state
            @test y_state(f) == original_f_y_state
            @test xz_state(g) == original_g_xz_state
            @test y_state(g) == original_g_y_state
        end
    end
end

# Helper function to run all tests
function run_flowfield_calculus_tests()
    println("Running FlowField curl! and cross! tests...")
    @testset "All FlowField Calculus Tests" begin
        include("flowfield_calculus_tests.jl")  # This would be the file containing the above tests
    end
    println("All tests completed!")
end

