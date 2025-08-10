using Test
using Channelflow

@testset "FlowField L2 Norms and Inner Products" begin

    @testset "Basic Check" begin
        ff = FlowField(8, 5, 16, 3, 2π, 1π, -1.0, 1.0)
        make_spectral_xz!(ff)
        make_spectral_y!(ff)
        @test L2Norm(ff) == 0.0
        @test L2Norm2(ff) == 0.0
        # @test L2InnerProduct(ff, ff) == 0.0
    end

    @testset "3D Trigonometry" begin
        #=
        Testing with a known equation:
        f(x, y, z) = sin(\pi x/Lx) * cos(\pi * (y - a)/(b - a)) * sin(pi * z/Lz)
        =#
        Lx = 1.0
        Lz = 1.0
        a = -1.0
        b = 1.0
        f(x, y, z) = sin(π * x / Lx) * cos(π * (y - a) / (b - a)) * sin(π * z / Lz)
        N = 2
        Nx = N
        Ny = N + 1
        Nz = N
        Nd = 1

        ff = FlowField(Nx, Ny, Nz, Nd, Lx, Lz, a, b; xz_state=Spectral, y_state=Spectral)  # Only y variation
        make_physical!(ff)

        for nx = 1:Nx, ny = 1:Ny, nz = 1:Nz
            x_val = x_coord(ff.domain, nx)
            y_val = y_coord(ff.domain, ny)
            z_val = z_coord(ff.domain, nz)
            ff[nx, ny, nz, 1] = f(x_val, y_val, z_val)
        end

        make_spectral_xz!(ff)
        make_spectral_y!(ff)

        @test L2Norm2(ff, false) ≈ (b - a) / 8

    end
    @testset "Vector Valued Function" begin
        #= 
        Testing with the vector valued function u:
        u(x, y, z) = [0, 0, sin(αx) * (1 - y^2)]

        Analytically, we find the L2Norm2 to be (32pi^2)/(15 alpha * gamma).
        Dividing by the normalizing constant of Lx * Lz = (2pi/alpha) * (2pi/gamma) = 4pi^2/(alpha * gamma),
        we find the final result to be: 
        8/(15 * alpha * gamma)

        This tests that for various (alpha, gamma) in the unnormalized case, and one particular normalized one.
        =#
        Nx = 16
        Ny = 33
        Nz = 16
        ua = -1.0
        ub = 1.0

        # unnormalized tests
        for alpha in 1:10
            for gamma in 1:10
                Lx = 2 * pi / alpha
                Lz = 2 * pi / gamma
                u = FlowField(Nx, Ny, Nz, 3, Lx, Lz, ua, ub)
                make_state!(u, Physical, Physical)

                for nx = 1:Nx
                    x = x_coord(u.domain, nx)
                    for nz = 1:Nz
                        z = z_coord(u.domain, nz)
                        for ny = 1:Ny
                            y = y_coord(u.domain, ny)
                            u[nx, ny, nz, 1] = 0.0
                            u[nx, ny, nz, 2] = 0.0
                            u[nx, ny, nz, 3] = sin(alpha * x) * (1 - y * y)
                        end
                    end
                end
                make_spectral!(u)
                @test L2Norm2(u, false) ≈ (32 * pi^2) / (15 * alpha * gamma)
                @test L2Norm(u, false) ≈ sqrt((32 * pi^2) / (15 * alpha * gamma))
            end
        end
        alpha = 1.0
        gamma = 2.0
        Lx = 2 * pi / alpha
        Lz = 2 * pi / gamma
        u = FlowField(Nx, Ny, Nz, 3, Lx, Lz, ua, ub)
        make_state!(u, Physical, Physical)

        for nx = 1:Nx
            x = x_coord(u.domain, nx)
            for nz = 1:Nz
                z = z_coord(u.domain, nz)
                for ny = 1:Ny
                    y = y_coord(u.domain, ny)
                    u[nx, ny, nz, 1] = 0.0
                    u[nx, ny, nz, 2] = 0.0
                    u[nx, ny, nz, 3] = sin(alpha * x) * (1 - y * y)
                end
            end
        end
        make_spectral!(u)
        @test L2Norm2(u) ≈ 4 / 15 # alpha = 2, gamma =1
        @test L2Norm(u) ≈ sqrt(4 / 15)
    end
end