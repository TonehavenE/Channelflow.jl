export curl!, cross!

function curl!(f::FlowField, curlf::FlowField)
    @assert num_dimensions(f) == 3 "FlowField must have 3 dimensions for curl operation"
    sxz = xz_state(f)
    sy = y_state(f)
    make_spectral!(f)
    if !geom_congruent(f, curlf) || num_dimensions(curlf) != 3
        resize!(curlf, f.domain.Nx, f.domain.Ny, f.domain.Nz, 3, f.domain.Lx, f.domain.Lz, f.domain.a, f.domain.b)
    end
    make_state!(curlf, Spectral, Spectral)

    Mx = f.domain.Mx
    My = f.domain.My
    Mz = f.domain.Mz
    kxmax = kx_max(f)
    kzmax = kz_max(f)
    Lx_ = Lx(f)
    Lz_ = Lz(f)
    Myb = My - 1
    scale = 4.0 / Ly(f)

    # curlf[1] = df[3]/dy - df[2]/dz  (Julia 1-based indexing)
    # assign curlf[1] =  df[3]/dy;
    for mx = 1:Mx, mz = 1:Mz
        curlf[mx, My, mz, 1] = ComplexF64(0)
        curlf[mx, My-1, mz, 1] = (My - 1) * scale * f[mx, My, mz, 3]
    end
    for mx = 1:Mx, my = (My-2):-1:1, mz = 1:Mz
        curlf[mx, my, mz, 1] = curlf[mx, my+2, mz, 1] + (my + 1) * scale * f[mx, my+1, mz, 3]
    end
    for mx = 1:Mx, mz = 1:Mz
        curlf[mx, 1, mz, 1] *= 0.5
    end

    # curlf[3] = df[2]/dx - df[1]/dy  (Julia 1-based indexing)  
    # assign curlf[3] = -df[1]/dy;
    for mx = 1:Mx, mz = 1:Mz
        curlf[mx, My, mz, 3] = ComplexF64(0)
        curlf[mx, My-1, mz, 3] = -(My - 1) * scale * f[mx, My, mz, 1]
    end
    for my = (My-2):-1:1, mx = 1:Mx, mz = 1:Mz
        curlf[mx, my, mz, 3] = curlf[mx, my+2, mz, 3] - (my + 1) * scale * f[mx, my+1, mz, 1]
    end
    for mx = 1:Mx, mz = 1:Mz
        curlf[mx, 1, mz, 3] *= 0.5
    end

    # Assign d/dx and d/dz terms to curl
    for mx = 1:Mx, my = 1:My
        kx = mx_to_kx(f, mx)
        d_dx = Complex(0.0, 2 * pi * kx / Lx_ * zero_last_mode(kx, kxmax, 1))
        for mz = 1:Mz
            kz = mz_to_kz(f, mz)  # Note: was mx_to_kz in original, should be mz_to_kz
            d_dz = Complex(0.0, 2 * pi * kz / Lz_ * zero_last_mode(kz, kzmax, 1))
            f0 = f[mx, my, mz, 1]  # f[0] in C++ = f[1] in Julia
            f1 = f[mx, my, mz, 2]  # f[1] in C++ = f[2] in Julia
            f2 = f[mx, my, mz, 3]  # f[2] in C++ = f[3] in Julia
            curlf[mx, my, mz, 1] -= d_dz * f1
            curlf[mx, my, mz, 2] = d_dz * f0 - d_dx * f2
            curlf[mx, my, mz, 3] += d_dx * f1
        end
    end

    make_state!(f, sxz, sy)
end

function cross!(f::FlowField, g::FlowField, fcg::FlowField)
    # Store original states to restore later
    fxz = xz_state(f)
    fy = y_state(f)
    gxz = xz_state(g)
    gy = y_state(g)

    @assert congruent(g, f) "FlowFields f and g must be congruent"
    @assert num_dimensions(f) == 3 && num_dimensions(g) == 3 "FlowFields must have 3 dimensions for cross product"

    # Convert to physical space for cross product computation
    make_physical!(f)
    make_physical!(g)

    # Resize output field if necessary
    if !geom_congruent(f, fcg) || num_dimensions(fcg) != 3
        resize!(fcg, f.domain.Nx, f.domain.Ny, f.domain.Nz, 3, f.domain.Lx, f.domain.Lz, f.domain.a, f.domain.b)
    end

    # Set output to physical space
    make_state!(fcg, Physical, Physical)

    Nx = f.domain.Nx
    Ny = f.domain.Ny
    Nz = f.domain.Nz
    Nd = num_dimensions(f)

    # Compute cross product: fcg = f × g
    # fcg[i] = f[j]*g[k] - f[k]*g[j] where j = (i+1)%3, k = (i+2)%3
    for i = 1:Nd
        j = ((i - 1 + 1) % 3) + 1  # Convert to 1-based: (i+1)%3 in 0-based becomes ((i-1+1)%3)+1 in 1-based
        k = ((i - 1 + 2) % 3) + 1  # Convert to 1-based: (i+2)%3 in 0-based becomes ((i-1+2)%3)+1 in 1-based
        for ny = 1:Ny, nx = 1:Nx, nz = 1:Nz
            fcg[nx, ny, nz, i] = f[nx, ny, nz, j] * g[nx, ny, nz, k] - f[nx, ny, nz, k] * g[nx, ny, nz, j]
        end
    end

    # Restore original states
    make_state!(f, fxz, fy)
    make_state!(g, gxz, gy)

    # Convert result to spectral space
    make_spectral!(fcg)
end