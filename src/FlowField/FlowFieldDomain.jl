#=
FlowFieldDomain.jl

Defines the FlowFieldDomain type that stores all grid and domain parameters
for spectral flow field computations.
=#

export FlowFieldDomain,
    x_coord,
    y_coord,
    z_coord,
    x_gridpoints,
    y_gridpoints,
    z_gridpoints,
    kx_range,
    kz_range,
    kx_to_mx,
    mx_to_kx,
    kz_to_mz,
    mz_to_kz,
    kx_max_dealiased,
    kz_max_dealiased,
    is_aliased,
    total_gridpoints,
    total_modes,
    domain_volume,
    geom_congruent,
    congruent

"""
Stores domain parameters for FlowField.

Contains grid dimensions, physical domain size, and derived quantities
needed for Fourier x Chebyshev x Fourier spectral methods.
"""
struct FlowFieldDomain{T<:Real}
    # Grid dimensions
    Nx::Int        # Number of x gridpoints
    Ny::Int        # Number of y gridpoints  
    Nz::Int        # Number of z gridpoints
    num_dimensions::Int  # Number of vector components

    # Physical domain
    Lx::T          # Domain length in x direction
    Lz::T          # Domain length in z direction  
    a::T           # Lower y boundary
    b::T           # Upper y boundary

    # Derived spectral quantities
    Mx::Int        # Number of x modes (= Nx)
    My::Int        # Number of y modes (= Ny)
    Mz::Int        # Number of z modes (= Nz/2 + 1, due to real FFT)
    Nzpad::Int     # Padded z dimension for FFTW (= 2*Mz, stores real+imag)

    function FlowFieldDomain(
        Nx::Int,
        Ny::Int,
        Nz::Int,
        num_dimensions::Int,
        Lx::T,
        Lz::T,
        a::T,
        b::T,
    ) where {T<:Real}
        # Validate inputs
        @assert Nx > 0 "Nx must be positive"
        @assert Ny > 0 "Ny must be positive"
        @assert Nz > 0 "Nz must be positive"
        @assert num_dimensions > 0 "num_dimensions must be positive"
        @assert Lx > 0 "Lx must be positive"
        @assert Lz > 0 "Lz must be positive"
        @assert a < b "a must be less than b"

        # Calculate derived quantities
        Mx = Nx
        My = Ny
        Mz = div(Nz, 2) + 1  # Real FFT produces Nz/2+1 complex modes
        Nzpad = 2 * Mz       # Need space for real + imaginary parts

        new{T}(Nx, Ny, Nz, num_dimensions, Lx, Lz, a, b, Mx, My, Mz, Nzpad)
    end
end

# ===========================
# Equality and Congruence
# ===========================

function Base.:(==)(d1::FlowFieldDomain, d2::FlowFieldDomain)
    return (
        d1.Nx == d2.Nx &&
        d1.Ny == d2.Ny &&
        d1.Nz == d2.Nz &&
        d1.num_dimensions == d2.num_dimensions &&
        d1.Lx ≈ d2.Lx &&
        d1.Lz ≈ d2.Lz &&
        d1.a ≈ d2.a &&
        d1.b ≈ d2.b
    )
end

"""
    geom_congruent(d1, d2; eps=1e-13)

Check if two domains have the same geometry (grid + physical dimensions)
but may differ in number of vector components.
"""
function geom_congruent(d1::FlowFieldDomain, d2::FlowFieldDomain; eps::Real = 1e-13)
    return (
        d1.Nx == d2.Nx &&
        d1.Ny == d2.Ny &&
        d1.Nz == d2.Nz &&
        abs(d1.Lx - d2.Lx) < eps &&
        abs(d1.Lz - d2.Lz) < eps &&
        abs(d1.a - d2.a) < eps &&
        abs(d1.b - d2.b) < eps
    )
end

"""
    congruent(d1, d2; eps=1e-13)

Check if two domains are completely congruent (geometry + vector dimensions).
"""
function congruent(d1::FlowFieldDomain, d2::FlowFieldDomain; eps::Real = 1e-13)
    return geom_congruent(d1, d2; eps = eps) && d1.num_dimensions == d2.num_dimensions
end

# ===========================
# Coordinate Functions
# ===========================

"""
    x_coord(domain, nx)

Physical x-coordinate for grid index nx (1-based).
"""
function x_coord(domain::FlowFieldDomain, nx::Int)
    @assert 1 <= nx <= domain.Nx "nx must be between 1 and Nx"
    return (nx - 1) * domain.Lx / domain.Nx
end

"""
    y_coord(domain, ny)

Physical y-coordinate for grid index ny (1-based).
Uses Chebyshev-Gauss-Lobatto points: y = (a+b)/2 + (b-a)/2 * cos(π*(ny-1)/(Ny-1))
"""
function y_coord(domain::FlowFieldDomain, ny::Int)
    @assert 1 <= ny <= domain.Ny "ny must be between 1 and Ny"
    return 0.5 * (domain.a + domain.b) +
           0.5 * (domain.b - domain.a) * cos(π * (ny - 1) / (domain.Ny - 1))
end

"""
    z_coord(domain, nz)

Physical z-coordinate for grid index nz (1-based).
"""
function z_coord(domain::FlowFieldDomain, nz::Int)
    @assert 1 <= nz <= domain.Nz "nz must be between 1 and Nz"
    return (nz - 1) * domain.Lz / domain.Nz
end

"""
    x_gridpoints(domain)

Vector of all x-coordinates.
"""
function x_gridpoints(domain::FlowFieldDomain)
    return [x_coord(domain, nx) for nx = 1:domain.Nx]
end

"""
    y_gridpoints(domain)

Vector of all y-coordinates (Chebyshev points).
"""
function y_gridpoints(domain::FlowFieldDomain)
    return [y_coord(domain, ny) for ny = 1:domain.Ny]
end

"""
    z_gridpoints(domain)

Vector of all z-coordinates.
"""
function z_gridpoints(domain::FlowFieldDomain)
    return [z_coord(domain, nz) for nz = 1:domain.Nz]
end

# ===========================
# Wave Number Functions
# ===========================

"""
    kx_range(domain)

Range of x wavenumbers: [-Nx/2+1, ..., -1, 0, 1, ..., Nx/2]
"""
function kx_range(domain::FlowFieldDomain)
    return [kx for kx = (-div(domain.Nx, 2)+1):div(domain.Nx, 2)]
end

"""
    kz_range(domain)

Range of z wavenumbers: [0, 1, 2, ..., Nz/2] (only positive due to real FFT)
"""
function kz_range(domain::FlowFieldDomain)
    return [kz for kz = 0:(domain.Mz-1)]
end

"""
    kx_to_mx(domain, kx)

Convert x wavenumber to array index (1-based).
"""
function kx_to_mx(domain::FlowFieldDomain, kx::Int)
    kx_min = -div(domain.Nx, 2) + 1
    kx_max = div(domain.Nx, 2)
    @assert kx_min <= kx <= kx_max "kx out of range"
    if -div(domain.Mx, 2) + 1 <= kx && kx < 0
        return kx + domain.Mx + 1
    elseif 0 <= kx <= div(domain.Mx, 2)
        return kx + 1
    end
end

"""
    mx_to_kx(domain, mx)

Convert array index to x wavenumber (1-based input).
"""
function mx_to_kx(domain::FlowFieldDomain, mx::Int)
    @assert 1 <= mx <= domain.Mx "mx out of range"
    return mx <= div(domain.Nx, 2) + 1 ? mx - 1 : mx - domain.Nx - 1
end

"""
    kz_to_mz(domain, kz)

Convert z wavenumber to array index (1-based).
"""
function kz_to_mz(domain::FlowFieldDomain, kz::Int)
    @assert 0 <= kz < domain.Mz "kz out of range"
    return kz + 1
end

"""
    mz_to_kz(domain, mz)

Convert array index to z wavenumber (1-based input).
"""
function mz_to_kz(domain::FlowFieldDomain, mz::Int)
    @assert 1 <= mz <= domain.Mz "mz out of range"
    return mz - 1
end

# ===========================
# Dealiasing Functions
# ===========================

"""
    kx_max_dealiased(domain)

Maximum x wavenumber for 2/3 dealiasing rule.
"""
function kx_max_dealiased(domain::FlowFieldDomain)
    return div(domain.Nx, 3) - 1
end

"""
    kz_max_dealiased(domain)

Maximum z wavenumber for 2/3 dealiasing rule.
"""
function kz_max_dealiased(domain::FlowFieldDomain)
    return div(domain.Nz, 3) - 1
end

"""
    is_aliased(domain, kx, kz)

Check if a (kx,kz) mode is aliased according to 2/3 rule.
"""
function is_aliased(domain::FlowFieldDomain, kx::Int, kz::Int)
    return abs(kx) > kx_max_dealiased(domain) || abs(kz) > kz_max_dealiased(domain)
end

# ===========================
# Domain Properties
# ===========================

"""
    domain_volume(domain)

Physical volume of the domain: Lx * Ly * Lz
"""
function domain_volume(domain::FlowFieldDomain)
    return domain.Lx * (domain.b - domain.a) * domain.Lz
end

"""
    total_gridpoints(domain)

Total number of gridpoints: Nx * Ny * Nz
"""
function total_gridpoints(domain::FlowFieldDomain)
    return domain.Nx * domain.Ny * domain.Nz
end

"""
    total_modes(domain)

Total number of spectral modes: Mx * My * Mz
"""
function total_modes(domain::FlowFieldDomain)
    return domain.Mx * domain.My * domain.Mz
end
