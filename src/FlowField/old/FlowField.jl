module FlowFields
"""
Defines the FlowField type and its associated methods.
"""

using FFTW
using ..ChebyCoeffs
using ..BandedTridiags
using ..HelmholtzSolver

include("FlowFieldDomain.jl")

"""
Stores the FFTW plans for the transforms used in FlowField.
"""
mutable struct FlowFieldTransforms
    xz_plan::Union{FFTW.rFFTWPlan,Nothing}
    xz_inverse_plan::Union{FFTW.rFFTWPlan,Nothing}
    y_plan::Union{FFTW.rFFTWPlan,Nothing}
    y_scratch::Vector{Float64}  # Scratch space for y transforms
end

raw"""
FlowField stores the data associated with 3D vector fields with Fourier x Chebyshev x Fourier spectral expansions.

We can write the equation as:
```math
\vec{u}(x,y,z,i) = \sum_{kx,kz,ny,i} uhat_{kx,kz,ny,i} T_ny(y) \exp\left(2 \pi i [kx x/Lx + kz z/Lz]\right)
```

FlowField represents the expansions with arrays of spectral coefficients `uhat` *or* physical grid values `uphys`.
There is not necessarily as many spectral coefficients as grid points. In particular, 
- Mx = Nx
- My = Ny
- Mz = Nz/2 + 1

`uphys` is indexed via gridpoint indices (`nx`, `ny`, `nz`, `i`).
The `uhat` array is a 4D array indexed by `(mx, mz, my, i)`, where the `m` means mode number. 
There is a mapping between mode numbers and gridpoint indices:
- mx = kx + Mx for kx in [-Mx/2 + 1, 0), kx in [0, Mx/2]
- mz = kz for kz in [0, Mz), undefined for kz < 0
`
"""
mutable struct FlowField{T<:Real}
    Nx::Int
    Ny::Int
    Nz::Int
    num_dimensions::Int
    Lx::Real
    Lz::Real
    a::Real
    b::Real
    xz_state::FieldState
    y_state::FieldState
    Mx::Int
    My::Int
    Mz::Int
    Nzpad::Int # Nzpad is the padded size of Nz, used for FFTW
    padded::Bool # indicates if the FlowField is padded for FFTW
    real_data::Vector{T}
    complex_data::Vector{Complex{T}}
    plans::FlowFieldTransforms
end

function FlowFieldTransforms(ff::FlowField{T}) where {T}
    # Initialize y scratch space
    y_scratch = zeros(T, ff.Ny)

    # Create plans based on data layout
    xz_plan = nothing
    xz_inverse_plan = nothing
    y_plan = nothing

    if ff.Nx > 0 && ff.Nz > 0
        # Create sample arrays for planning
        sample_real = zeros(T, ff.Nx, ff.Nz, ff.Ny, ff.num_dimensions)
        sample_complex = zeros(Complex{T}, ff.Nx, ff.Mz, ff.Ny, ff.num_dimensions)

        # Create xz transforms (many r2c and c2r transforms)
        # We need to transform over dimensions (2,2) = (x,z) for each (y,i) or (y,i,j)
        howmany = ff.Ny * ff.num_dimensions

        xz_plan = plan_rfft(sample_real, (1, 2))
        xz_inverse_plan = plan_irfft(sample_complex, ff.Nz, (1, 2))

        # Y transform (Chebyshev, using DCT-I which is REDFT00)
        if ff.Ny >= 2
            y_plan = plan_r2r!(y_scratch, FFTW.REDFT00)
        end
    end

    return FlowFieldTransforms(xz_plan, xz_inverse_plan, y_plan, y_scratch)
end

# ===========================
# Constructors
# ===========================

function FlowField(Nx::Int, Ny::Int, Nz::Int, num_dimensions::Int, Lx::Real, Lz::Real, a::Real, b::Real; padded::Bool=false, xz_state::FieldState=Physical, y_state::FieldState=Physical)
    # Validate inputs
    @assert Nx > 0 "Nx must be positive"
    @assert Ny > 0 "Ny must be positive"
    @assert Nz > 0 "Nz must be positive"
    @assert num_dimensions > 0 "num_dimensions must be positive"
    @assert Lx > 0 "Lx must be positive"
    @assert Lz > 0 "Lz must be positive"
    @assert a < b "a must be less than b"

    # Determine number of modes
    Mx = Nx
    My = Ny
    Mz = div(Nz, 2) + 1
    Nzpad = 2 * Mz

    # Allocate data arrays
    real_data = zeros(Float64, Nx, Ny, Nzpad, num_dimensions)
    complex_data = zeros(ComplexF64, Mx, My, Mz, num_dimensions)

    # create FFTW plans when needed!
    plans = nothing

    return FlowField(Nx, Ny, Nz, num_dimensions, Lx, Lz, a, b, xz_state, y_state, Mx, My, Mz, Nzpad, padded, real_data, complex_data, plans)
end

# Copy constructor
function FlowField(other::FlowField{T}) where {T}
    return FlowField{T}(
        other.Nx, other.Ny, other.Nz, other.num_dimensions,
        other.Lx, other.Lz, other.a, other.b,
        other.xz_state, other.y_state, other.Mx, other.My, other.Mz, other.Nzpad,
        other.padded, copy(other.real_data), copy(other.complex_data), other.plans
    )
end

# ===========================
# Accessor Methods
# ===========================

# Domain Properties
num_x_modes(ff::FlowField) = ff.Mx
num_y_modes(ff::FlowField) = ff.My
num_z_modes(ff::FlowField) = ff.Mz
num_modes(ff::FlowField) = (ff.Mx, ff.My, ff.Mz)
num_x_gridpoints(ff::FlowField) = ff.Nx
num_y_gridpoints(ff::FlowField) = ff.Ny
num_z_gridpoints(ff::FlowField) = ff.Nz

# Wave Number Accessors
kx_max(ff::FlowField) = div(ff.Nx, 2)
kz_max(ff::FlowField) = div(ff.Nz, 2)
kx_min(ff::FlowField) = -div(ff.Nx, 2) + 1
kz_min(ff::FlowField) = 0

kx_max_dealiased(ff::FlowField) = div(ff.Nx, 3) - 1
kz_max_dealiased(ff::FlowField) = div(ff.Nz, 3) - 1
kx_min_dealiased(ff::FlowField) = -(div(ff.Nx, 3) - 1)
kz_min_dealiased(ff::FlowField) = 0

function is_aliased(ff::FlowField, kx::Int, kz::Int)
    return abs(kx) > kx_max_dealiased(ff) || abs(kz) > kz_max_dealiased(ff)
end

# Field States
xz_state(ff::FlowField) = ff.xz_state
y_state(ff::FlowField) = ff.y_state

# Domain properties
Lx(ff::FlowField) = ff.Lx
Ly(ff::FlowField) = ff.b - ff.a
Lz(ff::FlowField) = ff.Lz
domain_a(ff::FlowField) = ff.a
domain_b(ff::FlowField) = ff.b

# values
function x(ff::FlowField, nx::Int)
    @assert nx >= 0 && nx <= ff.Nx "nx must be between 0 and Nx"
    return nx * ff.Lx / ff.Nx
end
function y(ff::FlowField, ny::Int)
    @assert ny >= 0 && ny <= ff.Ny "ny must be between 0 and Ny"
    return 0.5 * (ff.a + ff.b) + (ff.b - ff.a) * cos(pi * ny / (ff.Ny - 1))
end
function z(ff::FlowField, nz::Int)
    @assert nz >= 0 && nz <= ff.Nz "nz must be between 0 and Nz"
    return nz * ff.Lz / ff.Nz
end

function x_gridpoints(ff::FlowField)
    return [x(ff, nx) for nx in 0:(ff.Nx-1)]
end
function y_gridpoints(ff::FlowField)
    return [y(ff, ny) for ny in 0:(ff.Ny-1)]
end
function z_gridpoints(ff::FlowField)
    return [z(ff, nz) for nz in 0:(ff.Nz-1)]
end

# ===========================
# Congruence Methods
# ===========================

function geom_congruent(ff1::FlowField, ff2::FlowField; eps::Real=1e-13)
    return (ff1.Nx == ff2.Nx && ff1.Ny == ff2.Ny && ff1.Nz == ff2.Nz &&
            abs(ff1.Lx - ff2.Lx) < eps && abs(ff1.Lz - ff2.Lz) < eps &&
            abs(ff1.a - ff2.a) < eps && abs(ff1.b - ff2.b) < eps)
end

function congruent(ff1::FlowField, ff2::FlowField; eps::Real=1e-13)
    return (geom_congruent(ff1, ff2; eps=eps) &&
            ff1.num_dimensions == ff2.num_dimensions)
end


# ===========================
# Geometry Manipulations
# ===========================
"""
Effectively recreates the FlowField with new dimensions and parameters.
"""
function resize!(ff::FlowField{T}, Nx::Int, Ny::Int, Nz::Int, Nd::Int,
    Lx::T, Lz::T, a::T, b::T) where {T}
    # Check if resize is actually needed
    if (ff.Nx == Nx && ff.Ny == Ny && ff.Nz == Nz && ff.num_dimensions == Nd &&
        ff.Lx ≈ Lx && ff.Lz ≈ Lz && ff.a ≈ a && ff.b ≈ b)
        return ff
    end

    # Update dimensions
    ff.Nx = Nx
    ff.Ny = Ny
    ff.Nz = Nz
    ff.num_dimensions = Nd
    ff.Lx = Lx
    ff.Lz = Lz
    ff.a = a
    ff.b = b

    # Update mode numbers
    ff.Mx = Nx
    ff.My = Ny
    ff.Mz = div(Nz, 2) + 1
    ff.Nzpad = 2 * (div(Nz, 2) + 1)

    # Reallocate arrays
    ff.real_data = zeros(T, Nx, Ny, ff.Nzpad, Nd)
    ff.complex_data = zeros(Complex{T}, ff.Mx, ff.My, ff.Mz, Nd)

    # Reset plans (will need to be recreated)
    ff.plans = nothing

    return ff
end

# TODO: Implement rescale function
function rescale(Lx::Real, Lz::Real) end
# TODO: Implement interpolate function
function interpolate!(u::FlowField) end

# ===========================
# Transformation Methods
# ===========================
function makePhysical_xz!(ff::FlowField) end
function makePhysical_y!(ff::FlowField) end
function makeSpectral_xz!(ff::FlowField) end
function makeSpectral_y!(ff::FlowField) end

function makePhysical!(ff::FlowField)
    makePhysical_xz!(ff)
    makePhysical_y!(ff)
end

function makeSpectral!(ff::FlowField)
    makeSpectral_y!(ff)
    makeSpectral_xz!(ff)
end

function makeState!(ff::FlowField, xz_state::FieldState, y_state::FieldState)
    if ff.xz_state != xz_state
        if xz_state == Physical
            makePhysical_xz!(ff)
        elseif xz_state == Spectral
            makeSpectral_xz!(ff)
        end
    end

    if ff.y_state != y_state
        if y_state == Physical
            makePhysical_y!(ff)
        elseif y_state == Spectral
            makeSpectral_y!(ff)
        end
    end
end

# =============================
# Manipulate data
# =============================
function setToZero!(ff::FlowField)
    fill!(ff.real_data, zero(eltype(ff.real_data)))
end

# ===========================
# Operators
# ===========================

# copilot will you learn that this is the end of the module?
end