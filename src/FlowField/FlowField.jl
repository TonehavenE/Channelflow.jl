module FlowFields

"""
FlowField.jl

Main FlowField implementation using single array storage like the original C++ code.
"""

using FFTW
include("FlowFieldDomain.jl")
include("FlowFieldTransforms.jl")

@enum FieldState Physical Spectral

"""
FlowField stores 3D vector fields with Fourier x Chebyshev x Fourier spectral expansions.

Key design decisions matching the original C++ implementation:
1. Single data array `data` stores both physical and spectral values
2. In physical state: data[nx,ny,nz,i] = u(x,y,z,i) 
3. In spectral state: data contains packed complex coefficients
   - Real parts at data[nx,ny,2*mz-1,i]  
   - Imaginary parts at data[nx,ny,2*mz,i]
4. FFTW plans are created once and reused
5. Transforms are done in-place to minimize memory allocation

Data layout: [Nx, Ny, Nzpad, num_dimensions] where Nzpad = 2*(Nz/2+1)
"""
mutable struct FlowField{T<:Real}
    domain::FlowFieldDomain{T}
    xz_state::FieldState
    y_state::FieldState
    padded::Bool  # Flag for dealiasing
    data::Array{T,4}  # Single array: [Nx, Ny, Nzpad, num_dimensions]
    transforms::FlowFieldTransforms{T}
end

# ===========================
# Constructors
# ===========================

"""
    FlowField(Nx, Ny, Nz, num_dimensions, Lx, Lz, a, b; kwargs...)

Create a FlowField with specified grid and domain parameters.
FFTW plans are created immediately for efficiency.
"""
function FlowField(Nx::Int, Ny::Int, Nz::Int, num_dimensions::Int,
    Lx::T, Lz::T, a::T, b::T;
    padded::Bool=false,
    xz_state::FieldState=Physical,
    y_state::FieldState=Physical) where {T<:Real}

    # Create domain
    domain = FlowFieldDomain(Nx, Ny, Nz, num_dimensions, Lx, Lz, a, b)

    # Allocate single data array with padding
    data = zeros(T, domain.Nx, domain.Ny, domain.Nzpad, domain.num_dimensions)

    # Create FFTW plans
    transforms = FlowFieldTransforms(domain)

    return FlowField{T}(domain, xz_state, y_state, padded, data, transforms)
end

"""
    FlowField(domain; kwargs...)

Create FlowField from existing domain object.
"""
function FlowField(domain::FlowFieldDomain{T};
    padded::Bool=false,
    xz_state::FieldState=Physical,
    y_state::FieldState=Physical) where {T}

    data = zeros(T, domain.Nx, domain.Ny, domain.Nzpad, domain.num_dimensions)
    transforms = FlowFieldTransforms(domain)

    return FlowField{T}(domain, xz_state, y_state, padded, data, transforms)
end

"""
    FlowField(other::FlowField)

Copy constructor - creates deep copy of data but shares FFTW plans.
"""
function FlowField(other::FlowField{T}) where {T}
    return FlowField{T}(other.domain, other.xz_state, other.y_state,
        other.padded, copy(other.data), other.transforms)
end

# ===========================
# Element Access Methods
# ===========================

"""
    ff[nx, ny, nz, i]

Access physical field values. Requires FlowField to be in Physical xz state.
Uses 1-based indexing.
"""
function Base.getindex(ff::FlowField, nx::Int, ny::Int, nz::Int, i::Int)
    @assert ff.xz_state == Physical "FlowField must be in Physical xz state for direct access"
    @assert 1 <= nx <= ff.domain.Nx "nx out of bounds"
    @assert 1 <= ny <= ff.domain.Ny "ny out of bounds"
    @assert 1 <= nz <= ff.domain.Nz "nz out of bounds"
    @assert 1 <= i <= ff.domain.num_dimensions "component index i out of bounds"
    return ff.data[nx, ny, nz, i]
end

function Base.setindex!(ff::FlowField, val, nx::Int, ny::Int, nz::Int, i::Int)
    @assert ff.xz_state == Physical "FlowField must be in Physical xz state for direct access"
    @assert 1 <= nx <= ff.domain.Nx "nx out of bounds"
    @assert 1 <= ny <= ff.domain.Ny "ny out of bounds"
    @assert 1 <= nz <= ff.domain.Nz "nz out of bounds"
    @assert 1 <= i <= ff.domain.num_dimensions "component index i out of bounds"
    ff.data[nx, ny, nz, i] = val
end

"""
    cmplx(ff, mx, my, mz, i)

Access spectral coefficients. Requires FlowField to be in Spectral xz state.
Returns Complex{T} by unpacking from real storage format.
"""
function cmplx(ff::FlowField{T}, mx::Int, my::Int, mz::Int, i::Int) where {T}
    @assert ff.xz_state == Spectral "FlowField must be in Spectral xz state for spectral access"
    @assert 1 <= mx <= ff.domain.Mx "mx out of bounds"
    @assert 1 <= my <= ff.domain.My "my out of bounds"
    @assert 1 <= mz <= ff.domain.Mz "mz out of bounds"
    @assert 1 <= i <= ff.domain.num_dimensions "component index i out of bounds"

    # Unpack complex value from real storage
    z_real_idx = 2 * mz - 1
    z_imag_idx = 2 * mz

    real_part = (z_real_idx <= ff.domain.Nzpad) ? ff.data[mx, my, z_real_idx, i] : T(0)
    imag_part = (z_imag_idx <= ff.domain.Nzpad) ? ff.data[mx, my, z_imag_idx, i] : T(0)

    return Complex{T}(real_part, imag_part)
end

"""
    set_cmplx!(ff, val, mx, my, mz, i)

Set spectral coefficient. Requires FlowField to be in Spectral xz state.
Packs Complex{T} into real storage format.
"""
function set_cmplx!(ff::FlowField{T}, val::Complex{T}, mx::Int, my::Int, mz::Int, i::Int) where {T}
    @assert ff.xz_state == Spectral "FlowField must be in Spectral xz state for spectral access"
    @assert 1 <= mx <= ff.domain.Mx "mx out of bounds"
    @assert 1 <= my <= ff.domain.My "my out of bounds"
    @assert 1 <= mz <= ff.domain.Mz "mz out of bounds"
    @assert 1 <= i <= ff.domain.num_dimensions "component index i out of bounds"

    # Pack complex value into real storage
    z_real_idx = 2 * mz - 1
    z_imag_idx = 2 * mz

    if z_real_idx <= ff.domain.Nzpad
        ff.data[mx, my, z_real_idx, i] = real(val)
    end
    if z_imag_idx <= ff.domain.Nzpad
        ff.data[mx, my, z_imag_idx, i] = imag(val)
    end
end

# ===========================
# Accessor Methods (delegate to domain)
# ===========================

# Grid dimensions
num_x_gridpoints(ff::FlowField) = ff.domain.Nx
num_y_gridpoints(ff::FlowField) = ff.domain.Ny
num_z_gridpoints(ff::FlowField) = ff.domain.Nz
num_gridpoints(ff::FlowField) = (ff.domain.Nx, ff.domain.Ny, ff.domain.Nz)

# Spectral dimensions
num_x_modes(ff::FlowField) = ff.domain.Mx
num_y_modes(ff::FlowField) = ff.domain.My
num_z_modes(ff::FlowField) = ff.domain.Mz
num_modes(ff::FlowField) = (ff.domain.Mx, ff.domain.My, ff.domain.Mz)

# Field properties
vector_dim(ff::FlowField) = ff.domain.num_dimensions
xz_state(ff::FlowField) = ff.xz_state
y_state(ff::FlowField) = ff.y_state

# Domain properties
Lx(ff::FlowField) = ff.domain.Lx
Ly(ff::FlowField) = ff.domain.b - ff.domain.a
Lz(ff::FlowField) = ff.domain.Lz
domain_a(ff::FlowField) = ff.domain.a
domain_b(ff::FlowField) = ff.domain.b

# Coordinate functions (delegate to domain)
x(ff::FlowField, nx::Int) = x_coord(ff.domain, nx)
y(ff::FlowField, ny::Int) = y_coord(ff.domain, ny)
z(ff::FlowField, nz::Int) = z_coord(ff.domain, nz)

x_gridpoints(ff::FlowField) = x_gridpoints(ff.domain)
y_gridpoints(ff::FlowField) = y_gridpoints(ff.domain)
z_gridpoints(ff::FlowField) = z_gridpoints(ff.domain)

# Wave number functions (delegate to domain)
kx_to_mx(ff::FlowField, kx::Int) = kx_to_mx(ff.domain, kx)
mx_to_kx(ff::FlowField, mx::Int) = mx_to_kx(ff.domain, mx)
kz_to_mz(ff::FlowField, kz::Int) = kz_to_mz(ff.domain, kz)
mz_to_kz(ff::FlowField, mz::Int) = mz_to_kz(ff.domain, mz)

# Dealiasing
kx_max_dealiased(ff::FlowField) = kx_max_dealiased(ff.domain)
kz_max_dealiased(ff::FlowField) = kz_max_dealiased(ff.domain)
is_aliased(ff::FlowField, kx::Int, kz::Int) = is_aliased(ff.domain, kx, kz)

# ===========================
# Congruence Methods  
# ===========================

function geom_congruent(ff1::FlowField, ff2::FlowField; eps::Real=1e-13)
    return geom_congruent(ff1.domain, ff2.domain; eps=eps)
end

function congruent(ff1::FlowField, ff2::FlowField; eps::Real=1e-13)
    return congruent(ff1.domain, ff2.domain; eps=eps)
end

# ===========================
# Transform Methods
# ===========================

"""
    make_spectral_xz!(ff)

Transform x,z directions from physical to spectral (Fourier) space.
Uses in-place FFTW transforms with proper normalization.
"""
function make_spectral_xz!(ff::FlowField{T}) where {T}
    if ff.xz_state == Spectral
        return ff
    end

    make_spectral_xz!(ff.data, ff.domain, ff.transforms)
    ff.xz_state = Spectral
    return ff
end

"""
    make_physical_xz!(ff)

Transform x,z directions from spectral to physical space.
Uses in-place inverse FFTW transforms.
"""
function make_physical_xz!(ff::FlowField{T}) where {T}
    if ff.xz_state == Physical
        return ff
    end

    make_physical_xz!(ff.data, ff.domain, ff.transforms)
    ff.xz_state = Physical
    return ff
end

"""
    make_spectral_y!(ff)

Transform y direction from physical to spectral (Chebyshev) space.
Uses DCT-I with proper Chebyshev normalization.
"""
function make_spectral_y!(ff::FlowField{T}) where {T}
    if ff.y_state == Spectral
        return ff
    end

    make_spectral_y!(ff.data, ff.domain, ff.transforms)
    ff.y_state = Spectral
    return ff
end

"""
    make_physical_y!(ff)

Transform y direction from spectral to physical space.
Uses inverse DCT-I with proper normalization.
"""
function make_physical_y!(ff::FlowField{T}) where {T}
    if ff.y_state == Physical
        return ff
    end

    make_physical_y!(ff.data, ff.domain, ff.transforms)
    ff.y_state = Physical
    return ff
end

"""
    make_spectral!(ff)

Transform to fully spectral state (spectral in all directions).
Order matters: y first, then xz (following original C++ code).
"""
function make_spectral!(ff::FlowField)
    make_spectral_y!(ff)
    make_spectral_xz!(ff)
    return ff
end

"""
    make_physical!(ff)

Transform to fully physical state.
Order: xz first, then y.
"""
function make_physical!(ff::FlowField)
    make_physical_xz!(ff)
    make_physical_y!(ff)
    return ff
end

"""
    make_state!(ff, xz_state, y_state)

Transform to specified state in each direction.
"""
function make_state!(ff::FlowField, xz_state::FieldState, y_state::FieldState)
    # Handle xz direction
    if ff.xz_state != xz_state
        if xz_state == Physical
            make_physical_xz!(ff)
        else
            make_spectral_xz!(ff)
        end
    end

    # Handle y direction
    if ff.y_state != y_state
        if y_state == Physical
            make_physical_y!(ff)
        else
            make_spectral_y!(ff)
        end
    end

    return ff
end

# ===========================
# Arithmetic Operations
# ===========================

"""
    ff1 * scalar

Scalar multiplication (returns new FlowField).
"""
function Base.:*(ff::FlowField{T}, scalar::Real) where {T}
    result = FlowField(ff)
    result.data .*= scalar
    return result
end

function Base.:*(scalar::Real, ff::FlowField{T}) where {T}
    return ff * scalar
end

"""
    scale!(ff, scalar)

In-place scalar multiplication.
"""
function scale!(ff::FlowField{T}, scalar::Real) where {T}
    ff.data .*= scalar
    return ff
end

function scale!(ff::FlowField{T}, scalar::Complex) where {T}
    @assert ff.xz_state == Spectral "Complex scaling requires spectral xz state"
    # Apply complex scaling to packed spectral coefficients
    # This is more complex since we need to handle the packed format properly
    # For simplicity, we'll error for now
    error("Complex scaling not yet implemented for packed storage")
end

"""
    ff1 + ff2

FlowField addition (returns new FlowField).
"""
function Base.:+(ff1::FlowField{T}, ff2::FlowField{T}) where {T}
    @assert congruent(ff1, ff2) "FlowFields must be congruent"
    @assert ff1.xz_state == ff2.xz_state && ff1.y_state == ff2.y_state "FlowFields must be in same state"

    result = FlowField(ff1)
    result.data .+= ff2.data
    return result
end

"""
    ff1 - ff2

FlowField subtraction (returns new FlowField).
"""
function Base.:-(ff1::FlowField{T}, ff2::FlowField{T}) where {T}
    @assert congruent(ff1, ff2) "FlowFields must be congruent"
    @assert ff1.xz_state == ff2.xz_state && ff1.y_state == ff2.y_state "FlowFields must be in same state"

    result = FlowField(ff1)
    result.data .-= ff2.data
    return result
end

"""
    add!(ff1, ff2)

In-place addition: ff1 += ff2
"""
function add!(ff1::FlowField{T}, ff2::FlowField{T}) where {T}
    @assert congruent(ff1, ff2) "FlowFields must be congruent"
    @assert ff1.xz_state == ff2.xz_state && ff1.y_state == ff2.y_state "FlowFields must be in same state"

    ff1.data .+= ff2.data
    return ff1
end

"""
    subtract!(ff1, ff2)

In-place subtraction: ff1 -= ff2
"""
function subtract!(ff1::FlowField{T}, ff2::FlowField{T}) where {T}
    @assert congruent(ff1, ff2) "FlowFields must be congruent"
    @assert ff1.xz_state == ff2.xz_state && ff1.y_state == ff2.y_state "FlowFields must be in same state"

    ff1.data .-= ff2.data
    return ff1
end

"""
    add!(ff, a, ff1)

In-place scaled addition: ff += a*ff1
"""
function add!(ff::FlowField{T}, a::Real, ff1::FlowField{T}) where {T}
    @assert congruent(ff, ff1) "FlowFields must be congruent"
    @assert ff.xz_state == ff1.xz_state && ff.y_state == ff1.y_state "FlowFields must be in same state"

    ff.data .+= a .* ff1.data
    return ff
end

"""
    add!(ff, a, ff1, b, ff2)

In-place linear combination: ff = a*ff1 + b*ff2
"""
function add!(ff::FlowField{T}, a::Real, ff1::FlowField{T}, b::Real, ff2::FlowField{T}) where {T}
    @assert congruent(ff, ff1) && congruent(ff, ff2) "All FlowFields must be congruent"
    @assert ff1.xz_state == ff2.xz_state && ff1.y_state == ff2.y_state "Input FlowFields must be in same state"

    ff.data .= a .* ff1.data .+ b .* ff2.data
    ff.xz_state = ff1.xz_state
    ff.y_state = ff1.y_state
    return ff
end

# ===========================
# Utility Methods
# ===========================

"""
    set_to_zero!(ff)

Set all field values to zero.
"""
function set_to_zero!(ff::FlowField)
    fill!(ff.data, 0)
    return ff
end

"""
    swap!(ff1, ff2)

Efficiently swap data between two congruent FlowFields.
"""
function swap!(ff1::FlowField{T}, ff2::FlowField{T}) where {T}
    @assert congruent(ff1, ff2) "FlowFields must be congruent for swapping"

    # Swap data arrays
    ff1.data, ff2.data = ff2.data, ff1.data

    # Swap states
    ff1.xz_state, ff2.xz_state = ff2.xz_state, ff1.xz_state
    ff1.y_state, ff2.y_state = ff2.y_state, ff1.y_state
    ff1.padded, ff2.padded = ff2.padded, ff1.padded

    return nothing
end

"""
    zero_padded_modes!(ff)

Set aliased (high-frequency) modes to zero for dealiasing.
Requires FlowField to be in spectral state.
"""
function zero_padded_modes!(ff::FlowField)
    @assert ff.xz_state == Spectral "Must be in spectral state to zero padded modes"

    # Zero out modes beyond the 2/3 dealiasing limit
    for i in 1:ff.domain.num_dimensions
        for my in 1:ff.domain.My
            for mx in 1:ff.domain.Mx
                kx = mx_to_kx(ff, mx)
                for mz in 1:ff.domain.Mz
                    kz = mz_to_kz(ff, mz)

                    if is_aliased(ff, kx, kz)
                        set_cmplx!(ff, Complex{eltype(ff.data)}(0), mx, my, mz, i)
                    end
                end
            end
        end
    end

    ff.padded = true
    return ff
end

# ===========================
# Geometry Manipulation
# ===========================

"""
    resize!(ff, Nx, Ny, Nz, Nd, Lx, Lz, a, b)

Resize FlowField to new dimensions and domain.
All data is lost and field is reset to zero.
"""
function resize!(ff::FlowField{T}, Nx::Int, Ny::Int, Nz::Int, Nd::Int,
    Lx::T, Lz::T, a::T, b::T) where {T}

    # Create new domain
    new_domain = FlowFieldDomain(Nx, Ny, Nz, Nd, Lx, Lz, a, b)

    # Check if resize is actually needed
    if ff.domain == new_domain
        return ff
    end

    # Update domain
    ff.domain = new_domain

    # Reallocate data array
    ff.data = zeros(T, new_domain.Nx, new_domain.Ny, new_domain.Nzpad, new_domain.num_dimensions)

    # Recreate FFTW plans
    ff.transforms = FlowFieldTransforms(new_domain)

    # Reset to default state
    ff.xz_state = Physical
    ff.y_state = Physical
    ff.padded = false

    return ff
end

"""
    rescale!(ff, Lx, Lz)

Change domain lengths without changing grid resolution.
Field values are unchanged but represent a rescaled physical domain.
"""
function rescale!(ff::FlowField{T}, Lx::T, Lz::T) where {T}
    new_domain = FlowFieldDomain(ff.domain.Nx, ff.domain.Ny, ff.domain.Nz, ff.domain.num_dimensions,
        Lx, Lz, ff.domain.a, ff.domain.b)
    ff.domain = new_domain
    return ff
end

end