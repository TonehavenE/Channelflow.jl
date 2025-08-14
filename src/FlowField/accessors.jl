#=
Defines accessor methods for FlowField structures. Should be included after types_and_constructors.jl.
=#

import ..ChebyCoeffs: num_modes

export cmplx,
    set_cmplx!,
    num_x_gridpoints,
    num_y_gridpoints,
    num_z_gridpoints,
    num_gridpoints,
    num_x_modes,
    num_y_modes,
    num_z_modes,
    num_modes,
    vector_dim,
    num_dimensions,
    xz_state,
    y_state,
    Lx,
    Ly,
    Lz,
    domain_a,
    domain_b,
    nx_to_x,
    ny_to_y,
    nz_to_z,
    x_gridpoints,
    y_gridpoints,
    z_gridpoints,
    kx_max,
    kz_max,
    kx_min,
    kz_min,
    kx_to_mx,
    mx_to_kx,
    kz_to_mz,
    mz_to_kz,
    kx_max_dealiased,
    kz_max_dealiased,
    is_aliased,
    tensor_shape,
    tensor_rank,
    is_scalar_field,
    is_vector_field,
    is_matrix_field,
    is_symmetric_field

# ===========================
# Element Access Methods
# ===========================

"""
    ff[nx, ny, nz, indices...]

Access physical field values. Requires FlowField to be in Physical xz state.
For vector valued FlowFields: ff[nx, ny, nz, i]
For matrix valued FlowFields: ff[nx, ny, nz, i, j]
"""
function Base.getindex(ff::FlowField, nx::Int, ny::Int, nz::Int, indices...)
    @assert ff.xz_state == Physical "FlowField must be in Physical xz state for direct access"
    @assert ff.physical_data !== nothing "Physical data not allocated"
    @assert 1 <= nx <= ff.domain.Nx "nx out of bounds"
    @assert 1 <= ny <= ff.domain.Ny "ny out of bounds"
    @assert 1 <= nz <= ff.domain.Nz "nz out of bounds"
    component = tensor_index(ff.domain.tensor_shape, indices...)
    @assert 1 <= component <= ff.domain.num_dimensions "component index i out of bounds"
    return ff.physical_data[nx, ny, nz, component]
end

function Base.setindex!(ff::FlowField, val, nx::Int, ny::Int, nz::Int, indices...)
    @assert ff.xz_state == Physical "FlowField must be in Physical xz state for direct access"
    @assert 1 <= nx <= ff.domain.Nx "nx out of bounds"
    @assert 1 <= ny <= ff.domain.Ny "ny out of bounds"
    @assert 1 <= nz <= ff.domain.Nz "nz out of bounds"

    _ensure_data_allocated!(ff)
    component = tensor_index(ff.domain.tensor_shape, indices...)
    @assert 1 <= component <= ff.domain.num_dimensions "component index i out of bounds"

    ff.physical_data[nx, ny, nz, component] = val
end

"""
    cmplx(ff, mx, my, mz, i)

Access spectral coefficients. Requires FlowField to be in Spectral xz state.
Returns Complex{T} directly.
"""
function cmplx(ff::FlowField{T}, mx::Int, my::Int, mz::Int, indices...) where {T}
    @assert ff.xz_state == Spectral "FlowField must be in Spectral xz state for spectral access"
    @assert ff.spectral_data !== nothing "Spectral data not allocated"
    @assert 1 <= mx <= ff.domain.Mx "mx out of bounds"
    @assert 1 <= my <= ff.domain.My "my out of bounds"
    @assert 1 <= mz <= ff.domain.Mz "mz out of bounds"

    component = tensor_index(ff.domain.tensor_shape, indices...)
    @assert 1 <= component <= ff.domain.num_dimensions "component index i out of bounds"
    return ff.spectral_data[mx, my, mz, component]
end

"""
    set_cmplx!(ff, val, mx, my, mz, i)

Set spectral coefficient. Requires FlowField to be in Spectral xz state.
"""
function set_cmplx!(
    ff::FlowField{T},
    val::Complex{T},
    mx::Int,
    my::Int,
    mz::Int,
    indices...,
) where {T}
    @assert ff.xz_state == Spectral "FlowField must be in Spectral xz state for spectral access"
    _ensure_data_allocated!(ff)
    @assert 1 <= mx <= ff.domain.Mx "mx out of bounds"
    @assert 1 <= my <= ff.domain.My "my out of bounds"
    @assert 1 <= mz <= ff.domain.Mz "mz out of bounds"
    component = tensor_index(ff.domain.tensor_shape, indices...)
    @assert 1 <= component <= ff.domain.num_dimensions "component index i out of bounds"
    ff.spectral_data[mx, my, mz, component] = val
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
num_dimensions(ff::FlowField) = ff.domain.num_dimensions
xz_state(ff::FlowField) = ff.xz_state
y_state(ff::FlowField) = ff.y_state

# Domain properties
Lx(ff::FlowField) = ff.domain.Lx
Ly(ff::FlowField) = ff.domain.b - ff.domain.a
Lz(ff::FlowField) = ff.domain.Lz
domain_a(ff::FlowField) = ff.domain.a
domain_b(ff::FlowField) = ff.domain.b

# Coordinate functions (delegate to domain)
nx_to_x(ff::FlowField, nx::Int) = x_coord(ff.domain, nx)
ny_to_y(ff::FlowField, ny::Int) = y_coord(ff.domain, ny)
nz_to_z(ff::FlowField, nz::Int) = z_coord(ff.domain, nz)

x_gridpoints(ff::FlowField) = x_gridpoints(ff.domain)
y_gridpoints(ff::FlowField) = y_gridpoints(ff.domain)
z_gridpoints(ff::FlowField) = z_gridpoints(ff.domain)

# Wave number functions (delegate to domain)
kx_to_mx(ff::FlowField, kx::Int) = kx_to_mx(ff.domain, kx)
mx_to_kx(ff::FlowField, mx::Int) = mx_to_kx(ff.domain, mx)
kz_to_mz(ff::FlowField, kz::Int) = kz_to_mz(ff.domain, kz)
mz_to_kz(ff::FlowField, mz::Int) = mz_to_kz(ff.domain, mz)

# max and mins
kx_max(ff::FlowField) = div(ff.domain.Nx, 2)
kz_max(ff::FlowField) = div(ff.domain.Nz, 2)
kx_min(ff::FlowField) = -div(ff.domain.Nx, 2) + 1
kz_min(ff::FlowField) = 0

# Dealiasing
kx_max_dealiased(ff::FlowField) = kx_max_dealiased(ff.domain)
kz_max_dealiased(ff::FlowField) = kz_max_dealiased(ff.domain)
is_aliased(ff::FlowField, kx::Int, kz::Int) = is_aliased(ff.domain, kx, kz)

tensor_shape(ff::FlowField) = ff.domain.tensor_shape
tensor_rank(ff::FlowField) = length(ff.domain.tensor_shape.dims)
is_scalar_field(ff::FlowField) = ff.domain.tensor_shape == SCALAR_TENSOR
is_vector_field(ff::FlowField) = ff.domain.tensor_shape == VECTOR_TENSOR
is_matrix_field(ff::FlowField) = ff.domain.tensor_shape == MATRIX_TENSOR
is_symmetric_field(ff::FlowField) = ff.domain.tensor_shape == SYMMETRIC_TENSOR
