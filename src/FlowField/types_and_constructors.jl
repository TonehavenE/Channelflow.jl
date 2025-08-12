#=
Defines the FlowField structure and critical data operations. Should be included after FlowFieldDomain.jl and FlowFieldTransforms.jl.
Should be included before any other FlowField-related files (since it defines the type).
=#

export FlowField, _ensure_data_allocated!, _current_data

"""
FlowField stores 3D vector fields with Fourier x Chebyshev x Fourier spectral expansions.
"""
mutable struct FlowField{T<:Real}
    domain::FlowFieldDomain{T}
    xz_state::FieldState
    y_state::FieldState
    padded::Bool  # Flag for dealiasing

    # Separate storage for physical and spectral data
    physical_data::Union{Array{T,4},Nothing}          # [Nx, Ny, Nz, num_dimensions]
    spectral_data::Union{Array{Complex{T},4},Nothing}  # [Mx, My, Mz, num_dimensions]

    transforms::FlowFieldTransforms{T}
end

# ===========================
# Constructors
# ===========================


"""
    FlowField(Nx, Ny, Nz, tensor_shape, Lx, Lz, a, b; kwargs...)

Creates a tensor-valued FlowField.
"""
function FlowField(
    Nx::Int, Ny::Int, Nz::Int,
    tensor_shape::TensorShape,
    Lx::T, Lz::T, a::T, b::T;
    padded::Bool=false,
    xz_state::FieldState=Spectral,
    y_state::FieldState=Spectral,
) where {T<:Real}

    # Create domain with tensor shape
    domain = FlowFieldDomain(Nx, Ny, Nz, tensor_shape, Lx, Lz, a, b)

    # Allocate data based on state
    physical_data = nothing
    spectral_data = nothing

    if xz_state == Physical
        physical_data = zeros(T, domain.Nx, domain.Ny, domain.Nz, domain.num_dimensions)
    else
        spectral_data = zeros(Complex{T}, domain.Mx, domain.My, domain.Mz, domain.num_dimensions)
    end

    transforms = FlowFieldTransforms(domain)

    return FlowField{T}(
        domain, xz_state, y_state, padded,
        physical_data, spectral_data, transforms
    )
end

"""
    FlowField(Nx, Ny, Nz, num_dimensions, Lx, Lz, a, b; kwargs...)

Create a FlowField with specified grid and domain parameters.
"""
function FlowField(
    Nx::Int,
    Ny::Int,
    Nz::Int,
    num_dimensions::Int,
    Lx::T,
    Lz::T,
    a::T,
    b::T;
    padded::Bool=false,
    xz_state::FieldState=Spectral,
    y_state::FieldState=Spectral,
) where {T<:Real}
    tensor_shape = TensorShape((num_dimensions,))
    return FlowField(Nx, Ny, Nz, tensor_shape, Lx, Lz, a, b;
        padded=padded, xz_state=xz_state, y_state=y_state)
end
"""
    FlowField(domain; kwargs...)

Create FlowField from existing domain object.
"""
function FlowField(
    domain::FlowFieldDomain{T};
    padded::Bool=false,
    xz_state::FieldState=Spectral,
    y_state::FieldState=Spectral,
) where {T}

    physical_data = nothing
    spectral_data = nothing

    if xz_state == Physical
        physical_data = zeros(T, domain.Nx, domain.Ny, domain.Nz, domain.num_dimensions)
    else
        spectral_data =
            zeros(Complex{T}, domain.Mx, domain.My, domain.Mz, domain.num_dimensions)
    end

    transforms = FlowFieldTransforms(domain)

    return FlowField{T}(
        domain,
        xz_state,
        y_state,
        padded,
        physical_data,
        spectral_data,
        transforms,
    )
end

"""
    FlowField(other::FlowField)

Copy constructor - creates deep copy of data but shares FFTW plans.
"""
function FlowField(other::FlowField{T}) where {T}
    physical_data = other.physical_data === nothing ? nothing : copy(other.physical_data)
    spectral_data = other.spectral_data === nothing ? nothing : copy(other.spectral_data)

    return FlowField{T}(
        other.domain,
        other.xz_state,
        other.y_state,
        other.padded,
        physical_data,
        spectral_data,
        other.transforms,
    )
end

# ===========================
# Internal Data Management
# ===========================

"""
Ensure appropriate data array exists for current state.
"""
function _ensure_data_allocated!(ff::FlowField{T}) where {T}
    if ff.xz_state == Physical && ff.physical_data === nothing
        ff.physical_data =
            zeros(T, ff.domain.Nx, ff.domain.Ny, ff.domain.Nz, ff.domain.num_dimensions)
    elseif ff.xz_state == Spectral && ff.spectral_data === nothing
        ff.spectral_data = zeros(
            Complex{T},
            ff.domain.Mx,
            ff.domain.My,
            ff.domain.Mz,
            ff.domain.num_dimensions,
        )
    end
end

"""
Get reference to current active data array.
"""
function _current_data(ff::FlowField)
    if ff.xz_state == Physical
        return ff.physical_data
    else
        return ff.spectral_data
    end
end
