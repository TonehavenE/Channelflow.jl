#=
Defines operations for manipulating the geometry of FlowField structures.
Also defines the congruence operations. Should be included after types_and_constructors.jl.
=#

export geom_congruent, congruent, resize!, rescale!

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
# Geometry Manipulation
# ===========================

"""
    resize!(ff, Nx, Ny, Nz, Nd, Lx, Lz, a, b)

Resize FlowField to new dimensions and domain.
All data is lost and field is reset to zero.
"""
function Base.resize!(
    ff::FlowField{T},
    Nx::Int,
    Ny::Int,
    Nz::Int,
    Nd::Int,
    Lx::T,
    Lz::T,
    a::T,
    b::T,
) where {T}

    # Create new domain
    new_domain = FlowFieldDomain(Nx, Ny, Nz, Nd, Lx, Lz, a, b)

    # Check if resize is actually needed
    if ff.domain == new_domain
        return ff
    end

    # Update domain
    ff.domain = new_domain

    # Reallocate data arrays
    ff.physical_data = nothing
    ff.spectral_data = nothing

    # Recreate FFTW plans
    ff.transforms = FlowFieldTransforms(new_domain)

    # Reset to default state
    ff.xz_state = Physical
    ff.y_state = Physical
    ff.padded = false

    # Allocate initial data
    _ensure_data_allocated!(ff)

    return ff
end

"""
    rescale!(ff, Lx, Lz)

Change domain lengths without changing grid resolution.
Field values are unchanged but represent a rescaled physical domain.
"""
function rescale!(ff::FlowField{T}, Lx::T, Lz::T) where {T}
    new_domain = FlowFieldDomain(
        ff.domain.Nx,
        ff.domain.Ny,
        ff.domain.Nz,
        ff.domain.num_dimensions,
        Lx,
        Lz,
        ff.domain.a,
        ff.domain.b,
    )
    ff.domain = new_domain
    return ff
end
