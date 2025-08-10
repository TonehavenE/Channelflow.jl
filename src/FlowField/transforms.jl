#=
Defines transform methods for FlowField structures. Should be included after types_and_constructors.jl and FlowFieldTransforms.jl.
=#

export make_physical!,
    make_spectral!,
    make_state!,
    make_physical_xz!,
    make_spectral_xz!,
    make_physical_y!,
    make_spectral_y!

# ===========================
# Transform Methods
# ===========================

"""
    make_spectral_xz!(ff)

Transform x,z directions from physical to spectral (Fourier) space.
Uses FFTW transforms with proper normalization.
"""
function make_spectral_xz!(ff::FlowField{T}) where {T}
    if ff.xz_state == Spectral
        return ff # already in spectral
    end

    @assert ff.physical_data !== nothing "Physical data must be allocated to convert to spectral"

    # Allocate spectral data if needed
    if ff.spectral_data === nothing
        ff.spectral_data = zeros(
            Complex{T},
            ff.domain.Mx,
            ff.domain.My,
            ff.domain.Mz,
            ff.domain.num_dimensions,
        )
    end

    make_spectral_xz!(ff.physical_data, ff.spectral_data, ff.domain, ff.transforms) # delegates to FlowFieldTransforms.jl
    ff.xz_state = Spectral

    return ff
end

"""
    make_physical_xz!(ff)

Transform x,z directions from spectral to physical space.
Uses inverse FFTW transforms.
"""
function make_physical_xz!(ff::FlowField{T}) where {T}
    if ff.xz_state == Physical
        return ff # already in physical
    end

    @assert ff.spectral_data !== nothing "Spectral data must be allocated to convert to physical"

    # Allocate physical data if needed
    if ff.physical_data === nothing
        ff.physical_data =
            zeros(T, ff.domain.Nx, ff.domain.Ny, ff.domain.Nz, ff.domain.num_dimensions)
    end

    make_physical_xz!(ff.spectral_data, ff.physical_data, ff.domain, ff.transforms)
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
        return ff # already in spectral
    end

    _ensure_data_allocated!(ff)
    current_data = _current_data(ff)
    make_spectral_y!(current_data, ff.domain, ff.transforms)
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
        return ff # already in physical
    end

    _ensure_data_allocated!(ff)
    current_data = _current_data(ff)
    make_physical_y!(current_data, ff.domain, ff.transforms)
    ff.y_state = Physical

    return ff
end

"""
    make_spectral!(ff)

Transform to fully spectral state (spectral in all directions).
Order matters: y first, then xz (following original C++ code).
"""
function make_spectral!(ff::FlowField)
    make_spectral_xz!(ff)
    make_spectral_y!(ff)
    return ff
end

"""
    make_physical!(ff)

Transform to fully physical state.
Order: xz first, then y.
"""
function make_physical!(ff::FlowField)
    make_physical_y!(ff)
    make_physical_xz!(ff)
    return ff
end

"""
    make_state!(ff, xz_state, y_state)

Transform to specified state in each direction.
"""
function make_state!(ff::FlowField, target_xz_state::FieldState, target_y_state::FieldState)
    # Handle xz direction
    if ff.xz_state != target_xz_state
        if target_xz_state == Physical
            make_physical_xz!(ff)
        else
            make_spectral_xz!(ff)
        end
    end

    # Handle y direction
    if ff.y_state != target_y_state
        if target_y_state == Physical
            make_physical_y!(ff)
        else
            make_spectral_y!(ff)
        end
    end

    return ff
end
