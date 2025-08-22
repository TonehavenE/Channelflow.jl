#=
Utility methods for the FlowField structure. Should be included after types_and_constructors.jl.
=#

# ===========================
# Utility Methods
# ===========================
import ..ChebyCoeffs: set_to_zero!
export swap!, zero_padded_modes!

"""
    set_to_zero!(ff)

Set all field values to zero.
"""
function set_to_zero!(ff::FlowField)
    if ff.physical_data !== nothing
        fill!(ff.physical_data, 0)
    end
    if ff.spectral_data !== nothing
        fill!(ff.spectral_data, 0)
    end
    return ff
end

"""
    swap!(ff1, ff2)

Efficiently swap data between two congruent FlowFields.
"""
function swap!(ff1::FlowField{T}, ff2::FlowField{T}) where {T}
    @assert congruent(ff1, ff2) "FlowFields must be congruent for swapping"

    # Swap data arrays
    ff1.physical_data, ff2.physical_data = ff2.physical_data, ff1.physical_data
    ff1.spectral_data, ff2.spectral_data = ff2.spectral_data, ff1.spectral_data

    # Swap states
    ff1.xz_state, ff2.xz_state = ff2.xz_state, ff1.xz_state
    ff1.y_state, ff2.y_state = ff2.y_state, ff1.y_state
    ff1.padded, ff2.padded = ff2.padded, ff1.padded
end

"""
    zero_padded_modes!(ff)

Set aliased (high-frequency) modes to zero for dealiasing.
Requires FlowField to be in spectral state.
"""
function zero_padded_modes!(ff::FlowField{T}) where {T<:Number}
    @assert ff.xz_state == Spectral "Must be in spectral state to zero padded modes"
    @assert ff.spectral_data !== nothing "Spectral data must be allocated"

    # Zero out modes beyond the 2/3 dealiasing limit
    for i = 1:ff.domain.num_dimensions
        for my = 1:ff.domain.My
            for mx = 1:ff.domain.Mx
                kx = mx_to_kx(ff, mx)
                for mz = 1:ff.domain.Mz
                    kz = mz_to_kz(ff, mz)

                    if is_aliased(ff, kx, kz)
                        set_cmplx!(ff, ComplexF64(0), mx, my, mz, i)
                    end
                end
            end
        end
    end

    ff.padded = true
end
