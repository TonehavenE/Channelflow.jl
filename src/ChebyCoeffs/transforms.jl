#=
Defines transform operations for ChebyCoeff. 
Must be included after types_and_constructors.jl.
=#

export makeSpectral!,
    makePhysical!, makeState!, setToZero!, setState!, chebyfft!, ichebyfft!

"""Transform from Physical to Spectral state using provided transform."""
function chebyfft!(u::ChebyCoeff{T}, t::ChebyTransform) where {T<:Number}
    @assert u.state == Physical "Must be in Physical state"
    @assert length(t) == length(u.data) "Transform size must match data size"

    if length(u.data) < 2
        u.state = Spectral
        return
    end

    if T <: Real
        # Real case: direct cosine transform
        u.data .= t.cos_plan * u.data
    else
        # Complex case: transform real and imaginary parts separately
        real_part = real.(u.data)
        imag_part = imag.(u.data)

        real_part .= t.cos_plan * real_part
        imag_part .= t.cos_plan * imag_part

        u.data .= complex.(real_part, imag_part)
    end

    # Apply normalization factors
    N = length(u.data)
    nrm = 1.0 / (N - 1)
    u.data[1] *= 0.5 * nrm
    u.data[2:end-1] .*= nrm
    u.data[end] *= 0.5 * nrm

    u.state = Spectral
end

"""Transform from Spectral to Physical state using provided transform."""
function ichebyfft!(u::ChebyCoeff{T}, t::ChebyTransform) where {T<:Number}
    @assert u.state == Spectral "Must be in Spectral state"
    @assert length(t) == length(u.data) "Transform size must match data size"

    if length(u.data) < 2
        u.state = Physical
        return
    end

    # Undo normalization factors
    u.data[1] *= 2.0
    u.data[end] *= 2.0

    if T <: Real
        # Real case: direct inverse cosine transform
        tmp = t.cos_plan * u.data
        u.data .= tmp
        u.data .*= 0.5
    else
        # Complex case: transform real and imaginary parts separately
        real_part = real.(u.data)
        imag_part = imag.(u.data)

        real_part .= t.cos_plan * real_part
        imag_part .= t.cos_plan * imag_part

        u.data .= complex.(real_part .* 0.5, imag_part .* 0.5)
    end

    u.state = Physical
end

# Convenience methods that create temporary transforms
function makeSpectral!(u::ChebyCoeff{T}) where {T<:Number}
    if u.state == Physical
        t = ChebyTransform(length(u.data))
        chebyfft!(u, t)
    end
end

function makePhysical!(u::ChebyCoeff{T}) where {T<:Number}
    if u.state == Spectral
        t = ChebyTransform(length(u.data))
        ichebyfft!(u, t)
    end
end

function makeState!(u::ChebyCoeff{T}, s::FieldState) where {T<:Number}
    if u.state != s
        if u.state == Physical
            makeSpectral!(u)
        else
            makePhysical!(u)
        end
    end
end
