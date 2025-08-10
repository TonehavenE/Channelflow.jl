#=
Defines norm and inner product operations for FlowFields.
=#

using ..FlowFields
using ..ChebyCoeffs

raw"""
    boundary_condition_norm2(ff, normalize=true)

Evaluates the norm of the FlowField at the boundary:
```math
\int ||f||^2 dx dz \quad \text{at}~y = a, b (/ (Lx * Lz)~\text{if normalize})
```
"""
function boundary_condition_norm2(ff::FlowField, normalize::Bool=true)
    @assert xz_state(ff) == Spectral "FlowField must be in Spectral xz state for spectral access"
    bc2 = 0.0
    Mx, My, Mz = num_modes(ff)

    profile = ChebyCoeff{Complex}(My, domain_a(ff), domain_b(ff), y_state(ff))

    for i = 1:num_dimensions(ff), mx = 1:Mx, mz = 1:Mz
        for my = 1:My
            profile[my] = cmplx(ff, mx, my, mz, i)
        end

        bc2 += abs2(eval_a(profile))
        bc2 += abs2(eval_b(profile))
    end
    if !normalize
        bc2 *= Lx(ff) * Lz(ff)
    end

    return bc2
end

"""
    boundary_condition_norm(ff, normalize=true)

Returns sqrt(boundary_condition_norm2(ff, normalize=normalize)).
"""
function boundary_condition_norm(ff, normalize::Bool=true)
    return sqrt(boundary_condition_norm2(ff, normalize))
end

raw"""
    boundary_condition_distance2(ff, gg, normalize=true)

Calculates the distance between two FlowFields at the boundary:
```math
\int ||f - g||^2 dx dz \quad \text{at}~y = a, b (/ (Lx * Lz)~\text{if normalize})
```
"""
function boundary_condition_distance2(ff::FlowField, gg::FlowField, normalize::Bool=true)
    @assert congruent(ff, gg) "FlowFields must be congruent for distance calculation"
    @assert xz_state(ff) == Spectral && xz_state(gg) == Spectral "FlowFields must be in Spectral xz state for spectral access"

    bc2 = 0.0
    Mx, My, Mz = num_modes(ff)
    difference = ChebyCoeff{Complex}(My, domain_a(ff), domain_b(ff), y_state(ff))

    for i = 1:num_dimensions(ff), mx = 1:Mx, mz = 1:Mz
        for my = 1:My
            difference[my] = cmplx(ff, mx, my, mz, i) - cmplx(gg, mx, my, mz, i)
        end

        bc2 += abs2(eval_a(difference))
        bc2 += abs2(eval_b(difference))
    end

    if !normalize
        bc2 *= Lx(ff) * Lz(ff)
    end
    return bc2
end

"""
    boundary_condition_distance(ff, gg, normalize=true)

Returns sqrt(boundary_condition_distance2(ff, gg, normalize=normalize)).
"""
function boundary_condition_distance(ff::FlowField, gg::FlowField, normalize::Bool=true)
    return sqrt(boundary_condition_distance2(ff, gg, normalize))
end

raw"""
    L2Norm2(ff::FlowField, normalize::Bool=true)

Returns the L2 norm squared of a FlowField.

```math
\int ||f||^2 dx dy dz (/ (Lx * Lz)~\text{if normalize})
```
"""
function L2Norm2(ff::FlowField, normalize::Bool=true)
    @assert xz_state(ff) == Spectral "FlowField must be in Spectral xz state for spectral access"
    @assert y_state(ff) == Spectral "FlowField must be in Spectral y state for spectral access"

    sum = 0.0

    kxmin = ff.padded ? -kx_max_dealiased(ff) : kx_min(ff)
    kxmax = ff.padded ? kx_max_dealiased(ff) : kx_max(ff)
    kzmin = 0
    kzmax = ff.padded ? kz_max_dealiased(ff) : kz_max(ff)

    cz = 1
    for kz = kzmin:kzmax
        mz = kz_to_mz(ff, kz)
        for kx = kxmin:kxmax
            mx = kx_to_mx(ff, kx)
            for i = 1:num_dimensions(ff)
                profile = ChebyCoeff{Complex}(num_y_gridpoints(ff), domain_a(ff), domain_b(ff), Spectral)
                for ny = 1:num_y_gridpoints(ff)
                    profile[ny] = cmplx(ff, mx, ny, mz, i)
                end
                sum += cz * L2Norm2(profile, normalize)
            end
        end
        cz = 2
    end
    if !normalize
        sum *= Lx(ff) * Lz(ff)
    end

    return sum
end

"""
    L2Norm(ff, normalize)

Calculates the L2 norm of a flow field.

L2Norm(ff, normalize) = sqrt(L2Norm2(ff, normalize))
"""
function L2Norm(ff::FlowField, normalize::Bool=true)
    return sqrt(L2Norm2(ff, normalize))
end

"""
    L2Dist2(ff, gg, normalize)

Calculates the L2 distance squared between two FlowFields.
"""
function L2Dist2(ff::FlowField, gg::FlowField, normalize::Bool=true)
    @assert congruent(ff, gg) "FlowFields must be congruent for distance calculation"
    @assert xz_state(ff) == Spectral && xz_state(gg) == Spectral "FlowFields must be in Spectral xz state for spectral access"
    @assert y_state(ff) == Spectral && y_state(gg) == Spectral "FlowFields must be in Spectral y state for spectral access"

    sum = 0.0

    difference = ChebyCoeff{Complex}(num_y_modes(ff), domain_a(ff), domain_b(ff), Spectral)

    dealiased = ff.padded && gg.padded

    kxmin = dealiased ? -kx_max_dealiased(ff) : kx_min(ff)
    kxmax = dealiased ? kx_max_dealiased(ff) : kx_max(ff)
    kzmin = dealiased ? -kz_max_dealiased(ff) : kz_min(ff)
    kzmax = dealiased ? kz_max_dealiased(ff) : kz_max(ff)

    for i = 1:num_dimensions(ff), kx = kxmin:kxmax
        mx = kx_to_mx(ff, kx)
        cz = 1
        for kz = kzmin:kzmax
            mz = kz_to_mz(ff, kz)
            for my = 1:num_y_modes(ff)
                difference[my] = cmplx(ff, mx, my, mz, i) - cmplx(gg, mx, my, mz, i)
            end
            sum += cz * L2Norm2(difference, normalize)
            cz = 2
        end
    end
    if !normalize
        sum *= Lx(ff) * Lz(ff)
    end

    return sum
end

"""
    L2Dist(ff::FlowField, gg::FlowField, normalize::Bool=true)

Calculates the L2 distance between two FlowFields.
"""
function L2Dist(ff::FlowField, gg::FlowField, normalize::Bool=true)
    return sqrt(L2Dist2(ff, gg, normalize))
end