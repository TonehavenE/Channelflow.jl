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
