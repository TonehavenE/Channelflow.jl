module FlowFields

using ..Collocation

export apply_derivative, FlowField, compute_vorticity, compute_rotational_rhs, compute_skew_symmetric_rhs, nonlinear_rhs

struct FlowField
    velocity::AbstractVector
    grids::Vector{AbstractGrid}
end

function apply_derivative(u::Array, D::Matrix, dim::Int)
    # Applies derivative matrix D along dimension `dim`
    axes = ntuple(i -> Colon(), ndims(u))
    perm = (dim, setdiff(1:ndims(u), dim)...)
    u_perm = permutedims(u, perm)
    u_reshaped = reshape(u_perm, size(u, dim), :)
    du = D * u_reshaped
    du = reshape(du, size(u_perm))
    return permutedims(du, invperm(perm))
end

"Compute ∇ × u (vorticity) assuming a 2D or 3D flow field"
function compute_vorticity(flow::FlowField)
    u = flow.velocity
    grids = flow.grids
    nd = length(u)

    if nd == 2
        ∂v_∂x = get_derivative_matrix(grids[1], 1) * u[2]
        ∂u_∂y = get_derivative_matrix(grids[2], 1) * u[1]
        return ∂v_∂x .- ∂u_∂y  # scalar vorticity
    elseif nd == 3
        Dx, Dy, Dz = get_derivative_matrix.(grids, 1)
        return (
            Dy * u[3] .- Dz * u[2],
            Dz * u[1] .- Dx * u[3],
            Dx * u[2] .- Dy * u[1],
        )
    else
        throw(ArgumentError("Vorticity not implemented for dimension $nd"))
    end
end

"Compute RHS using rotational form: u × (∇ × u)"
function compute_rotational_rhs(flow::FlowField)
    u = flow.velocity
    ζ = compute_vorticity(flow)
    nd = length(u)

    if nd == 2
        # u = (u, v), ζ = scalar
        return (
            -u[2] .* ζ,  # -v * ζ
            u[1] .* ζ,  #  u * ζ
        )
    elseif nd == 3
        # u, ζ = 3-component vectors
        ux, uy, uz = u
        ζx, ζy, ζz = ζ
        return (
            uy .* ζz .- uz .* ζy,
            uz .* ζx .- ux .* ζz,
            ux .* ζy .- uy .* ζx,
        )
    else
        throw(ArgumentError("Rotational form not implemented for dimension $nd"))
    end
end

"Compute RHS using skew-symmetric form: u ⋅ ∇u + (∇ ⋅ u)u"
function compute_skew_symmetric_rhs(flow::FlowField)
    u = flow.velocity
    grids = flow.grids
    nd = length(u)

    rhs = similar(u)
    for i in 1:nd
        rhs_i = zero(u[i])
        for j in 1:nd
            ∂uj_∂xi = get_derivative_matrix(grids[i], 1) * u[j]
            rhs_i .+= u[j] .* ∂uj_∂xi
        end
        rhs[i] = rhs_i
    end
    return rhs
end


function nonlinear_rhs(flow::FlowField; form=:rotational)
    @assert length(flow.grids) == length(flow.velocity)
    @assert form in (:rotational, :skew_symmetric) "form must be :rotational or :skew_symmetric"
    if form == :rotational
        return compute_rotational_rhs(flow)
    elseif form == :skew_symmetric
        return compute_skew_symmetric_rhs(flow)
    end
end

end


