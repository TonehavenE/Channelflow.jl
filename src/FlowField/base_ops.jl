#=
Defines basic arithmetic operations for FlowField structures. Should be included after types_and_constructors.jl.
=#

export scale!, add!, subtract!, add!, set_to_zero!


# ===========================
# Arithmetic Operations
# ===========================

"""
    ff1 * scalar

Scalar multiplication (returns new FlowField).
"""
function Base.:*(ff::FlowField{T}, scalar::Number) where {T}
    result = FlowField(ff)
    current_data = _current_data(result)
    if current_data !== nothing
        current_data .*= scalar
    end
    return result
end

function Base.:*(scalar::Number, ff::FlowField{T}) where {T}
    return ff * scalar
end

"""
    scale!(ff, scalar)

In-place scalar multiplication.
"""
function scale!(ff::FlowField{T}, scalar::Number) where {T}
    current_data = _current_data(ff)
    if current_data !== nothing
        current_data .*= scalar
    end
    return ff
end

"""
    ff1 + ff2

FlowField addition (returns new FlowField).
"""
function Base.:+(ff1::FlowField{T}, ff2::FlowField{T}) where {T}
    @assert congruent(ff1, ff2) "FlowFields must be congruent"
    @assert ff1.xz_state == ff2.xz_state && ff1.y_state == ff2.y_state "FlowFields must be in same state"

    result = FlowField(ff1)
    data1 = _current_data(result)
    data2 = _current_data(ff2)

    if data1 !== nothing && data2 !== nothing
        data1 .+= data2
    end

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
    data1 = _current_data(result)
    data2 = _current_data(ff2)

    if data1 !== nothing && data2 !== nothing
        data1 .-= data2
    end

    return result
end

"""
    add!(ff1, ff2)

In-place addition: ff1 += ff2
"""
function add!(ff1::FlowField{T}, ff2::FlowField{T}) where {T}
    @assert congruent(ff1, ff2) "FlowFields must be congruent"
    @assert ff1.xz_state == ff2.xz_state && ff1.y_state == ff2.y_state "FlowFields must be in same state"

    data1 = _current_data(ff1)
    data2 = _current_data(ff2)

    if data1 !== nothing && data2 !== nothing
        data1 .+= data2
    end

    return ff1
end

"""
    subtract!(ff1, ff2)

In-place subtraction: ff1 -= ff2
"""
function subtract!(ff1::FlowField{T}, ff2::FlowField{T}) where {T}
    @assert congruent(ff1, ff2) "FlowFields must be congruent"
    @assert ff1.xz_state == ff2.xz_state && ff1.y_state == ff2.y_state "FlowFields must be in same state"

    data1 = _current_data(ff1)
    data2 = _current_data(ff2)

    if data1 !== nothing && data2 !== nothing
        data1 .-= data2
    end

    return ff1
end

"""
    add!(ff, a, ff1)

In-place scaled addition: ff += a*ff1
"""
function add!(ff::FlowField{T}, a::Number, ff1::FlowField{T}) where {T}
    @assert congruent(ff, ff1) "FlowFields must be congruent"
    @assert ff.xz_state == ff1.xz_state && ff.y_state == ff1.y_state "FlowFields must be in same state"

    data = _current_data(ff)
    data1 = _current_data(ff1)

    if data !== nothing && data1 !== nothing
        data .+= a .* data1
    end

    return ff
end

"""
    add!(ff, a, ff1, b, ff2)

In-place linear combination: ff = a*ff1 + b*ff2
"""
function add!(
    ff::FlowField{T},
    a::Number,
    ff1::FlowField{T},
    b::Number,
    ff2::FlowField{T},
) where {T}
    @assert congruent(ff, ff1) && congruent(ff, ff2) "All FlowFields must be congruent"
    @assert ff1.xz_state == ff2.xz_state && ff1.y_state == ff2.y_state "Input FlowFields must be in same state"

    # Update state to match inputs
    ff.xz_state = ff1.xz_state
    ff.y_state = ff1.y_state

    _ensure_data_allocated!(ff)
    data = _current_data(ff)
    data1 = _current_data(ff1)
    data2 = _current_data(ff2)

    if data !== nothing && data1 !== nothing && data2 !== nothing
        data .= a .* data1 .+ b .* data2
    end

    return ff
end
