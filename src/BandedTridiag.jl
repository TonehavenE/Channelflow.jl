module BandedTridiags

using ..ChebyCoeffs

export BandedTridiag, UL_decompose!, UL_solve!, UL_solve_strided!, multiply_strided!, multiply!

mutable struct BandedTridiag{T<:Number}
    num_rows::Int
    first_row::Vector{T}
    lower::Vector{T}
    diag::Vector{T}
    inv_diag::Vector{T}
    upper::Vector{T}
    is_decomposed::Bool
end

function BandedTridiag(size::T) where {T<:Number}
    first_row = zeros(size)
    lower = zeros(size - 1)
    diag = zeros(size - 1)
    inv_diag = zeros(size - 1)
    upper = zeros(size - 1)
    BandedTridiag(size, first_row, lower, diag, inv_diag, upper, false)
end

Base.size(A::BandedTridiag) = (length(A.first_row), length(A.first_row)) # must be square

function Base.getindex(A::BandedTridiag, row::Int, col::Int)
    if row == 1
        return A.first_row[col]
    end

    shifted_row = row - 1 # diags start at row = 1, not row = 0

    if col == row - 1
        return shifted_row >= 1 ? A.lower[shifted_row] : zero(eltype(A.lower))
    elseif col == row
        return A.diag[shifted_row]
    elseif col == row + 1
        return shifted_row <= length(A.upper) ? A.upper[shifted_row] : zero(eltype(A.upper))
    else
        return zero(eltype(A.diag)) # off the diagonal
    end
end

function Base.setindex!(A::BandedTridiag{T}, val::T, i::Int, j::Int) where {T}
    if i == 1
        A.first_row[j] = val
    elseif j == i + 1
        A.upper[i] = val
    elseif i == j + 1
        A.lower[j] = val
    elseif i == j
        A.diag[i] = val
    else
        throw(ArgumentError("Cannot set value outside band in BandedTridiag"))
    end
end

function UL_decompose!(A::BandedTridiag)
    w = 0.0 # A_{k, k-1}
    Akk = 0.0 # A_{k, k}

    for k = A.num_rows-1:2
        Akk = A.diag[k]
        @assert Akk != 0.0

        w = A.lower[k]
        A.upper[k-1] /= Akk
        A.diag[k-1] -= w * A.upper[k-1]

        A.first_row[k-1] /= Akk
        A.first_row[k-1] -= w * A.first_row[k]
    end

    # Special Case for first row
    A.first_row[1] /= A.diag[2]
    A.first_row[1] -= A.lower[2] * A.first_row[2]

    # compute inverse diagonals
    A.inv_diag = 1.0 ./ A.diag

    A.is_decomposed = true
end

function UL_solve!(A::BandedTridiag{T}, b::AbstractVector{T}) where {T<:Number}
    @assert A.is_decomposed "Matrix must be UL-decomposed first"
    M = A.num_rows
    Mb = M - 1

    # I think all the indices are wrong.
    # 1. Solve U·y = b (backward substitution)
    for i = Mb-1:-1:2
        b[i+1] -= A.upper[i+1] * b[i+2]
    end
    println("\n\nafter solving upper: $b")

    for j = 2:M-1 # first row needs special care
        b[1] -= A.first_row[j+1] * b[j+1]
    end
    println("\n\nafter adjusting first row: $b")

    # 2. Solve L·x = y (forward substitution)
    b[1] /= A.diag[1]
    for i = 2:M-1
        b[i] -= A.lower[i] * b[i-1]
        b[i] *= A.inv_diag[i]
    end
    println("\n\nafter solving lower: $b")
end

function UL_solve!(A::BandedTridiag{T}, b::ChebyCoeff{T}) where {T<:Number}
    UL_solve!(A, b.data)
end

function UL_solve_strided!(
    A::BandedTridiag{T},
    b::AbstractVector{T},
    offset::Int,
    stride::Int,
) where {T<:Number}
    # view into B
    bv = @view b[offset+1:stride:end]
    UL_solve!(A, bv)
end
function UL_solve_strided!(
    A::BandedTridiag{T},
    b::ChebyCoeff{T},
    offset::Int,
    stride::Int,
) where {T<:Number}
    # view into B
    bv = @view b.data[offset+1:stride:end]
    UL_solve!(A, bv)
end

function multiply!(x::Vector{T}, A::BandedTridiag{T}, b::Vector{T}) where {T<:Number}
    M = A.num_rows
    Mbar = M - 1

    # row 0: dot product of first_row and x
    b[1] = A.first_row .* x[1:M]

    # rows 1 to Mbar - 1
    for i = 2:Mbar
        b[i] = A.lower[i] * x[i-1] + A.diag[i] * x[i] + A.upper[i] * x[i+1]
    end

    # final row
    b[M] = A.lower[M] * x[M-1] + A.diag[M] * x[M]
end


function multiply_strided!(
    x::Vector{T},
    A::BandedTridiag{T},
    b::Vector{T},
    offset::Int,
    stride::Int,
) where {T<:Number}
    @assert offset == 0 || offset == 1
    @assert stride == 1 || stride == 2

    M = A.num_rows
    Mbar = M - 1

    stride_index(stride_scale) = offset + stride * stride_scale + 1

    # row 0
    sum = zero(T)
    for j = 1:M
        sum += A.first_row[j] * x[offset+stride*(j-1)+1]
    end
    b[offset+1] = sum

    # rows 1 to Mbar - 1
    for i = 2:Mbar
        b[offset+stride*(i-1)+1] =
            A.lower[i] * x[stride_index(i - 2)] +
            A.diag[i] * x[stride_index(i - 1)] +
            A.upper[i] * x[stride_index(i)]
    end

    # final row CHECK THESE INDICES
    b[stride_index(M - 1)] =
        A.lower[Mbar] * x[stride_index(M - 2)] + A.diag[Mbar] * x[stride_index(M - 1)] # Are these indices right?
end

function multiply_strided!(x::ChebyCoeff, A::BandedTridiag{T}, b::ChebyCoeff, offset::Int, stride::Int) where {T<:Number}
    multiply_strided!(x.data, A, b.data, offset, stride)
end

end
