module BandedTridiags

using ..ChebyCoeffs
using Printf

export BandedTridiag, UL_decompose!, UL_solve!, UL_solve_strided!, multiply_strided!, multiply!

mutable struct BandedTridiag{T<:Number}
    num_rows::Int
    first_row::Vector{T}
    lower::Vector{T}
    diag::Vector{T}
    upper::Vector{T}
    is_decomposed::Bool
end

function BandedTridiag(size::T) where {T<:Number}
    first_row = zeros(size)
    lower = zeros(size - 1)
    diag = zeros(size)
    upper = zeros(size - 1)
    BandedTridiag(size, first_row, lower, diag, upper, false)
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

function Base.setindex!(A::BandedTridiag{T}, val::T, row::Int, col::Int) where {T}
    if row == 1
        if col == 1
            A.diag[1] = val
        elseif col == 2
            A.upper[1] = val
        end
        A.first_row[col] = val
    elseif col == row + 1
        A.upper[row] = val
    elseif row == col + 1
        A.lower[col] = val
    elseif row == col
        A.diag[row] = val
    else
        throw(ArgumentError("Cannot set value outside band in BandedTridiag"))
    end
end

function UL_decompose!(A::BandedTridiag)
    w = 0.0 # A_{k, k-1}
    Akk = 0.0 # A_{k, k}

    for k = A.num_rows-1:2
        # Akk = A.diag[k]
        Akk = A[k, k]
        @assert Akk != 0.0

        # w = A.lower[k]
        # A.upper[k-1] /= Akk
        # A.diag[k-1] -= w * A.upper[k-1]
        w = A[k+1, k]
        A[k-1, k] /= Akk
        A[k-1, k-1] -= w * A[k-1, k]

        # A.first_row[k-1] /= Akk
        # A.first_row[k-1] -= w * A.first_row[k]
        A[1, k-1] /= Akk
        A[1, k-1] -= w * A[0, k]
    end

    # Special Case for first row
    A[1, 1] /= A[2, 2]
    A[1, 1] -= A[3, 2] * A[1, 2]
    # A.first_row[1] /= A.diag[2]
    # A.first_row[1] -= A.lower[2] * A.first_row[2]

    A.is_decomposed = true
end

function UL_solve!(A::BandedTridiag{T}, b::AbstractVector{T}) where {T<:Number}
    @assert A.is_decomposed "Matrix must be UL-decomposed first"
    M = A.num_rows

    # 1. Solve U·y = b by backsubstitution
    # C++: for (i = M-2; i > 0; --i)  // rows M-2 through 1
    for i = (M-1):-1:2  # Julia 1-based: rows (M-1) down to 2
        b[i] -= A.upper[i-1] * b[i+1]  # updiag(i-1) * b[i+1]
    end

    # Handle first row (C++ row 0)
    # C++: for (j = i + 1; j < M; ++j) where i=0 after loop, so j from 1 to M-1
    for j = 2:M  # Julia 1-based: j from 2 to M
        b[1] -= A.first_row[j] * b[j]  # band(j-1) * b[j]
    end

    # 2. Solve L·x = y by forward substitution
    # C++: b[0] /= diag(0)  -> diag(0) is stored as first_row[1] in Julia
    b[1] /= A.first_row[1]  # Diagonal element of first row

    # C++: for (i = 1; i < M; ++i) => [1, 2, 3, ..., M - 1] 
    for i = 2:M  # Julia 1-based: i = 2:M => [2, 3, 4, ... M]
        # C++: (b[i] -= lodiag(i) * b[i-1]) /= diag(i)
        b[i] -= A.lower[i-1] * b[i-1]    # lodiag(i-1) * b[i-1] 
        b[i] /= A.diag[i-1]              # diag(i-1) - diagonal for row i
    end

    return b
end

function UL_solve!(A::BandedTridiag{T}, b::ChebyCoeff{T}) where {T<:Number}
    UL_solve!(A, b.data)
end

function UL_solve_strided!(
    A::BandedTridiag{T},
    b::AbstractVector{T},
    offset::Int,
    stride::Int
) where {T<:Number}
    @assert A.is_decomposed "Matrix must be UL-decomposed first"
    @assert offset in [0, 1] "offset must be 0 or 1"
    @assert stride in [1, 2] "stride must be 1 or 2"

    M = A.num_rows
    Mb = M - 1

    if offset == 0 && stride == 1
        println("====\n\n\n====offset = 0, stride = 1\n====\n====")
        # Standard case
        UL_solve!(A, b)
        return b
    elseif offset == 1 && stride == 1
        println("====\n\n\n====offset = 1, stride = 1\n====\n====")
        # Backsubstitution
        for i in Mb-1:-1:1
            b[i+2] -= A.upper[i+1] * b[i+3]
        end
        for j in 1:Mb
            b[1+1] -= A.first_row[j+1] * b[j+2]
        end

        # Forward substitution
        b[2] /= A.diag[1]
        for i in 1:(M-1)
            b[i+2] = (b[i+2] - A.lower[i+1] * b[i+1]) / A.diag[i+1]
        end

    elseif offset == 0 && stride == 2
        println("====\n\n\n====offset = 0, stride = 2\n====\n====")
        # Backsubstitution
        for i in (Mb-1):-1:1
            b[2*i+1] -= A.upper[i+1] * b[2(i+1)+1]
        end
        for j in 1:(M-1)
            b[1] -= A.first_row[j+1] * b[2*j+1]
        end

        # Forward substitution
        b[1] /= A.diag[1]
        for i in 1:(M-1)
            # b[2i+1] = (b[2i+1] - A.lower[i+1] * b[2(i-1)+1]) / A.diag[i+1]
            b[2i+1] = (b[2i+1] - A[i+2, i+1])
        end

    elseif offset == 1 && stride == 2
        println("====\n\n\n====offset = 1, stride = 2\n====\n====")
        # Backsubstitution
        for i in (M-2):-1:1
            b[2i+2] -= A.upper[i+1] * b[2i+4]
        end
        for j in 1:(M-1)
            b[2] -= A.first_row[j+1] * b[2j+2]
        end

        # Forward substitution
        b[2] /= A.diag[1]
        for i in 1:(M-1)
            b[2i+2] = (b[2i+2] - A.lower[i+1] * b[2i]) / A.diag[i+1]
        end

    else
        error("offset must be 0 or 1, and stride must be 1 or 2")
    end

    return b
end

function UL_solve_strided!(
    A::BandedTridiag{T},
    b::ChebyCoeff{T},
    offset::Int,
    stride::Int,
) where {T<:Number}
    # view into B
    UL_solve_strided!(A, b.data, offset, stride)
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

# Pretty printing
function Base.show(io::IO, A::BandedTridiag{T}) where {T}
    print(io, "$(A.M)×$(A.M) BandedTridiag{$T}")
    if A.is_decomposed
        print(io, " (UL-decomposed)")
    end
end

function Base.display(A::BandedTridiag)
    println("$(A.M)×$(A.M) BandedTridiag:")
    for i = 1:A.M
        print("  ")
        for j = 1:A.M
            if abs(i - j) ≤ 1 || i == 1
                @printf "%f " A[i, j]
            else
                print(" 0 ")
            end
        end
        println()
    end
    if A.is_decomposed
        println("  (UL-decomposed)")
    end
end

end
