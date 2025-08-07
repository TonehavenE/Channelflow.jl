module BandedTridiagsFixed

using ..ChebyCoeffs

export BandedTridiag, UL_decompose!, UL_solve!, UL_solve_strided!, multiply_strided!, multiply!

mutable struct BandedTridiag{T<:Number}
    num_rows::Int
    a::Vector{T}
    inv_diag::Vector{T}
    is_decomposed::Bool
end

function BandedTridiag(size::T) where {T<:Number}
    inv_diag = zeros(size - 1)
    a = zeros(4 * size - 2)
    BandedTridiag(size, a, inv_diag, false)
end

Base.size(A::BandedTridiag) = (length(A.first_row), length(A.first_row)) # must be square

function get_band(A::BandedTridiag, j::Int)
    # C++: return a_[Mbar_ - j] where Mbar_ = M_ - 1
    # Julia 1-based index: (A.M - 1) - j + 1  =  A.M - j
    return A.a[A.num_rows-j]
end

function get_diag(A::BandedTridiag, i::Int)
    # C++: return d_[3 * i] where d_ = a_ + M_ - 1
    # d_ starts at Julia index M. So, the final index is M + 3*i
    d_start_index = A.num_rows
    return A.a[d_start_index+3*i]
end

function get_upper(A::BandedTridiag, i::Int)
    # C++: return d_[3 * i - 1]
    d_start_index = A.num_rows
    return A.a[d_start_index+3*i-1]
end

function get_lower(A::BandedTridiag, i::Int)
    # C++: return d_[3 * i + 1]
    d_start_index = A.num_rows
    return A.a[d_start_index+3*i+1]
end

function set_band!(A::BandedTridiag{T}, j::int, val{T}) where {T<:Number}
    A.a[A.num_rows-j] = val
end
function set_diag!(A::BandedTridiag{T}, i::int, val{T}) where {T<:Number}
    d_start_index = A.num_rows
    A.a[d_start_index+3*i] = val
end
function set_upper!(A::BandedTridiag{T}, i::int, val{T}) where {T<:Number}
    d_start_index = A.num_rows
    A.a[d_start_index+3*i - 1] = val
end
function set_lower!(A::BandedTridiag{T}, i::int, val{T}) where {T<:Number}
    d_start_index = A.num_rows
    A.a[d_start_index+3*i + 1] = val
end

function Base.getindex(A::BandedTridiag, i::Int, j::Int)
	@assert i == 0 || (i >= 0 && i < A.num_rows && j >= 0 && j < A.num_rows && abs(i - j) <= 1);
    if (i == 0)
        return get_band(A, j);
    else if (i == j)
        return get_diag(A, i);
    else if (i < j)
        return get_upper(A, i);
    else
        return get_lower(A, i);
	end
end

function Base.setindex!(A::BandedTridiag{T}, val::T, i::Int, j::Int) where {T}
    if i == 1
		set_band!(A, j, val)
    elseif j == i + 1
		set_upper!(A, i, val)
    elseif i == j + 1
		set_lower!(A, j, val)
    elseif i == j
		set_diag!(A, i, val)
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

		w = get_lower(A, k)
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

    # C++: for (i = 1; i < M; ++i) 
    for i = 2:M  # Julia 1-based: i from 2 to M
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
            b[i+2] = (b[i+2] - A.lower[i+1] * b[i+1]) * A.inv_diag[i+1]
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
            b[2i+1] = (b[2i+1] - A.lower[i+1] * b[2(i-1)+1]) * A.inv_diag[i+1]
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
            b[2i+2] = (b[2i+2] - A.lower[i+1] * b[2i]) * A.inv_diag[i+1]
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

end

