module BandedTridiags

using ..ChebyCoeffs
using Printf

export BandedTridiag, UL_decompose!, UL_solve!, UL_solve_strided!, multiply_strided!, multiply!, multiply, multiply_strided, extract_UL_matrices, to_dense
export set_main_diag!, set_first_row!, set_upper_diag!, set_lower_diag!, first_row, upper_diag, lower_diag, main_diag

"""
A banded tridiagonal matrix with the structure:
```
b b b b b  <- first row (band)
l d u 0 0  <- tridiagonal rows
0 l d u 0
0 0 l d u
0 0 0 l d  <- last row
```

Storage follows C++ channelflow convention with a single array.
"""
mutable struct BandedTridiag{T<:Number}
    num_rows::Int              # number of rows (square matrix)
    data::Vector{T}     # single array for all elements
    d_offset::Int       # offset to diagonal elements in data array
    inv_diag::Vector{T}  # inverse diagonal elements for solving
    is_decomposed::Bool
end

# Constructor with type parameter
function BandedTridiag{T}(M::Int) where {T<:Number}
    @assert M >= 0 "Matrix size must be non-negative"
    data_size = max(4 * M - 2, 0)  # matches C++ allocation, handle M=0 case
    data = zeros(T, data_size)
    d_offset = M
    invdiag = zeros(T, M)

    BandedTridiag{T}(M, data, d_offset, invdiag, false)
end

# Convenience constructor - defaults to Float64
BandedTridiag(M::Int) = BandedTridiag{Float64}(M)

Base.size(A::BandedTridiag) = (A.num_rows, A.num_rows)

# Accessor methods matching C++ interface
"""Get/set band element A[0,j] (first row)"""
function first_row(A::BandedTridiag, j::Int)
    @boundscheck (1 ≤ j ≤ A.num_rows) || throw(BoundsError(A, (1, j)))
    num_diag_elements = A.num_rows - 1
    A.data[num_diag_elements-j+2]  # Convert to 1-based indexing
end

function set_first_row!(A::BandedTridiag{T}, j::Int, val::T) where {T}
    @boundscheck (1 ≤ j ≤ A.num_rows) || throw(BoundsError(A, (0, j)))
    num_diag_elements = A.num_rows - 1
    A.data[num_diag_elements-j+2] = val
    A.is_decomposed = false
end

"""Get/set diagonal element A[i,i]"""
function main_diag(A::BandedTridiag, i::Int)
    @boundscheck (1 ≤ i ≤ A.num_rows) || throw(BoundsError(A, (i, i)))
    A.data[A.d_offset+3*(i-1)]
end

function set_main_diag!(A::BandedTridiag{T}, i::Int, val::T) where {T}
    @boundscheck (1 ≤ i ≤ A.num_rows) || throw(BoundsError(A, (i, i)))
    A.data[A.d_offset+3*(i-1)] = val
    A.is_decomposed = false
end

"""Get/set upper diagonal element A[i,i+1]"""
function upper_diag(A::BandedTridiag, i::Int)
    @boundscheck (1 ≤ i ≤ A.num_rows) || throw(BoundsError(A, (i, i + 1)))
    A.data[A.d_offset+3*(i-1)-1]
end

function set_upper_diag!(A::BandedTridiag{T}, i::Int, val::T) where {T}
    @boundscheck (1 ≤ i ≤ A.num_rows) || throw(BoundsError(A, (i, i + 1)))
    A.data[A.d_offset+3*(i-1)-1] = val
    A.is_decomposed = false
end

"""Get/set lower diagonal element A[i,i-1]"""
function lower_diag(A::BandedTridiag, i::Int)
    @boundscheck (1 ≤ i ≤ A.num_rows) || throw(BoundsError(A, (i, i - 1)))
    A.data[A.d_offset+3*(i-1)+1]
end

function set_lower_diag!(A::BandedTridiag{T}, i::Int, val::T) where {T}
    @boundscheck (1 ≤ i ≤ A.num_rows) || throw(BoundsError(A, (i, i - 1)))
    A.data[A.d_offset+3*(i-1)+1] = val
    A.is_decomposed = false
end

# Matrix interface using 1-based indexing
function Base.getindex(A::BandedTridiag, i::Int, j::Int)
    @boundscheck (1 ≤ i ≤ A.num_rows && 1 ≤ j ≤ A.num_rows) || throw(BoundsError(A, (i, j)))

    if i == 1
        return first_row(A, j)
    elseif i == j
        return main_diag(A, i)
    elseif i == j + 1
        return lower_diag(A, i)
    elseif i == j - 1
        return upper_diag(A, i)
    else
        return zero(eltype(A.data))
    end
end

function Base.setindex!(A::BandedTridiag{T}, val::T, i::Int, j::Int) where {T}
    @boundscheck (1 ≤ i ≤ A.num_rows && 1 ≤ j ≤ A.num_rows) || throw(BoundsError(A, (i, j)))

    if i == 1
        set_first_row!(A, j, val)
    elseif i == j
        set_main_diag!(A, i, val)
    elseif i == j + 1
        set_lower_diag!(A, i, val)
    elseif i == j - 1
        set_upper_diag!(A, i, val)
    else
        abs(i - j) ≤ 1 || throw(ArgumentError("Cannot set element ($i,$j) outside the band structure"))
    end

    return val
end

"""UL decomposition (no pivoting) - matches C++ implementation exactly"""
function UL_decompose!(A::BandedTridiag{T}) where {T}
    @assert A.num_rows ≥ 2
    Mb = A.num_rows - 1

    # Main decomposition loop - matches C++ exactly
    for k = A.num_rows:-1:3  # C++: for (int k = Mb; k > 1; --k)
        Akk = main_diag(A, k)
        @assert Akk != 0.0 "Zero diagonal element encountered"

        w = lower_diag(A, k)  # A[k, k-1]

        # C++: diag(k-1) -= w * (upper_diag(k-1) /= Akk)
        updiag_val = upper_diag(A, k - 1) / Akk
        set_upper_diag!(A, k - 1, updiag_val)
        set_main_diag!(A, k - 1, main_diag(A, k - 1) - w * updiag_val)

        # C++: band(k-1) -= w * (band(k) /= Akk)
        band_val = first_row(A, k) / Akk
        set_first_row!(A, k, band_val)
        set_first_row!(A, k - 1, first_row(A, k - 1) - w * band_val)
    end

    # Handle first row - C++: band(0) -= lodiag(1) * (band(1) /= diag(1))
    band_1_val = first_row(A, 2) / main_diag(A, 2)  # band(1) in C++ is band(2) in Julia
    set_first_row!(A, 2, band_1_val)
    set_first_row!(A, 1, first_row(A, 1) - lower_diag(A, 2) * band_1_val)

    # Compute inverse diagonal elements
    for i = 1:A.num_rows
        A.inv_diag[i] = 1.0 / main_diag(A, i)
    end

    A.is_decomposed = true
    return A
end

"""UL solve with strided access - matches C++ implementation exactly"""
function UL_solve_strided!(A::BandedTridiag{T}, b::AbstractVector{T},
    offset::Int, stride::Int) where {T<:Number}
    @assert A.is_decomposed "Matrix must be UL-decomposed first"
    @assert offset in [0, 1] "offset must be 0 or 1"
    @assert stride in [1, 2] "stride must be 1 or 2"

    Mb = A.num_rows - 1

    # Convert to 1-based indexing for array access
    offset_1based = offset + 1

    if offset == 0 && stride == 1
        # Standard case - matches C++ exactly

        # Solve Uy=b by backsubstitution
        # C++: for (i = Mb - 1; i > 0; --i) b[i] -= upper_diag(i) * b[i + 1];
        for i = Mb:-1:2  # Julia: Mb down to 2 (C++: Mb-1 down to 1)
            b[i] -= upper_diag(A, i) * b[i+1]
        end

        # C++: for (j = i + 1; j < M_; ++j) b[0] -= band(j) * b[j];
        # After the loop, i = 0 in C++, so j goes from 1 to M-1
        for j = 2:A.num_rows  # Julia: 2 to M (C++: 1 to M-1)
            b[1] -= first_row(A, j) * b[j]
        end

        # Solve Lx=y by forward substitution
        # C++: b[0] /= diag(0)
        b[1] /= main_diag(A, 1)

        # C++: for (i = 1; i < M_; ++i) (b[i] -= lodiag(i) * b[i - 1]) /= diag(i);
        for i = 2:A.num_rows  # Julia: 2 to M (C++: 1 to M-1)
            b[i] = (b[i] - lower_diag(A, i) * b[i-1]) * A.inv_diag[i]
        end

    elseif offset == 1 && stride == 1
        # C++ implementation for offset=1, stride=1
        for i = Mb:-1:2
            b[offset_1based+i] -= upper_diag(A, i) * b[offset_1based+i+1]
        end
        for j = 2:A.num_rows
            b[offset_1based] -= first_row(A, j) * b[offset_1based+j-1]
        end
        b[offset_1based] /= main_diag(A, 1)
        for i = 2:A.num_rows
            b[offset_1based+i-1] = (b[offset_1based+i-1] - lower_diag(A, i) * b[offset_1based+i-2]) * A.inv_diag[i]
        end

    elseif offset == 0 && stride == 2
        # C++ implementation for offset=0, stride=2
        for i = Mb:-1:2
            b[stride*i-1] -= upper_diag(A, i) * b[stride*(i+1)-1]
        end
        for j = 2:A.num_rows
            b[1] -= first_row(A, j) * b[stride*j-1]
        end
        b[1] /= main_diag(A, 1)
        for i = 2:A.num_rows
            b[stride*i-1] = (b[stride*i-1] - lower_diag(A, i) * b[stride*(i-1)-1]) * A.inv_diag[i]
        end

    elseif offset == 1 && stride == 2
        # C++ implementation for offset=1, stride=2
        for i = Mb:-1:2
            b[offset_1based+stride*(i-1)] -= upper_diag(A, i) * b[offset_1based+stride*i]
        end
        for j = 2:A.num_rows
            b[offset_1based] -= first_row(A, j) * b[offset_1based+stride*(j-1)]
        end
        b[offset_1based] /= main_diag(A, 1)
        for i = 2:A.num_rows
            b[offset_1based+stride*(i-1)] = (b[offset_1based+stride*(i-1)] - lower_diag(A, i) * b[offset_1based+stride*(i-2)]) * A.inv_diag[i]
        end

    else
        error("Invalid offset/stride combination")
    end

    return b
end

# Convenience methods
function UL_solve!(A::BandedTridiag{T}, b::AbstractVector{T}) where {T<:Number}
    UL_solve_strided!(A, b, 0, 1)
end

function UL_solve!(A::BandedTridiag{T}, b::ChebyCoeff{T}) where {T<:Number}
    UL_solve!(A, b.data)
end

function UL_solve_strided!(A::BandedTridiag{T}, b::ChebyCoeff{T},
    offset::Int, stride::Int) where {T<:Number}
    UL_solve_strided!(A, b.data, offset, stride)
end

"""Matrix-vector multiplication with strided access"""
function multiply_strided!(x::AbstractVector{T}, A::BandedTridiag{T},
    b::AbstractVector{T}, offset::Int, stride::Int) where {T<:Number}
    @assert offset in [0, 1] "offset must be 0 or 1"
    @assert stride in [1, 2] "stride must be 1 or 2"

    offset_1based = offset + 1
    Mbar = A.num_rows - 1

    # Row 0 (first row) - full band multiplication
    sum_val = zero(T)
    for j = 1:A.num_rows
        sum_val += first_row(A, j) * x[offset_1based+stride*(j-1)]
    end
    b[offset_1based] = sum_val

    # Rows 1 to Mbar-1 (tridiagonal structure)
    for i = 2:Mbar
        b[offset_1based+stride*(i-1)] =
            lower_diag(A, i) * x[offset_1based+stride*(i-2)] +
            main_diag(A, i) * x[offset_1based+stride*(i-1)] +
            upper_diag(A, i) * x[offset_1based+stride*i]
    end

    # Final row (only lower diagonal and diagonal)
    b[offset_1based+stride*(A.num_rows-1)] =
        lower_diag(A, A.num_rows) * x[offset_1based+stride*(A.num_rows-2)] +
        main_diag(A, A.num_rows) * x[offset_1based+stride*(A.num_rows-1)]

    return b
end

function multiply!(x::AbstractVector{T}, A::BandedTridiag{T}, b::AbstractVector{T}) where {T<:Number}
    multiply_strided!(x, A, b, 0, 1)
end

function multiply_strided!(x::ChebyCoeff{T}, A::BandedTridiag{T},
    b::ChebyCoeff{T}, offset::Int, stride::Int) where {T<:Number}
    multiply_strided!(x.data, A, b.data, offset, stride)
end

function multiply_strided(x::AbstractVector{T}, A::BandedTridiag{T}, offset::Int, stride::Int) where {T<:Number}
    b = zeros(T, size(x))
    multiply_strided!(x, A, b, offset, stride)
    return b
end

function multiply(x::AbstractVector{T}, A::BandedTridiag{T}) where {T<:Number}
    b = zeros(T, size(x))
    multiply!(x, A, b)
    return b
end

function Base.:*(A::BandedTridiag{T}, x::AbstractVector{T}) where {T<:Number}
    multiply(x, A)
end
# Pretty printing
function Base.show(io::IO, A::BandedTridiag{T}) where {T}
    print(io, "$(A.num_rows)×$(A.num_rows) BandedTridiag{$T}")
    if A.is_decomposed
        print(io, " (UL-decomposed)")
    end
end

function Base.display(A::BandedTridiag)
    println("$(A.num_rows)×$(A.num_rows) BandedTridiag:")
    for i = 1:A.num_rows
        print("  ")
        for j = 1:A.num_rows
            if abs(i - j) ≤ 1 || i == 1
                @printf "%8.3f " A[i, j]
            else
                print("   0.000 ")
            end
        end
        println()
    end
    if A.is_decomposed
        println("  (UL-decomposed)")
    end
end
"""
Extract U and L matrices from UL-decomposed BandedTridiag
"""
function extract_UL_matrices(A::BandedTridiag{T}) where {T}
    @assert A.is_decomposed "Matrix must be UL-decomposed"
    M = A.num_rows
    U = zeros(T, M, M)
    L = zeros(T, M, M)

    # In UL decomposition, typically:
    # - U is upper triangular with 1's on diagonal
    # - L is lower triangular 
    # - The decomposed matrix stores L below diagonal, U above diagonal

    # Extract U matrix (upper triangular with 1's on diagonal)
    for i = 1:M
        U[i, i] = one(T)  # 1's on diagonal
        for j = i+1:M
            U[i, j] = A[i, j]  # Upper triangular part
        end
    end

    # Extract L matrix (lower triangular including diagonal)
    for i = 1:M
        for j = 1:i
            L[i, j] = A[i, j]  # Lower triangular part including diagonal
        end
    end

    return U, L
end
"""
Convert BandedTridiag to a full dense matrix for testing
"""
function to_dense(A::BandedTridiag{T}) where {T}
    M = A.num_rows
    dense = zeros(T, M, M)
    for i = 1:M, j = 1:M
        dense[i, j] = A[i, j]
    end
    return dense
end
end
