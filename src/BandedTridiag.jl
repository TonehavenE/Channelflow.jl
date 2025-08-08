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

"""
UL decomposition.
Performs elimination from bottom-right to top-left, creating:
- U: upper triangular factor (stored in upper diagonal and first row)
- L: lower triangular factor with unit diagonal (multipliers stored in lower diagonal)
"""
function UL_decompose!(A::BandedTridiag{T}) where {T}
    @assert A.num_rows ≥ 2 "Matrix must have at least 2 rows"
    
    # Main elimination loop: eliminate from bottom to top
    for k = A.num_rows:-1:3
        pivot = main_diag(A, k)
        @assert !iszero(pivot) "Zero pivot encountered at position ($k,$k)"
        
        # Get the lower diagonal element to eliminate
        multiplier = lower_diag(A, k)  # This becomes part of L
        
        # Update upper diagonal: U[k-1,k] = U[k-1,k] / U[k,k]
        upper_val = upper_diag(A, k-1) / pivot
        set_upper_diag!(A, k-1, upper_val)
        
        # Update main diagonal: U[k-1,k-1] -= L[k,k-1] * U[k-1,k]
        new_diag = main_diag(A, k-1) - multiplier * upper_val
        set_main_diag!(A, k-1, new_diag)
        
        # Update band elements: similar elimination for first row
        first_row_val = first_row(A, k) / pivot
        set_first_row!(A, k, first_row_val)
        
        new_first_row_val = first_row(A, k-1) - multiplier * first_row_val
        set_first_row!(A, k-1, new_first_row_val)
    end
    
    # Handle the boundary case between first row and second row
    pivot = main_diag(A, 2)
    @assert !iszero(pivot) "Zero pivot encountered at position (2,2)"
    
    multiplier = lower_diag(A, 2)
    first_row_normalized = first_row(A, 2) / pivot
    set_first_row!(A, 2, first_row_normalized)
    
    new_first_row = first_row(A, 1) - multiplier * first_row_normalized
    set_first_row!(A, 1, new_first_row)
    
    # Precompute inverse diagonal elements for efficient solving
    A.inv_diag .= inv.(main_diag(A, i) for i = 1:A.num_rows)
    
    A.is_decomposed = true
    return A
end

"""UL solve with strided access - matches C++ implementation exactly"""
function UL_solve_strided!(A::BandedTridiag{T}, b::AbstractVector{T},
    offset::Int, stride::Int) where {T<:Number}
    @assert A.is_decomposed "Matrix must be UL-decomposed first"
    @assert offset in [0, 1] "offset must be 0 or 1"
    @assert stride in [1, 2] "stride must be 1 or 2"
        
    # Define index mapping function based on offset and stride
    idx = if offset == 0 && stride == 1
        i -> i
    elseif offset == 1 && stride == 1
        i -> offset + 1 + i - 1
    elseif offset == 0 && stride == 2
        i -> stride * i - 1
    elseif offset == 1 && stride == 2
        i -> offset + 1 + stride * (i - 1)
    else
        error("Invalid offset/stride combination")
    end
    
    # Solve Uy=b by backsubstitution
    for i = (A.num_rows - 1):-1:2
        b[idx(i)] -= upper_diag(A, i) * b[idx(i+1)]
    end
    
    # Handle first row
    for j = 2:A.num_rows
        first_row_idx = if offset == 1 && stride == 1
            idx(1) + j - 1  # Special case for offset=1, stride=1
        else
            idx(j)
        end
        b[idx(1)] -= first_row(A, j) * b[first_row_idx]
    end
    
    # Solve Lx=y by forward substitution
    b[idx(1)] /= main_diag(A, 1)
    for i = 2:A.num_rows
        b[idx(i)] = (b[idx(i)] - lower_diag(A, i) * b[idx(i-1)]) * A.inv_diag[i]
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

"""Matrix-vector multiplication with strided access using Julia idioms"""
function multiply_strided!(x::AbstractVector{T}, A::BandedTridiag{T},
    b::AbstractVector{T}, offset::Int, stride::Int) where {T<:Number}
    @assert offset in [0, 1] "offset must be 0 or 1"
    @assert stride in [1, 2] "stride must be 1 or 2"
    
    # Create a view into the vectors with the appropriate stride pattern
    x_view = @view x[offset+1:stride:offset+1+stride*(A.num_rows-1)]
    b_view = @view b[offset+1:stride:offset+1+stride*(A.num_rows-1)]
    
    # Row 1 - full band multiplication
    b_view[1] = sum(first_row(A, j) * x_view[j] for j = 1:A.num_rows)
    
    # Rows 2 to num_rows-1 - tridiagonal structure
    for i = 2:(A.num_rows-1)
        b_view[i] = (lower_diag(A, i) * x_view[i-1] + 
                     main_diag(A, i) * x_view[i] + 
                     upper_diag(A, i) * x_view[i+1])
    end
    
    # Final row - only lower diagonal and main diagonal
    if A.num_rows > 1
        i = A.num_rows
        b_view[i] = (lower_diag(A, i) * x_view[i-1] + 
                     main_diag(A, i) * x_view[i])
    end
    
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
