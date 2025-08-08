module BandedTridiags

using ..ChebyCoeffs
using Printf

export BandedTridiag, UL_decompose!, UL_solve!, UL_solve_strided!, multiply_strided!, multiply!
export set_diag!, set_band!, set_updiag!, set_lodiag!, band, updiag, lodiag, diag

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
    M::Int              # number of rows (square matrix)
    Mbar::Int           # M - 1
    data::Vector{T}     # single array for all elements
    d_offset::Int       # offset to diagonal elements in data array
    invdiag::Vector{T}  # inverse diagonal elements for solving
    is_decomposed::Bool
end

# Constructor with type parameter
function BandedTridiag{T}(M::Int) where {T<:Number}
    @assert M >= 0 "Matrix size must be non-negative"

    Mbar = M - 1
    data_size = max(4 * M - 2, 0)  # matches C++ allocation, handle M=0 case
    data = zeros(T, data_size)
    d_offset = Mbar + 1    # Julia 1-based indexing adjustment
    invdiag = zeros(T, M)

    BandedTridiag{T}(M, Mbar, data, d_offset, invdiag, false)
end

# Convenience constructor - defaults to Float64
BandedTridiag(M::Int) = BandedTridiag{Float64}(M)

Base.size(A::BandedTridiag) = (A.M, A.M)

# Accessor methods matching C++ interface
"""Get/set band element A[0,j] (first row)"""
function band(A::BandedTridiag, j::Int)
    @boundscheck (1 ≤ j ≤ A.M) || throw(BoundsError(A, (1, j)))
    A.data[A.Mbar-j+2]  # Convert to 1-based indexing
end

function set_band!(A::BandedTridiag{T}, j::Int, val::T) where {T}
    @boundscheck (1 ≤ j ≤ A.M) || throw(BoundsError(A, (0, j)))
    A.data[A.Mbar-j+2] = val
    A.is_decomposed = false
end

"""Get/set diagonal element A[i,i]"""
function diag(A::BandedTridiag, i::Int)
    @boundscheck (1 ≤ i ≤ A.M) || throw(BoundsError(A, (i, i)))
    A.data[A.d_offset+3*(i-1)]
end

function set_diag!(A::BandedTridiag{T}, i::Int, val::T) where {T}
    @boundscheck (1 ≤ i ≤ A.M) || throw(BoundsError(A, (i, i)))
    A.data[A.d_offset+3*(i-1)] = val
    A.is_decomposed = false
end

"""Get/set upper diagonal element A[i,i+1]"""
function updiag(A::BandedTridiag, i::Int)
    @boundscheck (1 ≤ i ≤ A.M) || throw(BoundsError(A, (i, i + 1)))
    A.data[A.d_offset+3*(i-1)-1]
end

function set_updiag!(A::BandedTridiag{T}, i::Int, val::T) where {T}
    @boundscheck (1 ≤ i ≤ A.M) || throw(BoundsError(A, (i, i + 1)))
    A.data[A.d_offset+3*(i-1)-1] = val
    A.is_decomposed = false
end

"""Get/set lower diagonal element A[i,i-1]"""
function lodiag(A::BandedTridiag, i::Int)
    @boundscheck (1 ≤ i ≤ A.M) || throw(BoundsError(A, (i, i - 1)))
    A.data[A.d_offset+3*(i-1)+1]
end

function set_lodiag!(A::BandedTridiag{T}, i::Int, val::T) where {T}
    @boundscheck (1 ≤ i ≤ A.M) || throw(BoundsError(A, (i, i - 1)))
    A.data[A.d_offset+3*(i-1)+1] = val
    A.is_decomposed = false
end

# Matrix interface using 1-based indexing
function Base.getindex(A::BandedTridiag, i::Int, j::Int)
    @boundscheck (1 ≤ i ≤ A.M && 1 ≤ j ≤ A.M) || throw(BoundsError(A, (i, j)))

    if i == 1
        return band(A, j)
    elseif i == j
        return diag(A, i)
    elseif i == j + 1
        return lodiag(A, i)
    elseif i == j - 1
        return updiag(A, i)
    else
        return zero(eltype(A.data))
    end
end

function Base.setindex!(A::BandedTridiag{T}, val::T, i::Int, j::Int) where {T}
    @boundscheck (1 ≤ i ≤ A.M && 1 ≤ j ≤ A.M) || throw(BoundsError(A, (i, j)))

    if i == 1
        set_band!(A, j, val)
    elseif i == j
        set_diag!(A, i, val)
    elseif i == j + 1
        set_lodiag!(A, i, val)
    elseif i == j - 1
        set_updiag!(A, i, val)
    else
        abs(i - j) ≤ 1 || throw(ArgumentError("Cannot set element ($i,$j) outside the band structure"))
    end

    return val
end

"""UL decomposition (no pivoting) - matches C++ implementation exactly"""
function UL_decompose!(A::BandedTridiag{T}) where {T}
    @assert A.M ≥ 2
    Mb = A.M - 1

    # Main decomposition loop - matches C++ exactly
    for k = A.M:-1:3  # C++: for (int k = Mb; k > 1; --k)
        Akk = diag(A, k)
        @assert Akk != 0.0 "Zero diagonal element encountered"

        w = lodiag(A, k)  # A[k, k-1]

        # C++: diag(k-1) -= w * (updiag(k-1) /= Akk)
        updiag_val = updiag(A, k - 1) / Akk
        set_updiag!(A, k - 1, updiag_val)
        set_diag!(A, k - 1, diag(A, k - 1) - w * updiag_val)

        # C++: band(k-1) -= w * (band(k) /= Akk)
        band_val = band(A, k) / Akk
        set_band!(A, k, band_val)
        set_band!(A, k - 1, band(A, k - 1) - w * band_val)
    end

    # Handle first row - C++: band(0) -= lodiag(1) * (band(1) /= diag(1))
    band_1_val = band(A, 2) / diag(A, 2)  # band(1) in C++ is band(2) in Julia
    set_band!(A, 2, band_1_val)
    set_band!(A, 1, band(A, 1) - lodiag(A, 2) * band_1_val)

    # Compute inverse diagonal elements
    for i = 1:A.M
        A.invdiag[i] = 1.0 / diag(A, i)
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

    Mb = A.M - 1

    # Convert to 1-based indexing for array access
    offset_1based = offset + 1

    if offset == 0 && stride == 1
        # Standard case - matches C++ exactly

        # Solve Uy=b by backsubstitution
        # C++: for (i = Mb - 1; i > 0; --i) b[i] -= updiag(i) * b[i + 1];
        for i = Mb:-1:2  # Julia: Mb down to 2 (C++: Mb-1 down to 1)
            b[i] -= updiag(A, i) * b[i+1]
        end

        # C++: for (j = i + 1; j < M_; ++j) b[0] -= band(j) * b[j];
        # After the loop, i = 0 in C++, so j goes from 1 to M-1
        for j = 2:A.M  # Julia: 2 to M (C++: 1 to M-1)
            b[1] -= band(A, j) * b[j]
        end

        # Solve Lx=y by forward substitution
        # C++: b[0] /= diag(0)
        b[1] /= diag(A, 1)

        # C++: for (i = 1; i < M_; ++i) (b[i] -= lodiag(i) * b[i - 1]) /= diag(i);
        for i = 2:A.M  # Julia: 2 to M (C++: 1 to M-1)
            b[i] = (b[i] - lodiag(A, i) * b[i-1]) * A.invdiag[i]
        end

    elseif offset == 1 && stride == 1
        # C++ implementation for offset=1, stride=1
        for i = Mb:-1:2
            b[offset_1based+i] -= updiag(A, i) * b[offset_1based+i+1]
        end
        for j = 2:A.M
            b[offset_1based] -= band(A, j) * b[offset_1based+j-1]
        end
        b[offset_1based] /= diag(A, 1)
        for i = 2:A.M
            b[offset_1based+i-1] = (b[offset_1based+i-1] - lodiag(A, i) * b[offset_1based+i-2]) * A.invdiag[i]
        end

    elseif offset == 0 && stride == 2
        # C++ implementation for offset=0, stride=2
        for i = Mb:-1:2
            b[stride*i-1] -= updiag(A, i) * b[stride*(i+1)-1]
        end
        for j = 2:A.M
            b[1] -= band(A, j) * b[stride*j-1]
        end
        b[1] /= diag(A, 1)
        for i = 2:A.M
            # @info "i=$i, invdiag=$(A.invdiag[i]), lodiag=$(lodiag(A, i)), b[i-1]=$(b[stride*(i - 1) - 1])"
            b[stride*i-1] = (b[stride*i-1] - lodiag(A, i) * b[stride*(i-1)-1]) * A.invdiag[i]
        end

    elseif offset == 1 && stride == 2
        # C++ implementation for offset=1, stride=2
        for i = Mb:-1:2
            b[offset_1based+stride*(i-1)] -= updiag(A, i) * b[offset_1based+stride*i]
        end
        for j = 2:A.M
            b[offset_1based] -= band(A, j) * b[offset_1based+stride*(j-1)]
        end
        b[offset_1based] /= diag(A, 1)
        for i = 2:A.M
            b[offset_1based+stride*(i-1)] = (b[offset_1based+stride*(i-1)] - lodiag(A, i) * b[offset_1based+stride*(i-2)]) * A.invdiag[i]
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
    Mbar = A.M - 1
    # @info "Mbar = $Mbar"

    # Row 0 (first row) - full band multiplication
    sum_val = zero(T)
    for j = 1:A.M
        sum_val += band(A, j) * x[offset_1based+stride*(j-1)]
    end
    b[offset_1based] = sum_val

    # Rows 1 to Mbar-1 (tridiagonal structure)
    for i = 2:Mbar
        b[offset_1based+stride*(i-1)] =
            lodiag(A, i) * x[offset_1based+stride*(i-2)] +
            diag(A, i) * x[offset_1based+stride*(i-1)] +
            updiag(A, i) * x[offset_1based+stride*i]
    end

    # Final row (only lower diagonal and diagonal)
    b[offset_1based+stride*(A.M-1)] =
        lodiag(A, A.M) * x[offset_1based+stride*(A.M-2)] +
        diag(A, A.M) * x[offset_1based+stride*(A.M-1)]
    @info "offset_1based = $offset_1based, stride = $stride, A.M-1 = $(A.M - 1), total: $(offset_1based + stride * (A.M - 1))"
    @info "lodiag(A, A.M) = $(lodiag(A, A.M)), x[offset+stride] = $(x[offset_1based+stride*(A.M-2)]), diag(A, A.M) = $(diag(A, A.M)), and x[offset + stride(A.M - 1)] =  $(x[offset_1based+stride*(A.M-1)])"
    @info "so total: $(lodiag(A, A.M) * x[offset_1based+stride*(A.M-2)] + diag(A, A.M) * x[offset_1based+stride*(A.M-1)])"

    return b
end

function multiply!(x::AbstractVector{T}, A::BandedTridiag{T}, b::AbstractVector{T}) where {T<:Number}
    multiply_strided!(x, A, b, 0, 1)
end

function multiply_strided!(x::ChebyCoeff{T}, A::BandedTridiag{T},
    b::ChebyCoeff{T}, offset::Int, stride::Int) where {T<:Number}
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
