#=
FlowFieldTransforms.jl

Handles FFTW plans and spectral transforms for FlowField objects.
Implements Fourier transforms in x,z directions and Chebyshev transforms in y direction.
=#

using FFTW
using AbstractFFTs
using LinearAlgebra: mul!
using Base.Threads

export FlowFieldTransforms,
    make_spectral_xz!, make_physical_xz!, make_physical_y!, make_spectral_y!

const _fftw_threads_initialized = Ref(false)

function _configured_fftw_threads()
    if haskey(ENV, "CHANNELFLOW_FFTW_THREADS")
        cfg = strip(ENV["CHANNELFLOW_FFTW_THREADS"])
        if lowercase(cfg) == "auto"
            return Threads.nthreads()
        end
        nthreads = tryparse(Int, cfg)
        if nthreads !== nothing && nthreads > 0
            return nthreads
        end
    end
    return 1
end

function _init_fftw_threads_once!()
    _fftw_threads_initialized[] && return
    FFTW.set_num_threads(max(1, _configured_fftw_threads()))
    _fftw_threads_initialized[] = true
    return
end

"""
Stores FFTW plans and scratch arrays for FlowField transforms.

The transforms handle:
- x,z directions: Real-to-complex FFT (forward) and complex-to-real FFT (inverse)
- y direction: Chebyshev transforms using DCT-I (Discrete Cosine Transform type I)
"""
mutable struct FlowFieldTransforms{T<:Real}
    # FFTW plans for x,z transforms
    xz_plan::FFTW.rFFTWPlan        # Real -> Complex (forward)
    xz_inverse_plan::AbstractFFTs.ScaledPlan # Complex -> Real (inverse)

    # FFTW plan for y transforms
    y_plan::FFTW.r2rFFTWPlan         # DCT-I for Chebyshev
    y_plan_mat::FFTW.r2rFFTWPlan     # Batched DCT-I along y for Ny x Nx slabs

    # Scratch arrays
    y_scratch::Vector{T}   # 1D scratch space for y transforms
    y_scratch2::Vector{T}  # secondary scratch to avoid per-profile allocations
    y_scratch_mat::Matrix{T}   # Ny x Nx batched scratch for y transforms
    y_scratch_mat2::Matrix{T}  # secondary Ny x Nx scratch
end

"""
    FlowFieldTransforms(domain)

Create FFTW plans for a given domain.
Plans are created immediately and stored for reuse.
"""
function FlowFieldTransforms(domain::FlowFieldDomain{T}) where {T}
    @assert domain.Nx > 0
    @assert domain.Nz > 0
    @assert domain.Ny >= 2

    _init_fftw_threads_once!()

    # Initialize scratch space for y transforms
    y_scratch = zeros(T, domain.Ny)
    y_scratch2 = zeros(T, domain.Ny)
    y_scratch_mat = zeros(T, domain.Ny, domain.Nx)
    y_scratch_mat2 = zeros(T, domain.Ny, domain.Nx)

    # xz_plan = nothing
    # xz_inverse_plan = nothing
    # y_plan = nothing

    # Create sample arrays for FFTW planning
    sample_physical = zeros(T, domain.Nx, domain.Ny, domain.Nz, domain.num_dimensions)
    sample_spectral =
        zeros(Complex{T}, domain.Nx, domain.My, domain.Mz, domain.num_dimensions)

    # Create xz transforms
    # Transform over dimensions (1,3) = (x,z) for each (y,i)
    xz_plan = plan_rfft(sample_physical, (3, 1); flags=FFTW.MEASURE)
    xz_inverse_plan =
        plan_irfft(sample_spectral, domain.Nz, (3, 1); flags=FFTW.MEASURE)

    # Y transform: DCT-I (REDFT00) for Chebyshev polynomials
    y_plan = FFTW.plan_r2r!(y_scratch, FFTW.REDFT00; flags=FFTW.MEASURE)
    y_plan_mat = FFTW.plan_r2r!(y_scratch_mat, FFTW.REDFT00, 1; flags=FFTW.MEASURE)

    return FlowFieldTransforms{T}(xz_plan, xz_inverse_plan, y_plan, y_plan_mat, y_scratch, y_scratch2, y_scratch_mat, y_scratch_mat2)
end

# ===========================
# XZ Transforms (Fourier)
# ===========================

"""
    make_spectral_xz!(physical_data, spectral_data, domain, transforms)

Transform x,z directions from physical to spectral space using real-to-complex FFT.

Input:  physical_data[Nx, Ny, Nz, num_dimensions] - real values u(x,y,z,i)
Output: spectral_data[Mx, My, Mz, num_dimensions] - complex coefficients û(kx,y,kz,i)
"""
function make_spectral_xz!(
    physical_data::Array{T,4},
    spectral_data::Array{Complex{T},4},
    domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T},
) where {T}

    if transforms.xz_plan === nothing
        error("XZ transform plan not initialized")
    end

    # Perform forward FFT: real -> complex
    # FFTW transforms over dimensions (1,3) = (x,z) for each (y,i).
    # Use mul! to avoid allocating a temporary transform array.
    mul!(spectral_data, transforms.xz_plan, physical_data)

    # Apply FFTW normalization (forward transform)
    scale_factor = T(1) / (domain.Nx * domain.Nz) # hmm, no longer needed...
    spectral_data .*= scale_factor

    return spectral_data
end

"""
    make_physical_xz!(spectral_data, physical_data, domain, transforms) 

Transform x,z directions from spectral to physical space using complex-to-real FFT.

Input:  spectral_data[Mx, My, Mz, num_dimensions] - complex coefficients û(kx,y,kz,i)
Output: physical_data[Nx, Ny, Nz, num_dimensions] - real values u(x,y,z,i)
"""
function make_physical_xz!(
    spectral_data::Array{Complex{T},4},
    physical_data::Array{T,4},
    domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T},
) where {T}
    if transforms.xz_inverse_plan === nothing
        error("XZ inverse transform plan not initialized")
    end

    # Perform inverse FFT: complex -> real.
    # Use mul! to avoid allocating a temporary transform array.
    mul!(physical_data, transforms.xz_inverse_plan, spectral_data)
    scale_factor = T(domain.Nx * domain.Nz)
    physical_data .*= scale_factor

    return physical_data
end

# ===========================
# Y Transforms (Chebyshev) 
# ===========================

"""
    make_spectral_y!(data, domain, transforms)

Transform y direction from physical to spectral space using Chebyshev transform.
This uses DCT-I (Discrete Cosine Transform type I) with proper normalization.

Works on either real or complex data arrays.

Input:  data contains values at Chebyshev-Gauss-Lobatto points
Output: data contains Chebyshev polynomial coefficients
"""
function make_spectral_y!(
    data::Array{T,4},
    domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T},
) where {T<:Real}
    if domain.Ny < 2
        return data
    end

    if transforms.y_plan === nothing
        error("Y transform plan not initialized")
    end

    nrm = T(1) / (domain.Ny - 1)

    # Transform Ny x Nx slabs in batch for each (nz, i)
    mat = transforms.y_scratch_mat
    @inbounds for i = 1:domain.num_dimensions
        for nz = 1:size(data, 3)
            for ny = 1:domain.Ny, nx = 1:domain.Nx
                mat[ny, nx] = data[nx, ny, nz, i]
            end

            transforms.y_plan_mat * mat

            for nx = 1:domain.Nx
                data[nx, 1, nz, i] = 0.5 * nrm * mat[1, nx]
                for ny = 2:(domain.Ny-1)
                    data[nx, ny, nz, i] = nrm * mat[ny, nx]
                end
                data[nx, domain.Ny, nz, i] = 0.5 * nrm * mat[domain.Ny, nx]
            end
        end
    end

    return data
end

function make_spectral_y!(
    data::Array{Complex{T},4},
    domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T},
) where {T<:Real}
    if domain.Ny < 2
        return data
    end

    if transforms.y_plan === nothing
        error("Y transform plan not initialized")
    end

    nrm = T(1) / (domain.Ny - 1)

    # Transform Ny x Nx slabs in batch for each (nz, i), real/imag separately.
    mat = transforms.y_scratch_mat
    mat2 = transforms.y_scratch_mat2
    @inbounds for i = 1:domain.num_dimensions
        for nz = 1:size(data, 3)
            for ny = 1:domain.Ny, nx = 1:domain.Nx
                mat[ny, nx] = real(data[nx, ny, nz, i])
            end

            transforms.y_plan_mat * mat

            for nx = 1:domain.Nx
                mat2[1, nx] = 0.5 * nrm * mat[1, nx]
                for ny = 2:(domain.Ny-1)
                    mat2[ny, nx] = nrm * mat[ny, nx]
                end
                mat2[domain.Ny, nx] = 0.5 * nrm * mat[domain.Ny, nx]
            end

            for ny = 1:domain.Ny, nx = 1:domain.Nx
                mat[ny, nx] = imag(data[nx, ny, nz, i])
            end

            transforms.y_plan_mat * mat

            for nx = 1:domain.Nx
                data[nx, 1, nz, i] = Complex{T}(mat2[1, nx], 0.5 * nrm * mat[1, nx])
                for ny = 2:(domain.Ny-1)
                    data[nx, ny, nz, i] = Complex{T}(mat2[ny, nx], nrm * mat[ny, nx])
                end
                data[nx, domain.Ny, nz, i] = Complex{T}(mat2[domain.Ny, nx], 0.5 * nrm * mat[domain.Ny, nx])
            end
        end
    end

    return data
end

"""
    make_physical_y!(data, domain, transforms)

Transform y direction from spectral to physical space using inverse Chebyshev transform.
This uses DCT-I with proper inverse normalization to exactly undo make_spectral_y!.

The forward transform applies:
- Overall factor: 1/(Ny-1)  
- Endpoint additional factor: 0.5
- So endpoints get: 0.5/(Ny-1), interior gets: 1/(Ny-1)

The DCT-I applied twice scales by 2(Ny-1), so to invert we need:
- Undo the forward normalization
- Account for the 2(Ny-1) scaling from double DCT-I
- Net result: just undo the endpoint scaling by factor of 2

Input:  data contains Chebyshev polynomial coefficients  
Output: data contains values at Chebyshev-Gauss-Lobatto points
"""
function make_physical_y!(
    data::Array{T,4},
    domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T},
) where {T<:Real}
    if domain.Ny < 2
        return data
    end

    if transforms.y_plan === nothing
        error("Y transform plan not initialized")
    end

    # Transform Ny x Nx slabs in batch for each (nz, i)
    mat = transforms.y_scratch_mat
    @inbounds for i = 1:domain.num_dimensions
        for nz = 1:size(data, 3)
            for nx = 1:domain.Nx
                mat[1, nx] = 2.0 * data[nx, 1, nz, i]
                for ny = 2:(domain.Ny-1)
                    mat[ny, nx] = data[nx, ny, nz, i]
                end
                mat[domain.Ny, nx] = 2.0 * data[nx, domain.Ny, nz, i]
            end

            transforms.y_plan_mat * mat

            for ny = 1:domain.Ny, nx = 1:domain.Nx
                data[nx, ny, nz, i] = mat[ny, nx] / 2.0
            end
        end
    end

    return data
end

function make_physical_y!(
    data::Array{Complex{T},4},
    domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T},
) where {T<:Real}
    if domain.Ny < 2
        return data
    end


    if transforms.y_plan === nothing
        error("Y transform plan not initialized")
    end

    # Transform Ny x Nx slabs in batch for each (nz, i), real/imag separately.
    mat = transforms.y_scratch_mat
    mat2 = transforms.y_scratch_mat2
    @inbounds for i = 1:domain.num_dimensions
        for nz = 1:size(data, 3)
            for nx = 1:domain.Nx
                mat[1, nx] = 2.0 * real(data[nx, 1, nz, i])
                for ny = 2:(domain.Ny-1)
                    mat[ny, nx] = real(data[nx, ny, nz, i])
                end
                mat[domain.Ny, nx] = 2.0 * real(data[nx, domain.Ny, nz, i])
            end

            transforms.y_plan_mat * mat

            for ny = 1:domain.Ny, nx = 1:domain.Nx
                mat2[ny, nx] = mat[ny, nx] / 2.0
            end

            for nx = 1:domain.Nx
                mat[1, nx] = 2.0 * imag(data[nx, 1, nz, i])
                for ny = 2:(domain.Ny-1)
                    mat[ny, nx] = imag(data[nx, ny, nz, i])
                end
                mat[domain.Ny, nx] = 2.0 * imag(data[nx, domain.Ny, nz, i])
            end

            transforms.y_plan_mat * mat

            for ny = 1:domain.Ny, nx = 1:domain.Nx
                data[nx, ny, nz, i] = Complex{T}(mat2[ny, nx], mat[ny, nx] / 2.0)
            end
        end
    end

    return data
end
