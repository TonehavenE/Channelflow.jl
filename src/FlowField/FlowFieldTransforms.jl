#=
FlowFieldTransforms.jl

Handles FFTW plans and spectral transforms for FlowField objects.
Implements Fourier transforms in x,z directions and Chebyshev transforms in y direction.
=#

using FFTW
using AbstractFFTs

export FlowFieldTransforms, make_spectral_xz!, make_physical_xz!, make_physical_y!, make_spectral_y!

"""
Stores FFTW plans and scratch arrays for FlowField transforms.

The transforms handle:
- x,z directions: Real-to-complex FFT (forward) and complex-to-real FFT (inverse)
- y direction: Chebyshev transforms using DCT-I (Discrete Cosine Transform type I)
"""
mutable struct FlowFieldTransforms{T<:Real}
    # FFTW plans for x,z transforms
    xz_plan::Union{AbstractFFTs.Plan,Nothing,FFTW.r2rFFTWPlan}        # Real -> Complex (forward)
    xz_inverse_plan::Union{AbstractFFTs.Plan,Nothing,FFTW.r2rFFTWPlan} # Complex -> Real (inverse)

    # FFTW plan for y transforms
    y_plan::Union{AbstractFFTs.Plan,Nothing,FFTW.r2rFFTWPlan}         # DCT-I for Chebyshev

    # Scratch arrays
    y_scratch::Vector{T}  # 1D scratch space for y transforms
end

"""
    FlowFieldTransforms(domain)

Create FFTW plans for a given domain.
Plans are created immediately and stored for reuse.
"""
function FlowFieldTransforms(domain::FlowFieldDomain{T}) where {T}
    # Initialize scratch space for y transforms
    y_scratch = zeros(T, domain.Ny)

    xz_plan = nothing
    xz_inverse_plan = nothing
    y_plan = nothing

    if domain.Nx > 0 && domain.Nz > 0
        # Create sample arrays for FFTW planning
        sample_physical = zeros(T, domain.Nx, domain.Ny, domain.Nz, domain.num_dimensions)
        sample_spectral = zeros(Complex{T}, domain.Nx, domain.My, domain.Mz, domain.num_dimensions)

        # Create xz transforms
        # Transform over dimensions (1,3) = (x,z) for each (y,i)
        xz_plan = plan_rfft(sample_physical, (3, 1); flags=FFTW.MEASURE)
        # println("The size of sample_spectral is: $(size(sample_spectral))")
        # println("The domain is: $(domain)")
        # println("size(sample_spectral, 1): $(size(sample_spectral, 1))")
        # println("and div(2(domain.Mz - 1), 2) + 1 is $(div(2(domain.Mz - 1), 2) + 1)")
        xz_inverse_plan = plan_irfft(sample_spectral, domain.Nz, (3, 1); flags=FFTW.MEASURE)

        # Y transform: DCT-I (REDFT00) for Chebyshev polynomials
        if domain.Ny >= 2
            y_plan = FFTW.plan_r2r!(y_scratch, FFTW.REDFT00; flags=FFTW.MEASURE)
        end
    end

    return FlowFieldTransforms{T}(xz_plan, xz_inverse_plan, y_plan, y_scratch)
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
function make_spectral_xz!(physical_data::Array{T,4}, spectral_data::Array{Complex{T},4},
    domain::FlowFieldDomain{T}, transforms::FlowFieldTransforms{T}) where {T}

    if transforms.xz_plan === nothing
        error("XZ transform plan not initialized")
    end

    # Perform forward FFT: real -> complex
    # FFTW transforms over dimensions (1,3) = (x,z) for each (y,i)
    spectral_data .= transforms.xz_plan * physical_data

    # Apply FFTW normalization (forward transform)
    scale_factor = T(1) # / (domain.Nx * domain.Nz) # hmm, no longer needed...
    spectral_data .*= scale_factor

    return spectral_data
end

"""
    make_physical_xz!(spectral_data, physical_data, domain, transforms) 

Transform x,z directions from spectral to physical space using complex-to-real FFT.

Input:  spectral_data[Mx, My, Mz, num_dimensions] - complex coefficients û(kx,y,kz,i)
Output: physical_data[Nx, Ny, Nz, num_dimensions] - real values u(x,y,z,i)
"""
function make_physical_xz!(spectral_data::Array{Complex{T},4}, physical_data::Array{T,4},
    domain::FlowFieldDomain{T}, transforms::FlowFieldTransforms{T}) where {T}

    if transforms.xz_inverse_plan === nothing
        error("XZ inverse transform plan not initialized")
    end

    # Perform inverse FFT: complex -> real
    physical_data .= transforms.xz_inverse_plan * spectral_data

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
function make_spectral_y!(data::Array{T,4}, domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T}) where {T<:Real}

    if domain.Ny < 2
        return data  # Trivial case
    end

    if transforms.y_plan === nothing
        error("Y transform plan not initialized")
    end

    # Transform each y-profile separately
    # Data layout: [nx, ny, nz, i] - we transform over ny dimension

    for i in 1:domain.num_dimensions
        for nz in 1:size(data, 3)  # Handle both Nz and Mz cases
            for nx in 1:size(data, 1)  # Handle both Nx and Mx cases
                # Copy y-profile to scratch array
                for ny in 1:domain.Ny
                    transforms.y_scratch[ny] = real(data[nx, ny, nz, i])
                end

                # Perform DCT-I transform
                transforms.y_plan * transforms.y_scratch

                # Apply Chebyshev normalization and copy back
                # See original C++ code: special handling for endpoints
                nrm = T(1) / (domain.Ny - 1)

                # Endpoint normalization (0th and last coefficients get factor of 1/2)
                data[nx, 1, nz, i] = 0.5 * nrm * transforms.y_scratch[1]

                for ny in 2:(domain.Ny-1)
                    data[nx, ny, nz, i] = nrm * transforms.y_scratch[ny]
                end

                data[nx, domain.Ny, nz, i] = 0.5 * nrm * transforms.y_scratch[domain.Ny]
            end
        end
    end

    return data
end

function make_spectral_y!(data::Array{Complex{T},4}, domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T}) where {T<:Real}

    if domain.Ny < 2
        return data
    end

    if transforms.y_plan === nothing
        error("Y transform plan not initialized")
    end

    # Transform each y-profile separately for complex data
    # We need to handle real and imaginary parts separately

    for i in 1:domain.num_dimensions
        for nz in 1:size(data, 3)
            for nx in 1:size(data, 1)
                # Transform real part
                for ny in 1:domain.Ny
                    transforms.y_scratch[ny] = real(data[nx, ny, nz, i])
                end

                transforms.y_plan * transforms.y_scratch

                nrm = T(1) / (domain.Ny - 1)
                real_0 = 0.5 * nrm * transforms.y_scratch[1]
                real_end = 0.5 * nrm * transforms.y_scratch[domain.Ny]

                # Transform imaginary part
                for ny in 1:domain.Ny
                    transforms.y_scratch[ny] = imag(data[nx, ny, nz, i])
                end

                transforms.y_plan * transforms.y_scratch

                imag_0 = 0.5 * nrm * transforms.y_scratch[1]
                imag_end = 0.5 * nrm * transforms.y_scratch[domain.Ny]

                # Combine and store results
                data[nx, 1, nz, i] = Complex{T}(real_0, imag_0)

                for ny in 2:(domain.Ny-1)
                    # Recompute for middle coefficients
                    for ny2 in 1:domain.Ny
                        transforms.y_scratch[ny2] = real(data[nx, ny2, nz, i])
                    end
                    transforms.y_plan * transforms.y_scratch
                    real_part = nrm * transforms.y_scratch[ny]

                    for ny2 in 1:domain.Ny
                        transforms.y_scratch[ny2] = imag(data[nx, ny2, nz, i])
                    end
                    transforms.y_plan * transforms.y_scratch
                    imag_part = nrm * transforms.y_scratch[ny]

                    data[nx, ny, nz, i] = Complex{T}(real_part, imag_part)
                end

                data[nx, domain.Ny, nz, i] = Complex{T}(real_end, imag_end)
            end
        end
    end

    return data
end

"""
    make_physical_y!(data, domain, transforms)

Transform y direction from spectral to physical space using inverse Chebyshev transform.
This uses DCT-I with inverse normalization.

Works on either real or complex data arrays.

Input:  data contains Chebyshev polynomial coefficients  
Output: data contains values at Chebyshev-Gauss-Lobatto points
"""
function make_physical_y!(data::Array{T,4}, domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T}) where {T<:Real}

    if domain.Ny < 2
        return data  # Trivial case  
    end

    if transforms.y_plan === nothing
        error("Y transform plan not initialized")
    end

    # Transform each y-profile separately
    for i in 1:domain.num_dimensions
        for nz in 1:size(data, 3)
            for nx in 1:size(data, 1)
                # Copy y-profile to scratch with inverse normalization
                # See original C++ code: undo the endpoint scaling
                transforms.y_scratch[1] = real(data[nx, 1, nz, i])

                for ny in 2:(domain.Ny-1)
                    transforms.y_scratch[ny] = 0.5 * real(data[nx, ny, nz, i])
                end

                transforms.y_scratch[domain.Ny] = real(data[nx, domain.Ny, nz, i])

                # Perform inverse DCT-I transform (same as forward for DCT-I)
                transforms.y_plan * transforms.y_scratch

                # Copy back to data array
                for ny in 1:domain.Ny
                    data[nx, ny, nz, i] = transforms.y_scratch[ny]
                end
            end
        end
    end

    return data
end

function make_physical_y!(data::Array{Complex{T},4}, domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T}) where {T<:Real}

    if domain.Ny < 2
        return data  # Trivial case  
    end

    if transforms.y_plan === nothing
        error("Y transform plan not initialized")
    end

    # Transform each y-profile separately for complex data
    for i in 1:domain.num_dimensions
        for nz in 1:size(data, 3)
            for nx in 1:size(data, 1)
                # Transform real part
                transforms.y_scratch[1] = real(data[nx, 1, nz, i])

                for ny in 2:(domain.Ny-1)
                    transforms.y_scratch[ny] = 0.5 * real(data[nx, ny, nz, i])
                end

                transforms.y_scratch[domain.Ny] = real(data[nx, domain.Ny, nz, i])

                transforms.y_plan * transforms.y_scratch

                real_result = copy(transforms.y_scratch)

                # Transform imaginary part
                transforms.y_scratch[1] = imag(data[nx, 1, nz, i])

                for ny in 2:(domain.Ny-1)
                    transforms.y_scratch[ny] = 0.5 * imag(data[nx, ny, nz, i])
                end

                transforms.y_scratch[domain.Ny] = imag(data[nx, domain.Ny, nz, i])

                transforms.y_plan * transforms.y_scratch

                # Combine results
                for ny in 1:domain.Ny
                    data[nx, ny, nz, i] = Complex{T}(real_result[ny], transforms.y_scratch[ny])
                end
            end
        end
    end

    return data
end