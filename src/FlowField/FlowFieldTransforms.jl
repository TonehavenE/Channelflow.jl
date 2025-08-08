"""
FlowFieldTransforms.jl

Handles FFTW plans and spectral transforms for FlowField objects.
Implements Fourier transforms in x,z directions and Chebyshev transforms in y direction.
"""

using FFTW

"""
Stores FFTW plans and scratch arrays for FlowField transforms.

The transforms handle:
- x,z directions: Real-to-complex FFT (forward) and complex-to-real FFT (inverse)
- y direction: Chebyshev transforms using DCT-I (Discrete Cosine Transform type I)
"""
mutable struct FlowFieldTransforms{T<:Real}
    # FFTW plans for x,z transforms
    xz_plan::Union{FFTW.rFFTWPlan,Nothing}        # Real -> Complex (forward)
    xz_inverse_plan::Union{FFTW.rFFTWPlan,Nothing} # Complex -> Real (inverse)

    # FFTW plan for y transforms  
    y_plan::Union{FFTW.rFFTWPlan,Nothing}         # DCT-I for Chebyshev

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
        # Real array: [nx, ny, nz, i] but we only use non-padded z dimension for input
        sample_real = zeros(T, domain.Nx, domain.Ny, domain.Nz, domain.num_dimensions)

        # For planning the forward transform (real to complex)
        # FFTW expects the output to have size [Nx, Ny, Mz, num_dimensions]
        sample_complex = zeros(Complex{T}, domain.Nx, domain.Ny, domain.Mz, domain.num_dimensions)

        # Create xz transforms
        # Transform over dimensions (1,3) = (x,z) for each (y,i)
        xz_plan = plan_rfft(sample_real, (1, 3); flags=FFTW.MEASURE)
        xz_inverse_plan = plan_irfft(sample_complex, domain.Nz, (1, 3); flags=FFTW.MEASURE)

        # Y transform: DCT-I (REDFT00) for Chebyshev polynomials
        if domain.Ny >= 2
            y_plan = plan_r2r!(y_scratch, FFTW.REDFT00; flags=FFTW.MEASURE)
        end
    end

    return FlowFieldTransforms{T}(xz_plan, xz_inverse_plan, y_plan, y_scratch)
end

# ===========================
# XZ Transforms (Fourier)
# ===========================

"""
    make_spectral_xz!(data, domain, transforms)

Transform x,z directions from physical to spectral space using real-to-complex FFT.
The data array is modified in-place, reinterpreting real storage as complex.

Input:  data stores real values u(x,y,z,i)
Output: data stores complex coefficients û(kx,y,kz,i) in packed format
"""
function make_spectral_xz!(data::Array{T,4}, domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T}) where {T}

    if transforms.xz_plan === nothing
        error("XZ transform plan not initialized")
    end

    # Extract the unpadded portion for input to FFTW
    # data is [Nx, Ny, Nzpad, num_dimensions] but we only want [Nx, Ny, Nz, num_dimensions]
    input_data = view(data, :, :, 1:domain.Nz, :)

    # Perform forward FFT: real -> complex
    # Result has size [Nx, Ny, Mz, num_dimensions] with Mz = Nz/2+1
    complex_result = transforms.xz_plan * input_data

    # Now we need to pack the complex result back into the real array
    # FFTW packs complex numbers as [real, imag, real, imag, ...]
    # So data[ix, iy, 2*iz-1] = real part, data[ix, iy, 2*iz] = imag part

    for i in 1:domain.num_dimensions
        for ny in 1:domain.Ny
            for mx in 1:domain.Nx
                for mz in 1:domain.Mz
                    z_real_idx = 2 * mz - 1  # Real part index
                    z_imag_idx = 2 * mz      # Imaginary part index

                    if z_real_idx <= domain.Nzpad
                        data[mx, ny, z_real_idx, i] = real(complex_result[mx, ny, mz, i])
                    end
                    if z_imag_idx <= domain.Nzpad
                        data[mx, ny, z_imag_idx, i] = imag(complex_result[mx, ny, mz, i])
                    end
                end
            end
        end
    end

    # Apply FFTW normalization (forward transform)
    scale_factor = T(1) / (domain.Nx * domain.Nz)
    data .*= scale_factor

    return data
end

"""
    make_physical_xz!(data, domain, transforms) 

Transform x,z directions from spectral to physical space using complex-to-real FFT.
The data array is modified in-place, reinterpreting packed complex as real values.

Input:  data stores complex coefficients û(kx,y,kz,i) in packed format  
Output: data stores real values u(x,y,z,i)
"""
function make_physical_xz!(data::Array{T,4}, domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T}) where {T}

    if transforms.xz_inverse_plan === nothing
        error("XZ inverse transform plan not initialized")
    end

    # Unpack the complex data from real storage
    complex_input = zeros(Complex{T}, domain.Nx, domain.Ny, domain.Mz, domain.num_dimensions)

    for i in 1:domain.num_dimensions
        for ny in 1:domain.Ny
            for mx in 1:domain.Nx
                for mz in 1:domain.Mz
                    z_real_idx = 2 * mz - 1
                    z_imag_idx = 2 * mz

                    real_part = (z_real_idx <= domain.Nzpad) ? data[mx, ny, z_real_idx, i] : T(0)
                    imag_part = (z_imag_idx <= domain.Nzpad) ? data[mx, ny, z_imag_idx, i] : T(0)

                    complex_input[mx, ny, mz, i] = Complex{T}(real_part, imag_part)
                end
            end
        end
    end

    # Perform inverse FFT: complex -> real
    real_result = transforms.xz_inverse_plan * complex_input

    # Copy result back to data array (only the unpadded portion)
    data[:, :, 1:domain.Nz, :] .= real_result

    # Zero out the padded region
    if domain.Nzpad > domain.Nz
        data[:, :, (domain.Nz+1):domain.Nzpad, :] .= 0
    end

    return data
end

# ===========================
# Y Transforms (Chebyshev) 
# ===========================

"""
    make_spectral_y!(data, domain, transforms)

Transform y direction from physical to spectral space using Chebyshev transform.
This uses DCT-I (Discrete Cosine Transform type I) with proper normalization.

Input:  data contains values at Chebyshev-Gauss-Lobatto points
Output: data contains Chebyshev polynomial coefficients
"""
function make_spectral_y!(data::Array{T,4}, domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T}) where {T}

    if domain.Ny < 2
        return data  # Trivial case
    end

    if transforms.y_plan === nothing
        error("Y transform plan not initialized")
    end

    # Transform each y-profile separately
    # Data layout: [nx, ny, nz, i] - we transform over ny dimension

    for i in 1:domain.num_dimensions
        for nz in 1:domain.Nzpad  # Include padded region
            for nx in 1:domain.Nx
                # Copy y-profile to scratch array
                for ny in 1:domain.Ny
                    transforms.y_scratch[ny] = data[nx, ny, nz, i]
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

"""
    make_physical_y!(data, domain, transforms)

Transform y direction from spectral to physical space using inverse Chebyshev transform.
This uses DCT-I with inverse normalization.

Input:  data contains Chebyshev polynomial coefficients  
Output: data contains values at Chebyshev-Gauss-Lobatto points
"""
function make_physical_y!(data::Array{T,4}, domain::FlowFieldDomain{T},
    transforms::FlowFieldTransforms{T}) where {T}

    if domain.Ny < 2
        return data  # Trivial case  
    end

    if transforms.y_plan === nothing
        error("Y transform plan not initialized")
    end

    # Transform each y-profile separately
    for i in 1:domain.num_dimensions
        for nz in 1:domain.Nzpad
            for nx in 1:domain.Nx
                # Copy y-profile to scratch with inverse normalization
                # See original C++ code: undo the endpoint scaling
                transforms.y_scratch[1] = data[nx, 1, nz, i]

                for ny in 2:(domain.Ny-1)
                    transforms.y_scratch[ny] = 0.5 * data[nx, ny, nz, i]
                end

                transforms.y_scratch[domain.Ny] = data[nx, domain.Ny, nz, i]

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

# ===========================
# Utility Functions
# ===========================

"""
    optimize_plans!(transforms; flags=FFTW.PATIENT)

Re-optimize FFTW plans with different flags (e.g., FFTW.PATIENT for better performance).
"""
function optimize_plans!(transforms::FlowFieldTransforms{T}, domain::FlowFieldDomain{T};
    flags=FFTW.PATIENT) where {T}

    # This would recreate the plans with different optimization flags
    # For now, we'll leave the implementation as a stub since it requires
    # recreating all the sample arrays and plans

    @warn "Plan optimization not yet implemented"
    return transforms
end