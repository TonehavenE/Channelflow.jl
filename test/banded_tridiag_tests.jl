using Test
using Channelflow
using LinearAlgebra

@testset "BandedTridiag tests" begin
    N = 8

    # test constructors 

    @testset "Constructors and Basics" begin
        A = BandedTridiag(N)
        B = BandedTridiag{Rational}(N)

        @test typeof(A) != typeof(B)
        @test typeof(A) <: BandedTridiag
        @test any(x -> x == 0, A.data)
        @test size(A) == (N, N)

        C = BandedTridiag(0)
        @test size(C) == (0, 0)

        @test !(BandedTridiag(4) == BandedTridiag(64))
        @test_throws AssertionError BandedTridiag(-1)
    end


    # test accessors and getters
    @testset "Accessors and Getters" begin
        N = 64
        A = BandedTridiag(N)
        for i = 1:N
            set_first_row!(A, i, Float64(i))
            set_main_diag!(A, i, Float64(i^2))
            set_upper_diag!(A, i, Float64(i^3))
            set_lower_diag!(A, i, Float64(i^4))
        end
        for i = 1:N
            @test first_row(A, i) == i
            @test main_diag(A, i) == i^2
            @test (upper_diag(A, i) == i^3 || upper_diag(A, i) == i + 1) # because of overlap between band and upper
            @test lower_diag(A, i) == i^4
        end


        for out_of_bounds in [-100, -1, N + 1]
            @test_throws BoundsError set_first_row!(A, out_of_bounds, 0.0)
            @test_throws BoundsError set_main_diag!(A, out_of_bounds, 0.0)
            @test_throws BoundsError set_lower_diag!(A, out_of_bounds, 0.0)
            @test_throws BoundsError set_upper_diag!(A, out_of_bounds, 0.0)
            @test_throws BoundsError first_row(A, out_of_bounds)
            @test_throws BoundsError main_diag(A, out_of_bounds)
            @test_throws BoundsError lower_diag(A, out_of_bounds)
            @test_throws BoundsError upper_diag(A, out_of_bounds)
        end

        for bad_type in [-100.1, 1 / 4, true, "str", 1.11]
            @test_throws MethodError set_first_row!(A, bad_type, 0.0)
            @test_throws MethodError set_main_diag!(A, bad_type, 0.0)
            @test_throws MethodError set_lower_diag!(A, bad_type, 0.0)
            @test_throws MethodError set_upper_diag!(A, bad_type, 0.0)
            @test_throws MethodError first_row(A, bad_type)
            @test_throws MethodError main_diag(A, bad_type)
            @test_throws MethodError lower_diag(A, bad_type)
            @test_throws MethodError upper_diag(A, bad_type)
        end

        # direct indexing 
        for i = 2:N-1 # all should be valid, edge cases handled below
            @test A[i, i] == main_diag(A, i)
            @test A[i, i+1] == upper_diag(A, i)
            @test A[i, i-1] == lower_diag(A, i)
        end

        @test A[2, 1] == lower_diag(A, 2)
        @test A[N-1, N] == upper_diag(A, N - 1)

        @test_throws BoundsError A[1, 0]
        @test_throws BoundsError A[N, N+1]
        @test_throws BoundsError A[N+1, N]
    end

    # test decomposition 
    @testset "LU Decomposition and Solving" begin
        function test_UL_decomposition(A::BandedTridiag{T}) where {T<:Number}
            # -- Test decomposition --
            A_orig = to_dense(A)

            UL_decompose!(A)
            U, L = extract_UL_matrices(A)
            UL_product = U * L
            reconstruction_error = maximum(abs, UL_product - A_orig)
            @test reconstruction_error < 1e-12

            n = size(A_orig, 1)

            # -- Test solution with multiple random vectors --
            max_solution_error = 0.0
            for trial = 1:5
                x_true = randn(T, n)
                b = A_orig * x_true

                x_computed = copy(b)
                UL_solve!(A, x_computed)

                solution_error = maximum(abs, x_computed - x_true)
                max_solution_error = max(max_solution_error, solution_error)
            end
            @test max_solution_error < 1e-12


            # -- Test residuals -- 
            x_test = randn(T, n)
            b_test = A_orig * x_test

            # Solve using our UL decomposition
            x_solved = copy(b_test)
            UL_solve!(A, x_solved)

            # Compute residual using original matrix
            residual = A_orig * x_solved - b_test
            residual_norm = maximum(abs, residual)
            @test residual_norm < 1e-12

            # -- Comparison with LinearAlgebra.jl (if matrix is well-conditioned) -- 
            try
                # Julia's LU gives PA = LU, we want UL = A
                # So we solve the same system both ways and compare
                F = lu(A_orig)

                b_compare = randn(T, n)
                x_julia = F \ b_compare

                x_ours = copy(b_compare)
                UL_solve!(A, x_ours)

                comparison_error = maximum(abs, x_julia - x_ours)
                @test comparison_error < 1e-12
            catch e
                println("Comparison to LinearAlgebra SKIPPED ($(typeof(e)))")
            end
        end
        T = Float64
        A = BandedTridiag{T}(6)

        # construct a matrix A
        # band first, alternating pattern
        for j = 1:6
            A[1, j] = j % 2 == 1 ? 1.0 : -1.0
        end

        # Tridiagonal part - tried to replicate typical Helmholtz structure
        lambda, nu = 2.0, 1.5
        for i = 2:6
            if i > 1
                A[i, i-1] = -nu
            end     # -nu on lower diagonal
            A[i, i] = 2 * nu + lambda           # 2*nu + lambda on diagonal
            if i < 6
                A[i, i+1] = -nu
            end    # -nu on upper diagonal
        end
        test_UL_decomposition(A)

        # test on random matrices
        num_tests = 10
        for i = 1:num_tests
            n = rand(4:8)  # Random size
            A = BandedTridiag(n)

            # Fill first row
            for j = 1:n
                A[1, j] = randn()
            end

            # Fill tridiagonal part with diagonally dominant structure
            for i = 2:n
                if i > 1
                    A[i, i-1] = randn()
                end
                A[i, i] = 5.0 + rand()  # Ensure non-zero diagonal
                if i < n
                    A[i, i+1] = randn()
                end
            end

            test_UL_decomposition(A)
        end

        # -- edge cases --

        # Small matrices
        A1 = BandedTridiag(1)
        A1[1, 1] = 2.0
        @test_throws AssertionError UL_decompose!(A1)

        A2 = BandedTridiag(2)
        A2[1, 1] = 1.0
        A2[1, 2] = 2.0
        A2[2, 1] = 3.0
        A2[2, 2] = 4.0
        test_UL_decomposition(A2)

        # Identity-like matrix
        A3 = BandedTridiag(5)
        for i = 1:5
            A3[1, i] = 1.0
        end  # First row all ones
        for i = 2:5
            A3[i, i] = 1.0
        end  # Diagonal ones
        test_UL_decomposition(A3)

    end

    @testset "Matrix-Vector Multiplication" begin
        A = BandedTridiag(4)

        # Set up test matrix
        A[1, 1] = 1.0
        A[1, 2] = 2.0
        A[1, 3] = 3.0
        A[1, 4] = 4.0
        A[2, 1] = 1.0
        A[2, 2] = 2.0
        A[2, 3] = 1.0
        A[3, 2] = 1.0
        A[3, 3] = 3.0
        A[3, 4] = 1.0
        A[4, 3] = 1.0
        A[4, 4] = 2.0

        A_dense = to_dense(A)

        x = [1.0, 2.0, 3.0, 4.0]
        b_expected = A_dense * x

        # Test standard multiplication
        # b_computed = multiply(x, A)
        b_computed = A * x
        @test b_computed ≈ b_expected

        # Test strided multiplication
        for offset in [0, 1], stride in [1, 2]
            if offset == 0 && stride == 1
                continue
            end

            extended_size = offset + 1 + stride * (length(x) - 1)
            x_extended = zeros(extended_size)
            # b_extended = zeros(extended_size)

            for i in eachindex(x)
                idx = offset + 1 + stride * (i - 1)
                x_extended[idx] = x[i]
            end

            # multiply_strided!(x_extended, A, b_extended, offset, stride)
            b_extended = multiply_strided(x_extended, A, offset, stride)

            b_strided = zeros(length(b_expected))
            for i = 1:length(b_expected)
                idx = offset + 1 + stride * (i - 1)
                b_strided[i] = b_extended[idx]
            end

            @test b_strided ≈ b_expected atol = 1e-10
        end
    end

    # test multiply
end
