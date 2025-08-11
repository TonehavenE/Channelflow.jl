using Test
using Channelflow

@testset "FlowField with Tensor Support" begin

    # ===========================
    # Tensor FlowField Constructor Tests  
    # ===========================
    @testset "Tensor FlowField Constructors" begin
        @testset "Scalar Field" begin
            temp = FlowField(8, 5, 16, SCALAR_TENSOR, 2π, 1π, -1.0, 1.0)

            @test tensor_shape(temp) == SCALAR_TENSOR
            @test tensor_rank(temp) == 1
            @test is_scalar_field(temp)
            @test !is_vector_field(temp)
            @test !is_matrix_field(temp)
            @test num_dimensions(temp) == 1
            @test size(temp.spectral_data) == (8, 5, 9, 1)
        end
        @testset "Vector Field (Explicit Tensor)" begin
            vector_field = FlowField(8, 5, 16, VECTOR_TENSOR, 2π, 1π, -1.0, 1.0)

            @test tensor_shape(vector_field) == VECTOR_TENSOR
            @test tensor_rank(vector_field) == 1
            @test is_vector_field(vector_field)
            @test !is_scalar_field(vector_field)
            @test num_dimensions(vector_field) == 3
            @test size(vector_field.spectral_data) == (8, 5, 9, 3)
        end
        @testset "Matrix Field" begin
            stress = FlowField(8, 5, 16, MATRIX_TENSOR, 2π, 1π, -1.0, 1.0)

            @test tensor_shape(stress) == MATRIX_TENSOR
            @test tensor_rank(stress) == 2
            @test is_matrix_field(stress)
            @test !is_vector_field(stress)
            @test num_dimensions(stress) == 9
            @test size(stress.spectral_data) == (8, 5, 9, 9)
        end

        @testset "Symmetric Tensor Field" begin
            strain = FlowField(8, 5, 16, SYMMETRIC_TENSOR, 2π, 1π, -1.0, 1.0)

            @test tensor_shape(strain) == SYMMETRIC_TENSOR
            @test tensor_rank(strain) == 1  # stored as 1D with 6 components
            @test is_symmetric_field(strain)
            @test !is_matrix_field(strain)
            @test num_dimensions(strain) == 6
            @test size(strain.spectral_data) == (8, 5, 9, 6)
        end

        @testset "Custom Tensor Shapes" begin
            # 4th order tensor (elasticity)
            elasticity = FlowField(4, 3, 8, TensorShape((3, 3, 3, 3)), 2π, 1π, -1.0, 1.0)

            @test tensor_shape(elasticity) == TensorShape((3, 3, 3, 3))
            @test tensor_rank(elasticity) == 4
            @test num_dimensions(elasticity) == 3^4
            @test size(elasticity.spectral_data) == (4, 3, 5, 3^4)

            # Rank-5 tensor
            rank5 = FlowField(4, 3, 8, TensorShape((2, 2, 2, 2, 2)), 2π, 1π, -1.0, 1.0)
            @test num_dimensions(rank5) == 32
        end

        @testset "Backward Compatibility" begin
            # Integer constructor should still work
            vel_old = FlowField(8, 5, 16, 3, 2π, 1π, -1.0, 1.0)
            vel_new = FlowField(8, 5, 16, VECTOR_TENSOR, 2π, 1π, -1.0, 1.0)

            @test tensor_shape(vel_old) == TensorShape((3,))
            @test tensor_shape(vel_new) == VECTOR_TENSOR
            @test num_dimensions(vel_old) == num_dimensions(vel_new)
            @test congruent(vel_old, vel_new)
        end
    end
    # ===========================
    # Tensor Element Access Tests
    # ===========================
    @testset "Tensor Element Access" begin
        @testset "Scalar Field Access" begin
            temp = FlowField(4, 3, 8, SCALAR_TENSOR, 2π, 1π, -1.0, 1.0)
            make_physical!(temp)

            temp[2, 2, 4, 1] = 273.15
            @test temp[2, 2, 4, 1] == 273.15
        end

        @testset "Vector Field Access" begin
            velocity = FlowField(4, 3, 8, VECTOR_TENSOR, 2π, 1π, -1.0, 1.0)
            make_physical!(velocity)

            # Set velocity components
            velocity[2, 2, 4, 1] = 1.5  # u component
            velocity[2, 2, 4, 2] = -0.7 # v component
            velocity[2, 2, 4, 3] = 2.1  # w component

            @test velocity[2, 2, 4, 1] == 1.5
            @test velocity[2, 2, 4, 2] == -0.7
            @test velocity[2, 2, 4, 3] == 2.1
        end

        @testset "Matrix Field Access" begin
            stress = FlowField(4, 3, 8, MATRIX_TENSOR, 2π, 1π, -1.0, 1.0)
            make_physical!(stress)

            # Set stress tensor components σ_ij
            stress[2, 2, 4, 1, 1] = 100.0  # σ_11
            stress[2, 2, 4, 1, 2] = 25.0   # σ_12
            stress[2, 2, 4, 2, 1] = 30.0   # σ_21
            stress[2, 2, 4, 3, 3] = -50.0  # σ_33

            @test stress[2, 2, 4, 1, 1] == 100.0
            @test stress[2, 2, 4, 1, 2] == 25.0
            @test stress[2, 2, 4, 2, 1] == 30.0
            @test stress[2, 2, 4, 3, 3] == -50.0
        end

        @testset "Symmetric Tensor Access" begin
            strain = FlowField(4, 3, 8, SYMMETRIC_TENSOR, 2π, 1π, -1.0, 1.0)
            make_physical!(strain)

            # Set symmetric strain tensor components
            strain[2, 2, 4, 1, 1] = 0.001  # ε_11 (diagonal, position 1)
            strain[2, 2, 4, 2, 2] = 0.002  # ε_22 (diagonal, position 2)  
            strain[2, 2, 4, 3, 3] = 0.003  # ε_33 (diagonal, position 3)
            strain[2, 2, 4, 1, 2] = 0.0005 # ε_12 = ε_21 (off-diagonal, position 4)
            strain[2, 2, 4, 1, 3] = 0.0007 # ε_13 = ε_31 (off-diagonal, position 5)
            strain[2, 2, 4, 2, 3] = 0.0009 # ε_23 = ε_32 (off-diagonal, position 6)

            @test strain[2, 2, 4, 1, 1] == 0.001
            @test strain[2, 2, 4, 2, 2] == 0.002
            @test strain[2, 2, 4, 3, 3] == 0.003
            @test strain[2, 2, 4, 1, 2] == 0.0005
            @test strain[2, 2, 4, 2, 1] == 0.0005  # Should be same as (1,2)
            @test strain[2, 2, 4, 1, 3] == 0.0007
            @test strain[2, 2, 4, 3, 1] == 0.0007  # Should be same as (1,3)
            @test strain[2, 2, 4, 2, 3] == 0.0009
            @test strain[2, 2, 4, 3, 2] == 0.0009  # Should be same as (2,3)
        end

        @testset "Higher-Order Tensor Access" begin
            elasticity = FlowField(4, 3, 8, TensorShape((3, 3, 3, 3)), 2π, 1π, -1.0, 1.0)
            make_physical!(elasticity)

            elasticity[1, 1, 1, 1, 1, 1, 1] = 210000.0  # C_1111 (Young's modulus direction)
            elasticity[1, 1, 1, 1, 2, 2, 2] = 80000.0   # C_1122 (Poisson effect)

            @test elasticity[1, 1, 1, 1, 1, 1, 1] == 210000.0
            @test elasticity[1, 1, 1, 1, 2, 2, 2] == 80000.0
        end
    end

    # ===========================
    # Spectral Tensor Access Tests
    # ===========================
    @testset "Spectral Tensor Access" begin
        @testset "Matrix Spectral Access" begin
            stress = FlowField(4, 3, 8, MATRIX_TENSOR, 2π, 1π, -1.0, 1.0)

            # Set spectral coefficients
            set_cmplx!(stress, Complex(1.0, 0.5), 1, 1, 1, 1, 1)  # σ̂_11(k=0)
            set_cmplx!(stress, Complex(0.3, -0.2), 2, 1, 2, 1, 2) # σ̂_12(k_x=1, k_z=1)

            @test cmplx(stress, 1, 1, 1, 1, 1) == Complex(1.0, 0.5)
            @test cmplx(stress, 2, 1, 2, 1, 2) == Complex(0.3, -0.2)
        end

        @testset "Symmetric Tensor Spectral Access" begin
            strain = FlowField(4, 3, 8, SYMMETRIC_TENSOR, 2π, 1π, -1.0, 1.0)

            # Set spectral coefficients for symmetric tensor
            set_cmplx!(strain, Complex(0.001, 0.0), 1, 1, 1, 1, 1)   # ε̂_11
            set_cmplx!(strain, Complex(0.0005, 0.0001), 1, 1, 1, 1, 2) # ε̂_12 = ε̂_21

            @test cmplx(strain, 1, 1, 1, 1, 1) == Complex(0.001, 0.0)
            @test cmplx(strain, 1, 1, 1, 1, 2) == Complex(0.0005, 0.0001)
            @test cmplx(strain, 1, 1, 1, 2, 1) == Complex(0.0005, 0.0001)  # Symmetric access
        end
    end

    # ===========================
    # Tensor Field Operations Tests
    # ===========================
    @testset "Tensor Field Operations" begin
        @testset "Tensor Field Addition" begin
            stress1 = FlowField(4, 3, 8, MATRIX_TENSOR, 2π, 1π, -1.0, 1.0)
            stress2 = FlowField(4, 3, 8, MATRIX_TENSOR, 2π, 1π, -1.0, 1.0)
            make_physical!(stress1)
            make_physical!(stress2)

            stress1[1, 1, 1, 1, 1] = 100.0
            stress1[1, 1, 1, 1, 2] = 50.0
            stress2[1, 1, 1, 1, 1] = 200.0
            stress2[1, 1, 1, 1, 2] = -30.0

            stress3 = stress1 + stress2
            @test stress3[1, 1, 1, 1, 1] == 300.0
            @test stress3[1, 1, 1, 1, 2] == 20.0
        end

        @testset "Tensor Field Scaling" begin
            vel = FlowField(4, 3, 8, VECTOR_TENSOR, 2π, 1π, -1.0, 1.0)
            make_physical!(vel)

            vel[1, 1, 1, 1] = 2.0
            vel[1, 1, 1, 2] = -1.5
            vel[1, 1, 1, 3] = 3.0

            vel_scaled = vel * 2.0
            @test vel_scaled[1, 1, 1, 1] == 4.0
            @test vel_scaled[1, 1, 1, 2] == -3.0
            @test vel_scaled[1, 1, 1, 3] == 6.0
        end

        @testset "Mixed Tensor Operations" begin
            # Can't add tensors of different shapes
            vel = FlowField(4, 3, 8, VECTOR_TENSOR, 2π, 1π, -1.0, 1.0)
            stress = FlowField(4, 3, 8, MATRIX_TENSOR, 2π, 1π, -1.0, 1.0)

            @test !congruent(vel, stress)
            @test_throws AssertionError vel + stress
        end
    end

    # ===========================  
    # Tensor Transform Tests
    # ===========================
    @testset "Tensor Transform Tests" begin
        @testset "Matrix Field Transforms" begin
            stress = FlowField(8, 5, 16, MATRIX_TENSOR, 2π, 1π, -1.0, 1.0)
            make_physical!(stress)

            # Set a simple pattern
            for i = 1:3, j = 1:3
                stress[1, 1, 1, i, j] = Float64(i * j)
            end

            # Transform to spectral and back
            make_spectral!(stress)
            @test stress.xz_state == Spectral
            @test stress.y_state == Spectral

            make_physical!(stress)

            # Check values preserved through transform
            for i = 1:3, j = 1:3
                @test stress[1, 1, 1, i, j] ≈ Float64(i * j) atol = 1e-14
            end
        end

        @testset "Symmetric Tensor Transforms" begin
            strain = FlowField(8, 5, 16, SYMMETRIC_TENSOR, 2π, 1π, -1.0, 1.0)
            make_physical!(strain)

            # Set diagonal and off-diagonal components
            strain[2, 3, 4, 1, 1] = 0.001  # ε_11
            strain[2, 3, 4, 2, 2] = 0.002  # ε_22
            strain[2, 3, 4, 3, 3] = 0.003  # ε_33
            strain[2, 3, 4, 1, 2] = 0.0005 # ε_12 = ε_21
            strain[2, 3, 4, 1, 3] = 0.0007 # ε_13 = ε_31
            strain[2, 3, 4, 2, 3] = 0.0009 # ε_23 = ε_32

            make_spectral_xz!(strain)
            make_physical_xz!(strain)

            # Check symmetry preserved
            @test strain[2, 3, 4, 1, 2] ≈ strain[2, 3, 4, 2, 1] atol = 1e-14
            @test strain[2, 3, 4, 1, 3] ≈ strain[2, 3, 4, 3, 1] atol = 1e-14
            @test strain[2, 3, 4, 2, 3] ≈ strain[2, 3, 4, 3, 2] atol = 1e-14
        end
    end
end