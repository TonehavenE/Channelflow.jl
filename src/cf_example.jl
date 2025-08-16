using ..Channelflow

function simple_channel_flow_example()
    println("=== Simple Channel Flow Example ===")

    # Domain parameters
    Nx, Ny, Nz = 16, 17, 16  # Small grid for testing
    Lx, Lz = 2π, 1π
    a, b = -1.0, 1.0  # Channel walls at y = ±1

    # Physical parameters
    nu = 10.01  # Viscosity
    dPdx = -1.0  # Pressure gradient driving the flow

    # Create DNS flags
    flags = DNSFlags(
        nu=nu,
        dPdx=dPdx,
        dPdz=0.0,
        baseflow=ParabolicBase,
        constraint=PressureGradient,
        dealiasing=DealiasXZ,
        timestepping=SBDF3
    )

    # Create velocity and pressure fields
    u = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b)  # Velocity field
    p = FlowField(Nx, Ny, Nz, 1, Lx, Lz, a, b)  # Pressure field
    rhs = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b)  # Right-hand side

    make_physical!(u)
    make_physical!(p)
    make_physical!(rhs)

    # Initialize with some test data
    for mx = 1:u.domain.Mx, ny = 1:Ny, mz = 1:u.domain.Mz
        # Add some small perturbation to trigger the solver
        u[mx, ny, mz, 1] = 0.01 * (rand() - 0.5)
        u[mx, ny, mz, 2] = 0.01 * (rand() - 0.5)
        u[mx, ny, mz, 3] = 0.01 * (rand() - 0.5)

        # Set a simple RHS (representing time derivative terms)
        rhs[mx, ny, mz, 1] = ComplexF64(0.1, 0.0)
        rhs[mx, ny, mz, 2] = ComplexF64(0.0, 0.0)
        rhs[mx, ny, mz, 3] = ComplexF64(0.0, 0.0)
    end

    make_spectral!(u)
    make_spectral!(p)
    make_spectral!(rhs)

    # Create NSE equation object
    fields = [u, p]
    nse = NSE(fields, flags)
    dns = MultistepDNS(fields, nse, flags)

    println("Domain: $(Nx)×$(Ny)×$(Nz), Lx=$(Lx), Lz=$(Lz)")
    println("Viscosity: $(nu), Pressure gradient: $(dPdx)")
    println("Base flow: $(flags.baseflow)")

    # Test the solver
    outfields = [u, p]
    rhsfields = [rhs]
    s = 1  # Time step index

    max_u = maximum(abs.(u.physical_data))
    max_p = maximum(abs.(p.physical_data))
    println("Max velocity magnitude: $(max_u)")
    println("Max pressure magnitude: $(max_p)")
    println("Calling solve!...")
    try
        solve!(nse, outfields, rhsfields, s, flags)
        println("✓ Solver completed successfully")
        make_physical!(u)
        make_physical!(p)

        # Check that solution was modified
        max_u = maximum(abs.(u.physical_data))
        max_p = maximum(abs.(p.physical_data))
        println("Max velocity magnitude: $(max_u)")
        println("Max pressure magnitude: $(max_p)")

    catch e
        println("✗ Solver failed with error: $e")
        rethrow(e)
    end

    return
    # return nse, outfields, rhsfields
end

simple_channel_flow_example()