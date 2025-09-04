using ..Channelflow

function simple_channel_flow_example()
    println("=== Simple Channel Flow Example ===")

    # Domain parameters
    Nx, Ny, Nz = 4, 5, 4  # Small grid for testing
    Lx, Lz = 2π, 1π
    a, b = -1.0, 1.0  # Channel walls at y = ±1

    # Physical parameters

    u = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b)  # Velocity field
    v = FlowField(Nx, Ny, Nz, 3, Lx, Lz, a, b)  # Velocity field

    make_physical!(u)

    u[1, 1, 1, 1] = 1

    make_spectral!(u)

    curl!(u, v)

    display(v)
end

simple_channel_flow_example()
