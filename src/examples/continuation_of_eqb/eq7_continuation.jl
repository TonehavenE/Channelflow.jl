using Channelflow
using Printf

function eq7_continuation()
    println("=== EQ7 Continuation Test ===")

    # 1. SET STABLE PARAMETERS
    # A small grid, high viscosity, and small timestep are chosen for stability.
    println(pwd())
    domain_path = "./src/examples/continuation_of_eqb/eq7.geom"
    data_path = "./src/examples/continuation_of_eqb/eq7.asc"

    domain = read_geom(domain_path)

    Reynolds = 400
    nu = 1.0 / Reynolds  # High viscosity for stability
    dt = 0.01 # Small time step

    # Create DNS flags for a CNRK2 (Crank-Nicolson Runge-Kutta 2nd order) simulation
    flags = DNSFlags(
        nu=nu,
        dPdx=0.0, # No pressure gradient (no forcing)
        constraint=PressureGradient,
        baseflow=ParabolicBase,
        timestepping=CNRK2,
        dt=dt,
        T=0.1 # Total simulation time
    )

    # read the flowfield
    Nx, Ny, Nz = domain.Nx, domain.Ny, domain.Nz
    Lx, Lz = domain.Lx, domain.Lz
    a, b = domain.a, domain.b
    u = read_data(data_path, domain)
    p = FlowField(Nx, Ny, Nz, 1, Lx, Lz, a, b)
    make_spectral!(u)
    make_spectral!(p)
    fields = [u, p] # The list of fields to be advanced in time

    # Create the Navier-Stokes equation solver and the Runge-Kutta time-stepper
    nse = NSE(fields, flags)
    dns = RungeKuttaDNS(fields, nse, flags)

    println("Starting simulation...")
    @printf "%-10s %-20s\n" "Time" "L2Norm(u)"
    @printf "----------------------------------\n"

    # 3. RUN THE TIME-STEPPING LOOP
    # num_steps = round(Int, flags.T / flags.dt)
    num_steps = 10
    for n in 0:(num_steps-1)
        t = n * dt

        # Calculate and print the L2 Norm of the velocity field.
        # This represents the total kinetic energy of the perturbation.
        norm_u = L2Norm(u)
        @printf "%-10.4f %-20.12e\n" t norm_u
        # display(u)

        # If the norm becomes NaN or explodes, stop the simulation.
        if isnan(norm_u) || norm_u > 1e10
            println("\nERROR: Simulation became unstable!")
            break
        end

        # Advance the simulation by one time step
        advance!(dns, fields, 1)
    end

    println("\nSimulation finished.")
    println(domain)
    return L2Norm(u) # Return the final norm for testing purposes
end

# Run the example
eq7_continuation()

