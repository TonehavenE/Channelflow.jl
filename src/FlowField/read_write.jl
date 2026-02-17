#=
Defines operations for reading and writing data about FlowFields. 
=#

export read_geom, read_data

"""
	read_geom(io)

Constructs a FlowFieldDomain from IO.
"""
function read_geom(io::IOStream)::FlowFieldDomain
    # Have to make this strings for ease of conversion
    value_map = Dict(
        "Nx" => nothing,
        "Ny" => nothing,
        "Nz" => nothing,
        "Nd" => nothing,
        "Lx" => nothing,
        "Lz" => nothing,
        "a" => "-1.0",
        "b" => "1.0",
    )
    for line in readlines(io)
        words = split(line, "%"; limit=2)
        if length(words) == 2
            value, label = words[1], words[2]
            if label in keys(value_map)
                value_map[label] = value
            end
        else
            println("Error reading geom file: line doesn't contain label!")
        end
    end

    FlowFieldDomain(
        parse(Int, value_map["Nx"]),
        parse(Int, value_map["Ny"]),
        parse(Int, value_map["Nz"]),
        parse(Int, value_map["Nd"]),
        parse(Float64, value_map["Lx"]),
        parse(Float64, value_map["Lz"]),
        parse(Float64, value_map["a"]),
        parse(Float64, value_map["b"]),
    )
end

"""
	read_geom(file_path)

Constructs a FlowFieldDomain from a file path as a string.
"""
function read_geom(file_path::String)::FlowFieldDomain
    open(file_path, "r") do io
        domain = read_geom(io)
        return domain
    end
end

"""
	read_data(io, domain)

Constructs a FlowField object from a IOStream and a given domain.
"""
function read_data(io::IOStream, domain::FlowFieldDomain)
    # Detect file format by first non-empty line:
    # - legacy Julia format: one physical scalar per line
    # - ChannelFlow ASCII format: header lines starting with '%' and spectral Re/Im pairs
    first_payload = nothing
    for line in eachline(io)
        t = strip(line)
        isempty(t) && continue
        first_payload = t
        break
    end
    seekstart(io)

    if isnothing(first_payload)
        ff = FlowField(domain)
        set_to_zero!(ff)
        return ff
    elseif startswith(first_payload, "%")
        return _read_channelflow_ascii(io, domain)
    else
        return _read_legacy_physical_ascii(io, domain)
    end
end

function _read_legacy_physical_ascii(io::IOStream, domain::FlowFieldDomain)
    Nx, Ny, Nz = domain.Nx, domain.Ny, domain.Nz
    ff = FlowField(domain)
    make_physical!(ff)
    line_count = 0
    for line in eachline(io)
        t = strip(line)
        isempty(t) && continue

        i = line_count % domain.num_dimensions
        nz = (line_count ÷ domain.num_dimensions) % Nz
        ny = (line_count ÷ (domain.num_dimensions * Nz)) % Ny
        nx = (line_count ÷ (domain.num_dimensions * Nz * Ny)) % Nx
        ff[nx+1, ny+1, nz+1, i+1] = parse(Float64, t)

        line_count += 1
    end
    return ff
end

function _read_channelflow_ascii(io::IOStream, domain::FlowFieldDomain)
    ff = FlowField(domain; xz_state=Spectral, y_state=Spectral)
    set_to_zero!(ff)

    ncoeff = 0
    nmodes = domain.Mz * domain.Mx * domain.My * domain.num_dimensions

    for line in eachline(io)
        t = strip(line)
        (isempty(t) || startswith(t, "%")) && continue
        vals = split(t)
        length(vals) < 2 && continue

        mz = ncoeff % domain.Mz
        mx = (ncoeff ÷ domain.Mz) % domain.Mx
        ny = (ncoeff ÷ (domain.Mz * domain.Mx)) % domain.My
        i = (ncoeff ÷ (domain.Mz * domain.Mx * domain.My)) % domain.num_dimensions

        ff.spectral_data[mx+1, ny+1, mz+1, i+1] = Complex(parse(Float64, vals[1]), parse(Float64, vals[2]))
        ncoeff += 1
    end

    @assert ncoeff == nmodes "ChannelFlow ASCII size mismatch: read $ncoeff coefficients, expected $nmodes"
    return ff
end

"""
	read_data(file_path, domain)

Constructs a FlowField object from a file path and a given domain.
"""
function read_data(file_path::String, domain::FlowFieldDomain)
    open(file_path, "r") do io
        return read_data(io, domain)
    end
end
