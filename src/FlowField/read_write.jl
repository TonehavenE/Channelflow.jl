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
    Nx, Ny, Nz = domain.Nx, domain.Ny, domain.Nz
    ff = FlowField(domain)
    make_physical!(ff)
    line_count = 0
    for line in readlines(io)

        i = line_count % 3
        nz = (line_count ÷ 3) % Nz
        ny = (line_count ÷ (3 * Nz)) % Ny
        nx = (line_count ÷ (3 * Nz * Ny)) % Nx
        ff[nx+1, ny+1, nz+1, i+1] = parse(Float64, line)

        line_count += 1
    end
    ff
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
