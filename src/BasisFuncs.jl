module BasisFuncs

struct BasisFuncDomain
    num_dimensions::Int
    Ny::Int
    kx::Int
    kz::Int
    Lx::Real
    Lz::Real
    a::Real
    b::Real
    function BasisFuncDomain(num_dimensions::Int, Ny::Int, kx::Int, kz::Int, Lx::Real, Lz::Real, a::Real, b::Real)
        @assert num_dimensions > 0 "The number of dimensions must be positive"
        @assert kx > 0 "There must be at least 1 x mode"
        @assert Ny > 0 "There must be at least 1 y gridpoint"
        @assert kz > 0 "There must be at least 1 z mode"

        new(num_dimensions, Ny, kx, kz, Lx, Lz, a, b)
    end
end

function geom_congruent(d1::BasisFuncDomain, d2::BasisFuncDomain)
    return d1.num_dimensions == d2.num_dimensions &&
           d1.Ny == d2.Ny &&
           d1.kx == d2.kx &&
           d1.kz == d2.kz &&
           d1.Lx == d2.Lx &&
           d1.Lz == d2.Lz &&
           d1.a == d2.a &&
           d1.b == d2.a
end
struct BasisFunc
    domain::BasisFuncDomain
    state::FieldState
    functions::AbstractArray{ChebyCoeff}
end

# ===============
# Constructors
# ===============


"""
	BasisFunc()

Empty (default) constructor for BasisFunc. 
Initializes all values to 0. Should then call `resize` or `reconfigure`.
"""
function BasisFunc()
    domain = BasisFuncDomain(0, 0, 0, 0, 0, 0, 0, 0)
    BasisFunc(domain, Spectral, [])
end

"""
	BasisFunc(Ny, f)

Creates a basis function related to another BasisFunc `f`. 
"""
function BasisFunc(Ny::Int, f::BasisFunc)
    domain = BasisFuncDomain(f.domain.num_dimensions, Ny, f.domain.kx, f.domain.kz, f.domain.Lx, f.domain.Lz, f.domain.a, f.domain.b)
    state = f.state
    u = Array{ChebyCoeff,domain.num_dimensions}

    BasisFunc(domain, state, u)
end


"""
3D vector field from existing component functions.
"""
function BasisFunc(u::ChebyCoeff{T}, v::ChebyCoeff{T}, w::ChebyCoeff{T}, kx::Real, kz::Real, Lx::Real, Lz::Real) where {T<:Complex}
    num_dimensions = 3 # velocity field 
    a, b = bounds(u)
    Ny = num_modes(u)
    state = state(u)
    domain = BasisFuncDomain(num_dimensions, Ny, kx, kz, Lx, Lz, a, b)
    funcs = [u, v, w]

    BasisFunc(domain, state, funcs)
end

# ======================
# Geometry Comparsions
# ======================

function geom_congruent(f1::BasisFunc, f2::BasisFunc)
    return geom_congruent(f1.domain, f2.domain)
end

function congruent(f1::BasisFunc, f2::BasisFunc)
    return geom_congruent(f1, f2) && f1.domain.kx == f2.domain.kx && f1.domain.kz == f2.domain.kz
end

function is_interoperable(f::BasisFunc, g::BasisFunc)
    return congruent(f, g) && f.state == g.state
end

# ======================
# Accessors
# ======================

get_u(f::BasisFunc) = f.functions[1]
get_v(f::BasisFunc) = f.functions[2]
get_w(f::BasisFunc) = f.functions[3]

function Base.getindex(f::BasisFunc, i::Int)
    @assert i >= 1 "The index to BasisFunc must be 1 or greater"
    @assert i <= f.domain.num_dimensions "The index to BasisFunc must be less than or equal to the number of dimensions."
    return f.functions[i]
end

function Base.:(==)(f::BasisFunc, g::BasisFunc)
    if !congruent(f, g)
        return false
    end

    for i in eachindex(f.functions)
        if f[i] != g[i]
            return false
        end
    end
    return true
end

function Base.:*(f::BasisFunc, g::BasisFunc)
    @assert geom_congruent(f, g) "BasisFuncs must be geometrically congruent"
    @assert f.state == Physical && g.state == Physical "BasisFuncs must be physical to multiply"

    for i in eachindex(f)
        f[i] *= g[i]
    end

    f.domain.kx += g.domain.kx
    f.domain.kz += g.domain.kz

    return f
end

function Base.:*(f::BasisFunc, c::Number)
    for i in eachindex(f)
        f[i] *= c
    end
    return f
end

function Base.:*(c::Number, f::BasisFunc)
    return f * c
end

function Base.:+(f::BasisFunc, g::BasisFunc)
    @assert is_interoperable(f, g)
    for i in eachindex(f)
        f[i] += g[i]
    end
end

end
