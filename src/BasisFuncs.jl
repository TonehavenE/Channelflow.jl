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
end
