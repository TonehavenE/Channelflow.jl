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

# ================
# Mutators
# ================
#
"""
	set_state!(f, state)

Set's the state of `f` to `state`. Also set's each of its functions.
"""
function set_state!(f::BasisFunc, state::FieldState)
    @assert state == Spectral || state == Physical "State must be either physical or spectral."
    for i = 1:f.domain.num_dimensions
        set_state!(f.functions[i], state)
    end
    f.state = state
end

"""
	set_bounds!(f, Lx, Lz, a, b)

Set `f` to have a domain size of Lx, Lz, a, and b.
Also sets those bounds to each of the vector component ChebyCoeffs.
"""
function set_bounds!(f::BasisFunc, Lx::Real, Lz::Real, a::Real, b::Real)
    f.domain.Lx = Lx
    f.domain.Lz = Lz
    f.domain.a = a
    f.domain.b = b
    for i in eachindex(f.functions)
        set_bounds!(f.functions[i], a, b)
    end
end

"""
	set_kx_kz!(f, kz, kx)

Set `f` domain to kx and kz.
"""
function set_kx_kz!(f::BasisFunc, kz::Real, kx::Real)
    f.domain.kx = kx
    f.domain.kz = kz
end

"""
	set_to_zero!(f)

Sets all of `f`'s functions to 0.
"""
function set_to_zero!(f::BasisFunc)
    for func in f.functions
        set_to_zero!(func)
    end
end

"""
	resize!(f, Ny, Nd)

Resizes `f` to a domain length of `Ny` and a dimensionality of `Nd`.
"""
function Base.resize!(f::BasisFunc, Ny::Int, Nd::Int)
    if f.domain.num_dimensions != Nd
        f.domain.num_dimensions = Nd
        Base.resize!(f.functions, Nd)
    end
    f.domain.Ny = Ny
    for i = 1:f.domain.num_dimensions
        resize!(f[i], Ny)
    end
end

"""
	reconfig!(f, g)

Reconfigures `f` in place to have the same properties as `g`.
Clears `f`'s functions.
"""
function reconfig!(f::BasisFunc, g::BasisFunc)
    resize!(f, g.domain.Ny, g.domain.num_dimensions)
    set_bounds!(f, g.domain.Lx, g.domain.Lz, g.domain.a, g.domain.b)
    set_kx_kz!(f, g.domain.kx, g.domain.kz)
    set_state!(f, g.state)
    set_to_zero!(f)
end

# ======================
# Transforms
# ======================
function chebyfft!(f::BasisFunc, t::Union{ChebyTransform,Nothing}=nothing)
    @assert f.state == Physical "starting state must be physical"
    if isnothing(t)
        t = ChebyTransform(f.domain.Ny)
    end
    for i in eachindex(f.functions)
        chebyfft!(f.functions[i], t)
    end
    f.state = Spectral
end

function ichebyfft!(f::BasisFunc, t::Union{ChebyTransform,Nothing}=nothing)
    @assert f.state == Spectral "starting state must be spectral"
    if isnothing(t)
        t = ChebyTransform(f.domain.Ny)
    end
    for i in eachindex(f.functions)
        ichebyfft!(f.functions[i], t)
    end
    f.state = physical
end

function make_spectral!(f::BasisFunc, t::Union{ChebyTransform,Nothing}=nothing)
    if f.state == Spectral
        return
    end
    for i in eachindex(f.functions)
        make_spectral!(f.functions[i], t)
    end
    f.state = Spectral
end

function make_physical!(f::BasisFunc, t::Union{ChebyTransform,Nothing}=nothing)
    if f.state == Physical
        return
    end
    for i in eachindex(f.functions)
        make_physical!(f.functions[i], t)
    end
    f.state = Physical
end

function make_state!(f::BasisFunc, s::FieldState, t::Union{ChebyTransform,Nothing}=nothing)
    for i in eachindex(f.functions)
        make_state!(f.functions[i], s, t)
    end
end

# ======================
# Norms
# ======================
# TODO: Implement norms (just callouts to ChebyCoeff ones)
# should this be in a separate file? probably


# ======================
# Derivative Operators
# ======================

"""
	xdiff!(f, dfdx)

Calculates the x-derivative of `f` and stores it in `dfdx`.
"""
function xdiff!(f::BasisFunc, dfdx::BasisFunc)
    @assert geom_congruent(f, dfdx) "the destination must be geometrically congruent to the function"
    dfdx.functions = f.functions
    dfdx.state = f.state
    dfdx *= Complex(0.0, 2pi * f.domain.kx / f.domain.Lx)
end

function xdiff(f::BasisFunc)
    dfdx = BasisFunc(f.domain.Ny, f)
    xdiff!(f, dfdx)
    return dfdx
end

"""
	ydiff!(f, dfdz)

Calculates the y-derivative of `f` and stores it in `dfdy`.
"""
function ydiff!(f::BasisFunc, dfdy::BasisFunc)
    @assert geom_congruent(f, dfdy) "the destination must be geometrically congruent to the function"
    dfdy.functions = f.functions
    dfdy.state = f.state
    dfdy *= Complex(0.0, 2pi * f.domain.kz / f.domain.Lz)
end

function ydiff(f::BasisFunc)
    dfdy = BasisFunc(f.domain.Ny, f)
    ydiff!(f, dfdy)
    return dfdy
end

"""
	zdiff!(f, dfdz)
Calculates the z-derivative of `f` and stores it in `dfdz`.
"""
function zdiff!(f::BasisFunc, dfdz::BasisFunc)
    @assert geom_congruent(f, dfdz) "the destination must be geometrically congruent to the function"
    dfdz.functions = f.functions
    dfdz.state = f.state
    dfdz *= Complex(0.0, 2pi * f.domain.kz / f.domain.Lz)
end

function zdiff(f::BasisFunc)
    dfdz = BasisFunc(f.domain.Ny, f)
    zdiff!(f, dfdz)
    return dfdz
end

"""
	divergence!(f, divf)

Calculates the divergence of `f` and stores it in `divf`.
"""
function divergence!(f::BasisFunc, divf::BasisFunc)
    @assert f.state == Spectral
    @assert f.domain.num_dimensions == 3
    resize!(divf, f.domain.Ny, 1)
    set_bounds!(divf, f.domain.Lx, f.domain.Lz, f.domain.a, f.domain.b)
    set_kx_kz!(divf, f.domain.kx, f.domain.kz)

    tmp = ChebyCoeff(f[1])
    tmp *= Complex(0, 2pi * f.domain.kx / f.domain.Lx)
    divf[1] = tmp

    tmp = derivative(f[2])
    divf[1] += tmp

    tmp = ChebyCoeff(f[2])
    tmp *= Complex(0.0, 2pi * f.domain.kz / f.domain.Lz)
    divf[1] += tmp
end

function divergence(f::BasisFunc)
    divf = BasisFunc(f.domain.Ny, f)
    divergence!(f, divf)
    return divf
end

"""
	laplacian!(f, laplf)

Calculates the Laplacian of `f` and stores it inplace in `laplf`.
"""
function laplacian!(f::BasisFunc, laplf::BasisFunc)
    @assert f.state == Spectral

    # make laplf's functions the same as f
    for i in eachindex(f.functions)
        laplf[i] = ChebyCoeff(f[i])
    end

    c = -4pi^2 * ((f.domain.kx / f.domain.Lx)^2 + (f.domain.kz / f.domain.Lz)^2)

    for i in eachindex(f.functions)
        laplf[i] *= c
        tmp = derivative2(f[i])
        laplf[i] += tmp
    end
end

"""
	laplacian(f)

Returns a new BasisFunc equal to the Laplacian of `f`.
"""
function laplacian(f::BasisFunc)
    laplf = BasisFunc(f.domain.Ny, f)
    laplacian!(f, laplf)
    return laplf
end

# ======================
# Fix Boundary Conditions
# ======================

"""
	ubc_fix!(u, a_bc, b_bc)

Impose the boundary conditions at the bounds on `u`.
Modifies in place. 
"""
function ubc_fix!(u::ChebyCoeff, a_bc::BC, b_bc::BC)
    ua, ub = bounds(u)
    set_bounds!(u, -1.0, 1.0)

    if a_bc == Diri && b_bc == Diri
        u_at_a, u_at_b = eval_a(u), eval_b(u)
        u[1] -= 0.5 * (u_at_a + u_at_b)
        u[2] -= 0.5 * (u_at_a - u_at_b)
    elseif a_bc == Diri
        u[1] -= eval_a(u)
    elseif b_bc == Diri
        u[2] -= eval_b(u)
    end

    set_bounds!(u, ua, ub)
end

"""
	vbc_fix!(v, a_bc, b_bc)

Impose the boundary conditions at the bounds on `v`.
Modifies in place. 
"""
function vbc_fix!(v::ChebyCoeff, a_bc::BC, b_bc::BC)
    va, vb = bounds(v)
    set_bounds!(v, -1.0, 1.0)

    if a_bc == Diri && b_bc == Diri
        dvdy = derivative(v)
        a = eval_a(v)
        b = eval_b(v)
        c = eval_a(dvdy)
        d = eval_b(dvdy)

        v[1] -= 0.5 * (a + b) + 0.125 * (c - d)
        v[2] -= 0.5625 * (b - a) - 0.0625 * (c + d)
        v[3] -= 0.125 * (d - c)
        v[4] -= 0.0625 * (a - b + c + d)
    elseif a_bc == Diri
        dvdy = derivative(v)
        v[1] -= eval_a(v)
        v[2] -= eval_a(dvdy)
    elseif b_bc == Diri
        dvdy = derivative(v)
        v[1] -= eval_b(v)
        v[2] -= eval_b(dvdy)
    end
    set_bounds!(v, va, vb)
end
end
