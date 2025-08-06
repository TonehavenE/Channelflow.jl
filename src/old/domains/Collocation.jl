module Collocation

function get_derivative_matrix end

include("./grid.jl")
include("./chebyshev.jl")
include("./fourier.jl")

using .Chebyshev
using .Grid
using .Fourier

export ChebyshevGrid, FourierGrid, fourier_derivative_matrix, chebyshev_derivative_matrix, AbstractGrid, get_derivative_matrix

end
