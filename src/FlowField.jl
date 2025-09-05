module FlowFields

#=
Puts together all of the FlowField source files into one coherent module.
=#

import ..ChebyCoeffs: congruent, num_modes
using ..ChebyCoeffs

include("FlowField/FlowFieldDomain.jl")
include("FlowField/FlowFieldTransforms.jl")
include("FlowField/types_and_constructors.jl")
include("FlowField/accessors.jl")
include("FlowField/base_ops.jl")
include("FlowField/transforms.jl")
include("FlowField/utilities.jl")
include("FlowField/geometry.jl")
include("FlowField/calculus.jl")
include("FlowField/read_write.jl")

end
