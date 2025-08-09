module Metrics

#=
Define generics to be implemented. Prevents namespace clashing.
=#

function L2Norm2 end
function L2Norm end
function L2InnerProduct end

function chebyNorm2 end
function chebyNorm end
function chebyInnerProduct end

include("Metrics/cheby_metrics.jl")
include("Metrics/flow_metrics.jl")

export L2Norm,
    L2Norm2, L2InnerProduct, chebyNorm, chebyNorm2, chebyInnerProduct, LinfNorm, L1Norm
end