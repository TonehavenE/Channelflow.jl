module ChannelflowCUDAExt

using Channelflow
using CUDA

function __init__()
    Channelflow.NSolver.register_gpu_backend!(;
        to_gpu = x -> x isa CUDA.AbstractCuArray ? x : CUDA.CuArray(x),
        synchronize = () -> CUDA.synchronize(),
    )
end

end
