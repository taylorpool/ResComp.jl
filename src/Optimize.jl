module Optimize
include("ResComp.jl")
using LinearAlgebra
using Base.Threads
using PyCall
import Statistics

function try_find_vpt(parameters, vpt_function)
        try
                return vpt_function(parameters)
        catch e
                if isa(e, LinearAlgebra.SingularException)
                        @warn "Could not solve least squares formulation"
                else
                        rethrow()
                end
        end
end

function find_vpts(parameters, vpt_function)
        vpts = zeros(parameters["experiment_params"]["num_samples_per_trial"])
        @threads for i = 1:length(vpts)
                vpts[i] = try_find_vpt(parameters, vpt_function)
        end
        return vpts
end

function evaluate(parameters, vpt_function)
        vpts = find_vpts(parameters, vpt_function)
        return Statistics.mean(vpts), Statistics.std(vpts)
end

end