module Optimize
include("ResComp.jl")
using LinearAlgebra
using Base.Threads
using PyCall
import Statistics

function find_vpt(untrained, r₀, parameters)
        train_tspan = (0.0, 100.0);
        test_tspan = (0.0, 100.0) .+ train_tspan[2]
        if parameters["experiment_params"]["windows"]
                trained, train_sol = ResComp.train_windows(untrained, r₀, train_tspan, parameters["num_windows"])
        else
                trained, train_sol = ResComp.train(untrained, r₀, train_tspan);
        end
        test_sol = ResComp.test(trained, train_sol.u[end], test_tspan);
        return test_sol.t[end] - test_tspan[1];
end

function try_find_vpt(untrained, r₀, parameters)
        try
                return find_vpt(untrained, r₀, parameters);
        catch e
                if isa(e, LinearAlgebra.SingularException)
                        @warn "Could not solve least squares formulation"
                else
                        rethrow()
                end
        end
end

function find_vpts(parameters)
        nᵣ = parameters["reservoir_dimension"];
        system = parameters["system"]
        f = tanh
        γ = parameters["gamma"]
        σ = parameters["sigma"]
        ρ = parameters["rho"]
        system_dimension = parameters["system_dimension"]
        α = parameters["alpha"]

        vpts = zeros(50);
        @threads for i = 1:length(vpts)
                untrained = ResComp.initialize_rescomp(
                        system,
                        f,
                        γ,
                        σ,
                        ρ,
                        nᵣ,
                        system_dimension,
                        α)
                r₀ = 2*rand(Float64, nᵣ).-0.5
                vpts[i] = try_find_vpt(untrained, r₀, parameters);
        end;
        return vpts
end;

function evaluate(parameters)
        vpts = find_vpts(parameters)
        return Statistics.mean(vpts), Statistics.std(vpts)
end;

end;