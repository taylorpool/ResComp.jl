module Optimize
include("ResComp.jl")
using LinearAlgebra
using Base.Threads
using PyCall
import Statistics

function find_vpt(untrained, r₀, parameters)
        train_time = parameters["experiment_params"]["train_time"]
        train_tspan = (0.0, train_time);
        if parameters["experiment_params"]["windows"]
                trained, train_sol = ResComp.train_windows(untrained, r₀, train_tspan, parameters["num_windows"])
        else
                trained, train_sol = ResComp.train(untrained, r₀, train_tspan);
        end
        test_time = parameters["experiment_params"]["test_time"]
        if parameters["experiment_params"]["random_initial_condition"]
                system_duration = parameters["experiment_params"]["system_duration"]
                random_initial_time = train_time + rand()*(system_duration-test_time-train_time)
                test_tspan = (0.0, parameters["experiment_params"]["test_time"]) .+ random_initial_time
                test_sol = ResComp.test(trained, train_sol.u[end], test_tspan, parameters["system"]);
        else
                test_tspan = (0.0, test_time) .+ train_tspan[2]
                test_sol = ResComp.test(trained, train_sol.u[end], test_tspan, parameters["system"]);
        end
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
        reservoir_dimension = parameters["experiment_params"]["reservoir_dimension"];
        system = parameters["system"]
        f = tanh
        gamma = parameters["gamma"]
        sigma = parameters["sigma"]
        rho = parameters["rho"]
        system_dimension = parameters["experiment_params"]["system_dimension"]
        alpha = parameters["alpha"]

        vpts = zeros(parameters["experiment_params"]["num_samples_per_trial"]);
        @threads for i = 1:length(vpts)
                untrained = ResComp.initialize_rescomp(
                        system,
                        f,
                        gamma,
                        sigma,
                        rho,
                        reservoir_dimension,
                        system_dimension,
                        alpha)
                r₀ = 2*rand(Float64, reservoir_dimension).-0.5
                vpts[i] = try_find_vpt(untrained, r₀, parameters);
        end;
        return vpts
end;

function evaluate(parameters)
        vpts = find_vpts(parameters)
        return Statistics.mean(vpts), Statistics.std(vpts)
end;

end;