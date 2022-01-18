module Optimize
include("ResComp.jl")
using LinearAlgebra
using Base.Threads
using PyCall
import Statistics

function find_vpt(untrained, r₀)
        train_tspan = (0.0, 100.0);
        test_tspan = (0.0, 100.0) .+ train_tspan[2]
        trained, train_sol = ResComp.train(untrained, r₀, train_tspan);
        test_sol = ResComp.test(trained, train_sol.u[end], test_tspan);
        return test_sol.t[end] - test_tspan[1];
end

function try_find_vpt(untrained, r₀)
        try
                return find_vpt(untrained, r₀);
        catch e
                if isa(e, LinearAlgebra.SingularException)
                        @warn "Could not solve least squares formulation"
                else
                        rethrow()
                end
        end;
end

function find_vpts(system, nₛ, γ, σ, ρ, α)
        nᵣ::Int = 500;

        vpts = zeros(50);
        @threads for i = 1:length(vpts)
                untrained = ResComp.initialize_rescomp(system, tanh, γ, σ, ρ, nᵣ, nₛ, α)
                r₀ = 2*rand(Float64, nᵣ).-0.5
                vpts[i] = try_find_vpt(untrained, r₀);
        end;
        return vpts
end;

function evaluate(system, nₛ, parameters)
        vpts = find_vpts(system, nₛ, parameters["gamma"], parameters["sigma"], parameters["rho"], parameters["alpha"]);
        return Statistics.mean(vpts), Statistics.std(vpts)
end;

function torch_rescomp(system,nₛ,num_trials)
        ax_client = PyCall.pyimport("ax.service.ax_client")
        client = ax_client.AxClient()
        params = "[
                {
                        'name': 'gamma',
                        'type': 'range',
                        'bounds': [0.000001, 10],
                        'value_type': 'float'
                },
                {
                        'name': 'sigma',
                        'type': 'range',
                        'bounds': [0.000001, 5.0],
                        'value_type': 'float'
                },
                {
                        'name': 'rho',
                        'type': 'range',
                        'bounds': [0.000001, 10],
                        'value_type': 'float'
                },
                {
                        'name': 'alpha',
                        'type': 'range',
                        'bounds': [0.0000001, 0.5],
                        'value_type': 'float'
                },
                ]";
        PyCall.py"$client.create_experiment(name='hello', parameters=$$params, objective_name='vpt', minimize=False)"

        for trial_index = 1:num_trials
                parameters, trial_index = client.get_next_trial()
                client.complete_trial(trial_index=trial_index, raw_data=evaluate(system, nₛ, parameters))
        end
        return client;
end;

end;