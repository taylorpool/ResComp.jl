module ResComp
using Logging
using DifferentialEquations
using LinearAlgebra
using Plots

export UntrainedResComp, TrainedResComp, train, test

struct UntrainedResComp
    Wᵢₙ::AbstractArray
    u
    A::AbstractArray
    f
    γ::Number
    σ::Number
    ρ::Number
end;

struct TrainedResComp
    Wₒᵤₜ::AbstractArray
    Wᵢₙ::AbstractArray
    u
    A::AbstractArray
    f
    γ::Number
    σ::Number
    ρ::Number
end;

TrainedResComp(Wₒᵤₜ::AbstractArray, r::UntrainedResComp) = TrainedResComp(
                                                                          Wₒᵤₜ,
                                                                          r.Wᵢₙ,
                                                                          r.u,
                                                                          r.A,
                                                                          r.f,
                                                                          r.γ,
                                                                          r.σ,
                                                                          r.ρ)

function drive_transient!(dr, r::AbstractArray, rescomp::Union::{UntrainedResComp, TrainedResComp}, t)
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r))
end;

function drive!(dr, r::AbstractArray, rescomp::UntrainedResComp, t)
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r + rescomp.σ*rescomp.Wᵢₙ*rescomp.u(t)));
end;

function drive!(dr, r::AbstractArray, rescomp::TrainedResComp, t)
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r + rescomp.σ*rescomp.Wᵢₙ*rescomp.Wₒᵤₜ*r));
end;

function find_transient_time(rescomp::UntrainedResComp, r₀, tspan::Tuple{Float64, Float64})
    transient_prob = ODEProblem(transient!, r₀, tspan, rescomp);
    condition(r, t, integrator) = norm(r,2) < 0.01;
    affect!(integrator) = terminate!(integrator);
    transient_cb = DiscreteCallback(condition, affect!);
    transient_sol = solve(transient_prob, callback=transient_cb);
    return transient_sol.t[end];
end;

function calculateOutputMapping(rescomp::UntrainedResComp, drive_sol, transient_time)
        transient_mask = drive_sol.t .> transient_time;
        D = rescomp.u.(drive_sol.t[transient_mask]);
        R = drive_sol[:, transient_mask];
        return (R*R' \ R*D)';
end;

function calculateValidPredictionTime(true_fun, rescomp_pred, ϵ, W)
        true_prediction = true_fun.(rescomp_pred.t);
        i = 1;
        while norm(true_fun(rescomp_pred.t[i]) - W*rescomp_pred.u[i]) < ϵ
                i += 1;
        end;
        return rescomp_pred.t[i] - rescomp_pred.t[1];
end

function train(rescomp::UntrainedResComp, r₀::AbstractArray, tspan::Tuple{Float64, Float64})
        transient_time = find_transient_time(rescomp, r₀, tspan);
        drive_prob = ODEProblem(drive!, r₀, (tspan[1], tspan[2]+transient_time), rescomp);
        drive_sol = solve(drive_prob);
        Wₒᵤₜ = calculateOutputMapping(rescomp, drive_sol, transient_time);
        return TrainedResComp(Wₒᵤₜ, rescomp), drive_sol;
end;

function test(rescomp::TrainedResComp, r₀::AbstractArray, tspan::Tuple{Float64, Float64})
        drive_prob = ODEProblem(drive!, r₀, tspan, rescomp);
        condition(r, t, integrator) = norm(rescomp.Wₒᵤₜ*r - rescomp.u(t), 2) > 0.01;
        affect!(integrator) = terminate!(integrator);
        vpt_cb = DiscreteCallback(condition, affect!);
        drive_sol = solve(drive_prob, callback=vpt_cb);
        return drive_sol; 
end;

function plotResults(train_sol, test_sol, rescomp::TrainedResComp)
    train_nodes = hcat(train_sol.u...)';
    test_nodes = hcat(test_sol.u...)';
    test_outputs = test_nodes*rescomp.Wₒᵤₜ';
    train_node_plot = plot(train_sol, title="Training Nodes")
    test_node_plot = plot(test_sol, title="Test Nodes")
    test_output_plot = plot(test_sol.t, test_outputs, title="Test Output")
    plot(train_node_plot, test_node_plot, test_output_plot, layout=(1,3), legend=false)
end
        
end
