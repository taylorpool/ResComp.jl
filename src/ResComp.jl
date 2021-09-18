module ResComp
using DifferentialEquations
using LinearAlgebra
using Plots

export UntrainedResComp, TrainedResComp

struct UntrainedResComp
    Wᵢₙ::AbstractArray
    u
    A::AbstractArray
    f
    γ::Number
    σ::Number
    ρ::Number
    ϵ::Number
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
    ϵ::Number
end;

TrainedResComp(Wₒᵤₜ::AbstractArray, r::UntrainedResComp) = TrainedResComp(
                                                                          Wₒᵤₜ,
                                                                          r.Wᵢₙ,
                                                                          r.u,
                                                                          r.A,
                                                                          r.f,
                                                                          r.γ,
                                                                          r.σ,
                                                                          r.ρ,
                                                                         r.ϵ);

function drive!(dr, r::AbstractArray, rescomp::UntrainedResComp, t)
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r + rescomp.σ*rescomp.Wᵢₙ*rescomp.u(t)));
end;

function drive!(dr, r::AbstractArray, rescomp::TrainedResComp, t)
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r + rescomp.σ*rescomp.Wᵢₙ*rescomp.Wₒᵤₜ*r));
end;

function calculateTransientIndex(drive_sol, ϵ)
    i = 1
    while norm(drive_sol.u[i]) ≥ ϵ
            i = i + 1
    end
    return i;
end

function calculateOutputMapping(rescomp::UntrainedResComp, drive_sol)
        index = calculateTransientIndex(drive_sol, rescomp.ϵ);
        D = rescomp.u.(drive_sol.t[index:end]);
        R = hcat(drive_sol.u[index:end]...);
        Wₒᵤₜ = (R*R' \ R*D)';
        return Wₒᵤₜ;
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
        drive_prob = ODEProblem(drive!, r₀, tspan, rescomp);
        drive_sol = solve(drive_prob);
        Wₒᵤₜ = calculateOutputMapping(rescomp, drive_sol);
        return TrainedResComp(Wₒᵤₜ, rescomp), drive_sol;
end;

function test(rescomp::TrainedResComp, r₀::AbstractArray, tspan::Tuple{Float64, Float64})
        drive_prob = ODEProblem(drive!, r₀, tspan, rescomp);
        drive_sol = solve(drive_prob);
        return calculateValidPredictionTime(rescomp.u, drive_sol, rescomp.ϵ, rescomp.Wₒᵤₜ), drive_sol; 
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
