module ResComp
using DifferentialEquations
using LinearAlgebra

export UntrainedResComp, TrainedResComp, train, test

struct UntrainedResComp
    u 
    Wᵢₙ
    A
    f
    γ
    σ
    ρ
    α
end;

function initialize_rescomp(u, f, γ, σ, ρ, nᵣ::Int, nₛ::Int, α)
        A = rand(Float64, (nᵣ, nᵣ)).-0.5;
        Wᵢₙ = rand(Float64, (nᵣ,nₛ)).-0.5;
        return ResComp.UntrainedResComp(u, Wᵢₙ./opnorm(Wᵢₙ), A./opnorm(A), f, γ, σ, ρ, α);
end;

struct TrainedResComp
    Wₒᵤₜ
    Wᵢₙ
    A
    f
    γ
    σ
    ρ
    α
end;

TrainedResComp(Wₒᵤₜ, r::UntrainedResComp) = TrainedResComp(Wₒᵤₜ, r.Wᵢₙ, r.A, r.f, r.γ, r.σ, r.ρ, r.α);

function drive_transient!(dr, r, rescomp::UntrainedResComp, t)
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r));
end;

function drive!(dr, r, rescomp, t)
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r + rescomp.σ*rescomp.Wᵢₙ*rescomp.u(t)));
end;

function drive!(dr, r, rescomp, t)
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r + rescomp.σ*rescomp.Wᵢₙ*rescomp.Wₒᵤₜ*r));
end;

function find_transient_time(rescomp, r₀, tspan::Tuple{T, T}) where {T<:Real}
    transient_prob = ODEProblem(drive_transient!, r₀, tspan, rescomp);
    condition(r, t, integrator) = norm(r,2) < 0.01;
    affect!(integrator) = terminate!(integrator);
    transient_cb = DiscreteCallback(condition, affect!);
    transient_sol = solve(transient_prob, callback=transient_cb);
    return transient_sol.t[end];
end;

function calculateOutputMapping(rescomp::UntrainedResComp, drive_sol, transient_time::T) where {T<:Real}
        transient_mask = drive_sol.t .> transient_time;
        D = hcat(rescomp.u.(drive_sol.t[transient_mask])...)';
        R = drive_sol[:, transient_mask];
        return ((R*R'.-rescomp.α*rescomp.α) \ R*D)';
end;

function train(rescomp::UntrainedResComp, r₀, tspan::Tuple{T, T}) where {T<:Real}
        transient_time = find_transient_time(rescomp, r₀, tspan);
        drive_prob = ODEProblem(drive!, r₀, (tspan[1], tspan[2]+transient_time), rescomp);
        drive_sol = solve(drive_prob);
        Wₒᵤₜ = calculateOutputMapping(rescomp, drive_sol, transient_time);
        return TrainedResComp(Wₒᵤₜ, rescomp), drive_sol;
end;

function train_windows(rescomp::UntrainedResComp, r₀, tspan::Tuple{T, T}, num_windows::Int) where {T<:Real}
        transient_time = find_transient_time(rescomp, r₀, tspan);
        drive_prob = ODEProblem(drive!, r₀, (tspan[1], tspan[2]+transient_time), rescomp)
        affect!(integrator) = integrator.u = integrator.p.Wᵢₙ*integrator.p.u(integrator.t)
        callback = PeriodicCallback(affect!, (tspan[2]-tspan[1])/num_windows)
        drive_sol = solve(drive_prob, callback=callback)
        W_out = calculateOutputMapping(rescomp, drive_sol, transient_time);
        return TrainedResComp(W_out, rescomp), drive_sol
end

function test(rescomp::TrainedResComp, r₀, tspan::Tuple{T, T}) where {T<:Real}
        drive_prob = ODEProblem(drive!, r₀, tspan, rescomp);
        condition(r, t, integrator) = norm(rescomp.Wₒᵤₜ*r - rescomp.u(t), 2) > 0.01;
        affect!(integrator) = terminate!(integrator);
        vpt_cb = DiscreteCallback(condition, affect!);
        drive_sol = solve(drive_prob, callback=vpt_cb);
        return drive_sol; 
end;

end