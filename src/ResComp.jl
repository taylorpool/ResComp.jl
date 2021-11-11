module ResComp
using DifferentialEquations
using LinearAlgebra

export UntrainedResComp, TrainedResComp, train, test

struct UntrainedResComp{T<:Real}
    Wᵢₙ
    u
    A
    f
    γ::T
    σ::T
    ρ::T
    α::T
end;

function initialize_rescomp(u, f, γ::T, σ::T, ρ::T, nᵣ::Int, nₛ::Int, α::T) where {T<:Real}
        A = rand(T, (nᵣ, nᵣ)).-0.5;
        Wᵢₙ = rand(T, (nᵣ,nₛ)).-0.5;
        return ResComp.UntrainedResComp(Wᵢₙ./opnorm(Wᵢₙ), u, A./opnorm(A), f, γ, σ, ρ, α);
end;

struct TrainedResComp{T<:Real}
    Wₒᵤₜ
    Wᵢₙ
    u
    A
    f
    γ::T
    σ::T
    ρ::T
    α::T
end;

TrainedResComp(Wₒᵤₜ, r::UntrainedResComp{T}) where {T<:Real} = TrainedResComp(Wₒᵤₜ, r.Wᵢₙ, r.u, r.A, r.f, r.γ, r.σ, r.ρ, r.α);

function drive_transient!(dr::V, r::V, rescomp::UntrainedResComp{T}, t) where {T<:Real,V<:AbstractVector{T}}
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r));
end;

function drive!(dr, r, rescomp::UntrainedResComp{T}, t) where {T<:Real}
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r + rescomp.σ*rescomp.Wᵢₙ*rescomp.u(t)));
end;

function drive!(dr, r, rescomp::TrainedResComp{T}, t) where {T<:Real}
        dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r + rescomp.σ*rescomp.Wᵢₙ*rescomp.Wₒᵤₜ*r));
end;

function find_transient_time(rescomp::UntrainedResComp{T}, r₀, tspan::Tuple{Float64, Float64}) where {T<:Real}
    transient_prob = ODEProblem(drive_transient!, r₀, tspan, rescomp);
    condition(r, t, integrator) = norm(r,2) < 0.01;
    affect!(integrator) = terminate!(integrator);
    transient_cb = DiscreteCallback(condition, affect!);
    transient_sol = solve(transient_prob, callback=transient_cb);
    return transient_sol.t[end];
end;

function calculateOutputMapping(rescomp::UntrainedResComp{T}, drive_sol, transient_time::Float64) where {T<:Real}
        transient_mask = drive_sol.t .> transient_time;
        D = hcat(rescomp.u.(drive_sol.t[transient_mask])...)';
        R = drive_sol[:, transient_mask];
        return ((R*R'.-rescomp.α*rescomp.α) \ R*D)';
end;

function train(rescomp::UntrainedResComp{T}, r₀, tspan::Tuple{Float64, Float64}) where {T<:Real}
        transient_time = find_transient_time(rescomp, r₀, tspan);
        drive_prob = ODEProblem(drive!, r₀, (tspan[1], tspan[2]+transient_time), rescomp);
        drive_sol = solve(drive_prob);
        Wₒᵤₜ = calculateOutputMapping(rescomp, drive_sol, transient_time);
        return TrainedResComp(Wₒᵤₜ, rescomp), drive_sol;
end;

function test(rescomp::TrainedResComp{T}, r₀, tspan::Tuple{Float64, Float64}) where {T<:Real}
        drive_prob = ODEProblem(drive!, r₀, tspan, rescomp);
        condition(r, t, integrator) = norm(rescomp.Wₒᵤₜ*r - rescomp.u(t), 2) > 0.01;
        affect!(integrator) = terminate!(integrator);
        vpt_cb = DiscreteCallback(condition, affect!);
        drive_sol = solve(drive_prob, callback=vpt_cb);
        return drive_sol; 
end;

end