module ResComp
using DifferentialEquations
using LinearAlgebra

export UntrainedResComp, TrainedResComp, train, test

struct UntrainedResComp
    u 
    W_in
    A
    f
    gamma
    sigma
    rho
    alpha
end;

function initialize_rescomp(u, f, gamma, sigma, rho, reservoir_dimension::Int, system_dimension::Int, alpha)
        A = rand(Float64, (reservoir_dimension, reservoir_dimension)).-0.5;
        W_in = rand(Float64, (reservoir_dimension, system_dimension)).-0.5;
        return ResComp.UntrainedResComp(u, W_in./opnorm(W_in), A./opnorm(A), f, gamma, sigma, rho, alpha);
end;

struct TrainedResComp
    W_out
    W_in
    A
    f
    gamma
    sigma
    rho
    alpha
end;

TrainedResComp(W_out, r::UntrainedResComp) = TrainedResComp(W_out, r.W_in, r.A, r.f, r.gamma, r.sigma, r.rho, r.alpha);

function autonomous_drive(r, rescomp::Union{UntrainedResComp,TrainedResComp}, u)
        return -r + rescomp.f.(rescomp.rho.*rescomp.A*r + rescomp.sigma*rescomp.W_in*u);
end

function drive!(dr, r, rescomp::UntrainedResComp, t)
        dr[:] = rescomp.gamma.*autonomous_drive(r, rescomp, rescomp.u(t));
end;

function drive!(dr, r, rescomp::TrainedResComp, t)
        dr[:] = rescomp.gamma.*autonomous_drive(r, rescomp, rescomp.W_out*r)
end;

function calculateOutputMapping(rescomp::UntrainedResComp, drive_sol)
        R = hcat(drive_sol.u...)
        S = hcat(rescomp.u(drive_sol.t)...)
        return ((R*R'+rescomp.alpha*I) \ R*S')';
end;

function train(rescomp::UntrainedResComp, r₀, tspan::Tuple{T, T}) where {T<:Real}
        drive_prob = ODEProblem(drive!, r₀, tspan, rescomp);
        drive_sol = solve(drive_prob);
        W_out = calculateOutputMapping(rescomp, drive_sol);
        return TrainedResComp(W_out, rescomp), drive_sol;
end;

function train_windows(rescomp::UntrainedResComp, r₀, tspan::Tuple{T, T}, num_windows::Int) where {T<:Real}
        drive_prob = ODEProblem(drive!, r₀, tspan, rescomp)
        affect!(integrator) = integrator.u = integrator.p.Wᵢₙ*integrator.p.u(integrator.t)
        callback = PeriodicCallback(affect!, (tspan[2]-tspan[1])/num_windows)
        drive_sol = solve(drive_prob, callback=callback)
        W_out = calculateOutputMapping(rescomp, drive_sol);
        return TrainedResComp(W_out, rescomp), drive_sol
end

function test(rescomp::TrainedResComp, r₀, tspan::Tuple{T, T}, u) where {T<:Real}
        drive_prob = ODEProblem(drive!, r₀, tspan, rescomp);
        condition(r, t, integrator) = norm(rescomp.W_out*r - u(t), 2) > 1.0;
        affect!(integrator) = terminate!(integrator);
        vpt_cb = DiscreteCallback(condition, affect!);
        drive_sol = solve(drive_prob, callback=vpt_cb);
        return drive_sol; 
end;

end