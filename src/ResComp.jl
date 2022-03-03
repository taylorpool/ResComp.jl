module ResComp
using SparseArrays, DifferentialEquations, KrylovKit, LinearAlgebra

export UntrainedResComp, evolve!, burn_in, train, test, TrainedResComp, predict, vpt, evolve

struct UntrainedResComp
        W_in
        W
        f
        gamma
        u
end

struct TrainedResComp
        W_in
        W
        W_out
        f
        gamma
end

UntrainedResComp(u, rho, sigma, gamma, f, Nu, Nr, bias_scale) = begin 
        # Compute W_in
        W_in = 2*rand(Nr, 1+Nu).-1
        W_in[:,1] *= bias_scale
        W_in[:,2:end] *= sigma
        
        # Compute W
        W = 2*sprand(Nr, Nr, 0.05).-1
        spectral_radius = abs(eigsolve(W)[1][1])
        W *= rho/spectral_radius
    
        return UntrainedResComp(W_in, W, f, gamma, u)
end

TrainedResComp(W_out, urc::UntrainedResComp) = TrainedResComp(
        urc.W_in, urc.W, W_out, urc.f, urc.gamma)

function evolve(r, urc::UntrainedResComp, u::T) where T <: AbstractVector
        urc.gamma*(-r + urc.f.(urc.W_in*vcat(1,u) + urc.W*r))
end
function evolve(r, urc::UntrainedResComp, t::T) where T <: Real
        urc.gamma*(-r + urc.f.(urc.W_in*vcat(1,urc.u(t)) + urc.W*r))
end

function evolve!(dr, r, urc::UntrainedResComp, t)
        dr[:] = evolve(r, urc, t)
end

function burn_in(rc::Union{UntrainedResComp,TrainedResComp}, tspan)
        r0 = rand(size(rc.W)[1])
        prob = ODEProblem(evolve!, r0, tspan, rc)
        return solve(prob).u[end]
end

function train(rc::UntrainedResComp, r0, α, tspan)
        prob = ODEProblem(evolve!, r0, tspan, rc)
        solution = solve(prob, ABM54(), dt=0.02);
        R = hcat(solution.u...)
        U = hcat(rc.u.(solution.t)...)
        return ((R*R' - α*I) \ R*(U'))', solution
end

function evolve(r, trc::TrainedResComp, t)
        trc.gamma*(-r + trc.f.(trc.W_in*vcat(1,trc.W_out*r) + trc.W*r))
end

function evolve!(dr, r, trc::TrainedResComp, t)
        dr[:] = evolve(r, trc, t)
end


function predict(trc::TrainedResComp, r0, tspan)
        prob = ODEProblem(evolve!, r0, tspan, trc)
        return solve(prob);
end

function test(trc::TrainedResComp, r0, tspan, u)
        drive_prob = ODEProblem(evolve!, r0, tspan, trc)
        condition(r, t, integrator) = norm(trc.W_out*r - u(t), 2) > 1.0
        affect!(integrator) = terminate!(integrator)
        vpt_cb = DiscreteCallback(condition, affect!)
        drive_sol = solve(drive_prob, callback=vpt_cb)
        return drive_sol; 
end

function vpt(trc::TrainedResComp, r0, tspan, u)
        valid_prediction = test(trc, r0, tspan, u)
        return valid_prediction.t[end] - valid_prediction.t[1]
end

end