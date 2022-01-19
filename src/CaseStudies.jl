module CaseStudies
using DifferentialEquations

struct LorenzParams
    σ
    ρ
    β
end;

LorenzParams() = LorenzParams(10.0, 28.0, 8.0/3);

function lorenz!(du, u, p::LorenzParams, t)
    du[1] = p.σ*(u[2]-u[1])
    du[2] = u[1]*(p.ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - p.β*u[3]
end;

function rossler!(du, u, p, t)
    du[1] = -(u[2]+u[3])
    du[2] = u[1]+u[2]/5.0
    du[3] = 0.2+u[3]*(u[1]-5.7)
end;

function thomas!(du, u, b, t)
    du[1] = sin(u[2])-b*u[1]
    du[2] = sin(u[3])-b*u[2]
    du[3] = sin(u[1])-b*u[3]
end;

function get_solution(system, duration::Real, params)
    problem = ODEProblem(system, rand(3), (0.0, 10.0), params)
    solution = solve(problem)
    problem_on_attractor = ODEProblem(system, solution.u[end], (0.0, duration), params)
    solution_on_attractor = solve(problem_on_attractor)
    return solution_on_attractor
end;

function get_lorenz(duration::Real)
    return get_solution(lorenz!, duration, LorenzParams());
end;

function get_rossler(duration)
    return get_solution(rossler!, duration, 0);
end;

function get_thomas(duration)
    return get_solution(thomas!, duration, 0.2)
end;

function get_system(system_name, duration)
    if system_name == "lorenz"
        system = CaseStudies.get_lorenz(duration)
    elseif system_name == "rossler"
        system = CaseStudies.get_rossler(duration)
    elseif system_name == "thomas"
        system = CaseStudies.get_thomas(duration)
    end
    return system
end;

end;