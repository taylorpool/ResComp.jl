module Experiments
include("ResComp.jl")
using DifferentialEquations
using Optim
using LinearAlgebra

export continue_standard

struct WindowParams
    number
end

function standard_train(urc::ResComp.UntrainedResComp, initial_state, regularization, tspan)
    # Apply training
    W_out, train_solution = ResComp.train(urc, initial_state, regularization, tspan)
    # Create the trained reservoir
    trc = ResComp.TrainedResComp(W_out, urc)
    # Return the trained reservoir and the training solution
    return trc, train_solution
end

function initial_condition_mapping(urc::ResComp.UntrainedResComp, initial_signal)
    initial_guess = urc.f.(urc.W_in*vcat(1,initial_signal))
    cost_function(r) = sum(((1-urc.gamma)*r + urc.gamma*(urc.f.(urc.W_in*vcat(1,initial_signal) + urc.W*r))).^2)
    Optim.minimizer(optimize(cost_function, initial_guess, LBFGS(); autodiff=:forward))
end

function window_train(urc::ResComp.UntrainedResComp, initial_state, tspan, windows::WindowParams)
    window_length = (tspan[2]-tspan[1])/windows.number
    R_hat = zeros(size(urc.W))
    R_S = zeros(size(urc.W_in))
    
    for window_index = 1:windows.number
        window_tspan = window_length .* (window_index-1, window_index) .+ tspan[1]

        initial_state_system = urc.u(window_tspan[1])
        initial_state_reservoir = initial_condition_mapping(urc, initial_state_system)

        drive_prob = ODEProblem(ResComp.evolve!, initial_state_reservoir, window_tspan, urc)
        drive_sol = solve(drive_prob)
        R = hcat(drive_sol.u...)
        S = hcat(urc.u(drive_sol.t)...)
        R_hat += R*R'
        R_S += R*S'
    end

    W_out = ((R_hat+urc.alpha*I) \ R_S)'
    return ResComp.TrainedResComp(W_out, urc)
end

function burn_in(rescomp::Union{ResComp.UntrainedResComp, ResComp.TrainedResComp}, tspan)
    # Create burn in problem
    burn_in_problem = ODEProblem(ResComp.drive!, rand(size(rescomp.W_in)[1]), tspan, rescomp)
    # Solve burn in
    burn_in_solution = solve(burn_in_problem)
    # Return last burn in state
    return burn_in_solution.u[end]
end

function create_from_dict(params)
    urc = ResComp.UntrainedResComp(
        params["system"],
        params["rho"],
        params["sigma"],
        params["gamma"],
        params["function"],
        params["system_dimension"],
        params["reservoir_dimension"],
        params["bias_scale"]
    )
    return urc
end

function continue_standard(params)
    # Create untrained reservoir computer
    urc::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = ResComp.burn_in(urc, (0.0, 40.0))

    # Train
    trc, train_solution = standard_train(urc, initial_state, params["alpha"], (40.0, 60.0))

    # Return the valid prediction time
    return ResComp.vpt(trc, train_solution.u[end], (60.0, 80.0), urc.u)
end

function random_standard(params)
    # Create untrained reservoir computer
    urc::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = ResComp.burn_in(urc, (0.0, 40.0))

    # Train
    trc, train_solution = standard_train(urc, initial_state, params["alpha"], (40.0, 60.0))

    # Get initial condition mapping
    initial_time = rand()*20 + 60.0
    initial_state = initial_condition_mapping(urc, urc.u(initial_time))

    # Return the valid prediction time
    return ResComp.vpt(trc, initial_state, (initial_time, initial_time + 20.0), urc.u)
end

function continue_windows(params)
    # Create untrained reservoir computer
    urc::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = burn_in(urc, (0.0, 40.0))

    # Train using windowed approach
    windows = WindowParams(params["num_windows"])
    trc = window_train(urc, initial_state, (40.0, 60.0), windows)

    # Test
    test_solution = ResComp.test(trc, urc.u(60.0), (60.0, 80.0), urc.u)

    # Return the valid prediction time
    return test_solution.t[end] - test_solution.t[1]
end

function random_windows(params)
    # Create untrained reservoir computer
    urc::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = burn_in(urc, (0.0, 40.0))

    # Train using windowed approach
    windows = WindowParams(params["num_windows"])
    trc = window_train(urc, initial_state, (40.0, 60.0), windows)

    # Get initial condition mapping
    initial_time = rand()*20 + 60.0
    initial_state = initial_condition_mapping(urc, urc.u(initial_time))

    # Test
    test_solution = ResComp.test(trc, initial_state, (initial_time, initial_time + 20.0), urc.u)

    # Return the valid prediction time
    return test_solution.t[end] - test_solution.t[1]
end

end