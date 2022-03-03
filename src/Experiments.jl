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
    cost_function(r) = sum(ResComp.evolve(r,urc,initial_signal).^2)
    Optim.minimizer(optimize(cost_function, initial_guess, LBFGS(); autodiff=:forward))
end

function window_train(urc::ResComp.UntrainedResComp, regularization, tspan, windows::WindowParams)
    window_length = (tspan[2]-tspan[1])/windows.number
    Nr = size(urc.W)[1]
    Nu = size(urc.W_in)[2]-1
    R_hat = zeros(Nr, Nr)
    R_S = zeros(Nr, Nu)

    final_state = zeros(Nr)
    
    for window_index = 1:windows.number
        window_tspan = window_length .* (window_index-1, window_index) .+ tspan[1]

        initial_state_system = urc.u(window_tspan[1])
        initial_state_reservoir = initial_condition_mapping(urc, initial_state_system)

        drive_prob = ODEProblem(ResComp.evolve!, initial_state_reservoir, window_tspan, urc)
        drive_sol = solve(drive_prob)
        R = hcat(drive_sol.u...)
        S = hcat(urc.u.(drive_sol.t)...)
        R_hat += R*R'
        R_S += R*S'

        final_state = drive_sol.u[end]
    end

    W_out = ((R_hat-regularization*I) \ R_S)'
    return ResComp.TrainedResComp(W_out, urc), final_state
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

function burn_train_test(duration)
    (0.0, duration/3), (duration/3, 2//3*duration), (2//3*duration, duration)
end

function continue_standard(params)
    # Create burn in, train, and test timespans
    burn_tspan, train_tspan, test_tspan = burn_train_test(params["system_duration"])

    # Create untrained reservoir computer
    urc::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = ResComp.burn_in(urc, burn_tspan)

    # Train
    trc, train_sol = standard_train(urc, initial_state, params["alpha"], train_tspan)

    # Return the valid prediction time
    return ResComp.vpt(trc, train_sol.u[end], test_tspan, urc.u)
end

function random_standard(params)
    # Create burn in, train, and test timespans
    burn_tspan, train_tspan, test_tspan = burn_train_test(params["system_duration"])

    # Create untrained reservoir computer
    urc::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = ResComp.burn_in(urc, burn_tspan)

    # Train
    trc, train_sol = standard_train(urc, initial_state, params["alpha"], train_tspan)

    # Get initial condition mapping
    test_time = test_tspan[2]-test_tspan[1]
    initial_time = rand()/2*test_time + test_tspan[1]
    initial_state = initial_condition_mapping(urc, urc.u(initial_time))

    # Return the valid prediction time
    return ResComp.vpt(trc, initial_state, (initial_time, test_tspan[2]), urc.u)
end

function continue_windows(params)
    # Create burn in, train, and test timespans
    burn_tspan, train_tspan, test_tspan = burn_train_test(params["system_duration"])

    # Create untrained reservoir computer
    urc::ResComp.UntrainedResComp = create_from_dict(params)

    # Train using windowed approach
    windows = WindowParams(params["num_windows"])
    trc, final_state = window_train(urc, params["alpha"], train_tspan, windows)

    # Return the valid prediction time
    return ResComp.vpt(trc, final_state, test_tspan, urc.u)
end

function random_windows(params)
    # Create burn in, train, and test timespans
    burn_tspan, train_tspan, test_tspan = burn_train_test(params["system_duration"])

    # Create untrained reservoir computer
    urc::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = ResComp.burn_in(urc, burn_tspan)

    # Train using windowed approach
    windows = WindowParams(params["num_windows"])
    trc, final_state = window_train(urc, params["alpha"], train_tspan, windows)

    # Get initial condition mapping
    test_time = test_tspan[2]-test_tspan[1]
    initial_time = rand()/2*test_time + test_tspan[1]
    initial_state = initial_condition_mapping(urc, urc.u(initial_time))

    # Return the valid prediction time
    return ResComp.vpt(trc, initial_state, (initial_time, test_tspan[2]), urc.u)
end

end