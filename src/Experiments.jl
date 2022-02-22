module Experiments
include("ResComp.jl")
using DifferentialEquations
using Optim
using LinearAlgebra

export continue_standard

struct WindowParams
    number
end

function standard_train(untrained_rescomp::ResComp.UntrainedResComp, initial_state, tspan)
    # Create training problem
    train_problem = ODEProblem(ResComp.drive!, initial_state, tspan, untrained_rescomp)
    # Solve training problem
    train_solution = solve(train_problem)
    # Find the proper W_out matrix
    W_out = ResComp.calculateOutputMapping(untrained_rescomp, train_solution)
    # Create the trained reservoir
    trained_rescomp::ResComp.TrainedResComp = ResComp.TrainedResComp(W_out, untrained_rescomp)
    # Return the trained reservoir and the training solution
    return trained_rescomp, train_solution
end

function initial_condition_mapping(rescomp::ResComp.UntrainedResComp, initial_signal)
    initial_guess = rescomp.f.(rescomp.sigma*rescomp.W_in*initial_signal);
    num_nodes = size(rescomp.A)[1]
    initial_conditions = zeros(num_nodes)
    for index = 1:num_nodes
        cost_function = node -> begin
            abs(rescomp.f(rescomp.rho*node + rescomp.sigma*dot(rescomp.W_in[index], initial_signal)) + node)
        end
        initial_conditions[index] = Optim.minimizer(
            optimize(cost_function, initial_guess[index], LBFGS(); autodiff=:forward))
    end
    initial_conditions
end

function window_train(untrained_rescomp::ResComp.UntrainedResComp, initial_state, tspan, windows::WindowParams)
    window_length = (tspan[2]-tspan[1])/windows.number
    R_hat = zeros(size(untrained_rescomp.A))
    R_S = zeros(size(untrained_rescomp.W_in))
    
    for window_index = 1:windows.number
        window_tspan = window_length .* (window_index-1, window_index) .+ tspan[1]

        initial_state_system = untrained_rescomp.u(window_tspan[1])
        initial_state_reservoir = initial_condition_mapping(untrained_rescomp, initial_state_system)

        drive_prob = ODEProblem(ResComp.drive!, initial_state_reservoir, window_tspan, untrained_rescomp)
        drive_sol = solve(drive_prob)
        R = hcat(drive_sol.u...)
        S = hcat(untrained_rescomp.u(drive_sol.t)...)
        R_hat += R*R'
        R_S += R*S'
    end

    W_out = ((R_hat+untrained_rescomp.alpha*I) \ R_S)'
    return ResComp.TrainedResComp(W_out, untrained_rescomp)
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
    untrained_rescomp::ResComp.UntrainedResComp = ResComp.initialize_rescomp(
        params["system"],
        params["function"],
        params["gamma"],
        params["sigma"],
        params["rho"],
        params["reservoir_dimension"],
        params["system_dimension"],
        params["alpha"],
        params["density"]
    )
    return untrained_rescomp
end

function continue_standard(params)
    # Create untrained reservoir computer
    untrained_rescomp::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = burn_in(untrained_rescomp, (0.0, 40.0))

    # Train
    trained_rescomp, train_solution = standard_train(untrained_rescomp, initial_state, (40.0, 60.0))

    # Test
    test_solution = ResComp.test(trained_rescomp, train_solution.u[end], (60.0, 80.0), untrained_rescomp.u)

    # Return the valid prediction time
    return test_solution.t[end] - test_solution.t[1]
end

function random_standard(params)
    # Create untrained reservoir computer
    untrained_rescomp::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = burn_in(untrained_rescomp, (0.0, 40.0))

    # Train
    trained_rescomp, train_solution = standard_train(untrained_rescomp, initial_state, (40.0, 60.0))

    # Get initial condition mapping
    initial_time = rand()*20 + 60.0
    initial_state = initial_condition_mapping(untrained_rescomp, untrained_rescomp.u(initial_time))

    # Test
    test_solution = ResComp.test(trained_rescomp, initial_state, (initial_time, initial_time + 20.0), untrained_rescomp.u)

    # Return the valid prediction time
    return test_solution.t[end] - test_solution.t[1]
end

function continue_windows(params)
    # Create untrained reservoir computer
    untrained_rescomp::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = burn_in(untrained_rescomp, (0.0, 40.0))

    # Train using windowed approach
    windows = WindowParams(params["num_windows"])
    trained_rescomp = window_train(untrained_rescomp, initial_state, (40.0, 60.0), windows)

    # Test
    test_solution = ResComp.test(trained_rescomp, untrained_rescomp.u(60.0), (60.0, 80.0), untrained_rescomp.u)

    # Return the valid prediction time
    return test_solution.t[end] - test_solution.t[1]
end

function random_windows(params)
    # Create untrained reservoir computer
    untrained_rescomp::ResComp.UntrainedResComp = create_from_dict(params)

    # Burn in
    initial_state = burn_in(untrained_rescomp, (0.0, 40.0))

    # Train using windowed approach
    windows = WindowParams(params["num_windows"])
    trained_rescomp = window_train(untrained_rescomp, initial_state, (40.0, 60.0), windows)

    # Get initial condition mapping
    initial_time = rand()*20 + 60.0
    initial_state = initial_condition_mapping(untrained_rescomp, untrained_rescomp.u(initial_time))

    # Test
    test_solution = ResComp.test(trained_rescomp, initial_state, (initial_time, initial_time + 20.0), untrained_rescomp.u)

    # Return the valid prediction time
    return test_solution.t[end] - test_solution.t[1]
end

end