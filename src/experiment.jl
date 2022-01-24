include("CaseStudies.jl")
include("Optimize.jl")
using Dates
using PyCall
import JSON

# Import the ax_client module
ax_client = PyCall.pyimport("ax.service.ax_client")

# Get the filepath of the experiment
experiment_filepath = ARGS[1]
# Read in the experiment into a JSON file
experiment = JSON.parse(read(experiment_filepath, String))

# Get parameter string
params = JSON.json(experiment["optimization_parameters"])
# Get num_trials
num_trials = experiment["num_trials"]
# Get experiment name
experiment_name = experiment["experiment_name"]
# Get system name
system_name = experiment["system_name"]
# Get system_duration
system_duration = experiment["system_duration"]
# Get system dimension
system_dimension = experiment["system_dimension"]
# Get reservoir dimension
reservoir_dimension = experiment["reservoir_dimension"]
# Get results directory
results_directory = experiment["results_directory"]

# Set the system of interest
system = CaseStudies.get_system(
    system_name, 
    system_duration 
)

# Create the optimization client
client = ax_client.AxClient()

# Create the experiment
PyCall.py"$client.create_experiment(
    name=$experiment_name,
    parameters=$$params,
    objective_name='vpt',
    minimize=False)"

# Iterate through each trial
for trial_index = 1:num_trials
    trial_parameters, trial_index = client.get_next_trial()
    # Set system dimension
    trial_parameters["system_dimension"] = system_dimension
    # Set reservoir dimension
    trial_parameters["reservoir_dimension"] = reservoir_dimension
    # Set system
    trial_parameters["system"] = system
    trial_parameters["experiment_params"] = experiment
    # Complete the trial
    client.complete_trial(
        trial_index=trial_index,
        raw_data=Optimize.evaluate(trial_parameters))
end

# Get current datetime
time = Dates.format(now(), "yyyy-mm-dd-HH:MM:SS")
# Save the results to a json file
client.save_to_json_file(filepath=results_directory*experiment_name*time*".json")