include("CaseStudies.jl")
include("Optimize.jl")
using Dates

# Get the name of the system of interest
name = ARGS[1]
# Get the number of trials to perform
num_trials = parse(Int, ARGS[2])
# Set the duration
duration = 200.0

# Set the system of interest
if name == "lorenz"
    system = CaseStudies.get_lorenz(duration)
elseif name == "rossler"
    system = CaseStudies.get_rossler(duration)
elseif name == "thomas"
    system = CaseStudies.get_thomas(duration)
end

# Perform hyperparameter optimization
client = Optimize.torch_rescomp(system, 3, num_trials)
# Get current datetime
time = Dates.format(now(), "yyyy-mm-dd-HH:MM:SS")
# Save the results to a json file
client.save_to_json_file(filepath="/fslhome/tpool2/"*name*time*".json")