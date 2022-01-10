#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=2048M   # memory per CPU core
#SBATCH -J "resomp"   # job name
#SBATCH --mail-user=taylorpool.27@gmail.com   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load julia
source /fslhome/tpool2/RCInitialCond/venv/bin/activate
cd /fslhome/tpool2/ResComp.jl
julia --project src/setup.jl
julia --project -p 10 /fslhome/tpool2/ResComp.jl/src/experiment.jl $1
