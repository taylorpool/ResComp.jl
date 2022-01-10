import Pkg
Pkg.activate("/fslhome/tpool2/ResComp.jl")
ENV["PYTHON"] = "/fslhome/tpool2/RCInitialCond/venv/bin/python"
Pkg.add("PyCall")
Pkg.add("DifferentialEquations")
Pkg.add("Erdos")