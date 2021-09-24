include("ResComp.jl")

using Hyperopt

function f(γ, σ, ρ)
       untrained = ResComp.UntrainedResComp(
            2*(rand(Float64, (20, 1)).-0.5),
            t->0.5.*sin.(t),
            2*(rand(Float64, (20,20)).-0.5),
            tanh,
            γ,
            σ,
            ρ, 
            0.1)
        nᵣ = 20
        r₀ = 2*rand(Float64, nᵣ).-0.5
        trained, test_sol = ResComp.train(untrained, r₀, (-100.0, 200.0))
        vpt, train_sol = ResComp.test(trained, test_sol.u[end], (200.0, 4000.0))
        return -vpt
end

ho = @hyperopt for i = 50,
        γ = LinRange(0.1,25,10),
        σ = LinRange(0.01, 5.0,10),
        ρ = LinRange(0.1, 25,10)

        @show f(γ,σ,ρ)
end

printmin(ho)
