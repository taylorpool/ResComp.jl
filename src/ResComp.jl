module ResComp
    using DifferentialEquations;

    struct AbstractResComp
    end;

    export struct UntrainedResComp::AbstractResComp
        Wᵢₙ
        u
        A
        f
        γ
        σ
        ρ
    end;
    
    export struct TrainedResComp::AbstractResComp
        Wₒᵤₜ
        Wᵢₙ
        u
        A
        f
        γ
        σ
        ρ
    end;

    function drive!(dr, r, rescomp::UntrainedResComp, t)
            dr[:] = rescomp.γ.*(-r + rescomp.f.(rescomp.ρ.*rescomp.A*r + rescomp.σ*rescomp.Wᵢₙ*rescomp.u(t)));
    end
    
    export function train(rescomp::UntrainedResComp, r₀, tspan)
            drive_prob = ODEProblem(drive!, r₀, tspan, rescomp);
            drive_sol = solve(drive_prob);
            D = rescomp.u.(drive_sol.t);
            R = hcat(sol.u...);
            Wₒᵤₜ = (R*R' \ R*D)';
            return Wₒᵤₜ
    end
end
