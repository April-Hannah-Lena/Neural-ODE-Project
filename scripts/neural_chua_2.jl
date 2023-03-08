using StaticArrays, Random, ProgressMeter
using OrdinaryDiffEq, SciMLSensitivity, Flux, CUDA, ComponentArrays
using ChaoticNDETools, NODEData
using GAIO

Random.seed!(1234)
include("plotting.jl")

# Chua's circuit
function v(u, p, t)
    x, y, z = u
    a, b, m0, m1 = p
    SA{Float32}[ a*(y-m0*x-m1/3.0*x^3), x-y+z, -b*y ]
end

function v(u, p, t)
    x, y, z = u
    σ, ρ, β = p
    SA{Float32}[ σ*(y-x), -x*z + ρ*x - y, x*y - β*z ]
end

p_ode = SA{Float32}[ 10, 28, 8/3 ]
x0 = SA{Float32}[ 3.8, 1.9, 25.5 ]

p_ode = SA{Float32}[ 18.0, 33.0, -0.2, 0.01 ]
v(u) = v(u, p_ode, 0)

t0, t1 = 0f0, 1f-1
tspan = (t0, t1)
dt = 0.1f0

a, b, m0, m1 = p_ode
eq = SA{Float32}[ sqrt(-3*m0/m1), 0, -sqrt(-3*m0/m1) ]   # equilibrium
x0 = SA{Float32}[ 3.8, 1.9, 0.5 ]
#δ = 5f0
#x0 = eq .+ Tuple(2δ * rand(Float32, 3) .- δ)               # start at a perturbed equilibrium

prob = ODEProblem(v, x0, (t0, 500*t1), p_ode)
true_sol = solve(prob, Tsit5(), saveat=dt, sensealg=ForwardDiffSensitivity())#InterpolatingAdjoint())

plot_trajectory(true_sol)

train, valid = NODEDataloader(true_sol, 10; dt=dt, valid_set=0.8, GPU=false)

N_weights = 30
neural_net = Chain(
    Dense(3 => N_weights, swish),
    Dense(N_weights => N_weights, swish), 
    Dense(N_weights => N_weights, swish), 
    Dense(N_weights => N_weights, swish),
    Dense(N_weights => N_weights, swish),
    #SkipConnection(Dense(N_weights => N_weights, swish), +),
    #SkipConnection(Dense(N_weights => N_weights, swish), +),
    #SkipConnection(Dense(N_weights => N_weights, swish), +),
    Dense(N_weights => 3, swish)
) |> gpu

p, re_nn = Flux.destructure(neural_net)
p = p |> ComponentArray |> gpu
neural_ode(u, p, t) = re_nn(p)(u)
neural_ode_prob = ODEProblem(neural_ode, x0, tspan, p)
model = ChaoticNDE(neural_ode_prob, alg=Tsit5(), sensealg=ForwardDiffSensitivity())#InterpolatingAdjoint())

loss(x, y) = sum(abs2, x - y)

η = 1e-3
θ = 1f-2
opt = Flux.Adam(η)
opt_state = Flux.setup(opt, model)

opt_state.p.state[1] .+= 0.4 .* rand(Float32, size(p))
opt_state.p.state[2] .= opt_state.p.state[1]

TRAIN = true
if TRAIN
    @showprogress "Training..." for i_e = 1:30

        Flux.train!(model, train, opt_state) do m, t, x
            result = m((t,x))
            loss(result, x)
        end 

        display(plot_nde(true_sol, model, train))

        if i_e % 30 == 0
            η /= 2
            Flux.adjust!(opt_state, η)
        end

    end
end

# -------------------------------------

center, radius = (0,0,0), (12,3,20)
Q = Box{Float32}(center, radius)
P = BoxPartition(Q, (128,128,128))

no_of_points = 200

f(x) = rk4_flow_map(v, x, 0.01f0, 10)
F = BoxMap(:montecarlo, f, P, no_of_points=no_of_points)

trained_neural_ode = re_nn(model.p)
f_trained(x) = rk4_flow_map(trained_neural_ode, x, 0.01f0, 10)
F_trained = BoxMap(:montecarlo, f_trained, P, no_of_points=no_of_points)

# computing the attractor by covering the 2d unstable manifold of two equilibria
S = cover(P, [eq, -eq])

W = unstable_set(F, S)
W_trained = unstable_set(F_trained, S)

plot_unstable_sets(W, W_trained)
plot_symdiff(W, W_trained)

T = TransferOperator(F, W, W)
# we give Arpack some help converging to the eigenvalues,
# see the Arpack docs for explanations of keywords
tol, maxiter, v0 = eps()^(1/4), 1000, ones(Float32, size(T, 2))
λ, ev = eigs(T; nev=5, which=:LR, maxiter=maxiter, tol=tol, v0=v0)

μ = real ∘ ev[2]
μ_rescale = collect(values(μ))
μ_rescale .= sign(μ_rescale) .* (log ∘ abs).(μ_rescale)
μ = BoxFun(μ, μ_rescale)

T_trained = TransferOperator(F_trained, W, W)
λ_trained, ev_trained = eigs(T_trained; nev=5, which=:LR, maxiter=maxiter, tol=tol, v0=v0)

μ_trained = real ∘ ev[2]
μ_rescale = collect(values(μ_trained))
μ_rescale .= sign(μ_rescale) .* (log ∘ abs).(μ_rescale)
μ_trained = BoxFun(μ_trained, μ_rescale)

plot_measures(μ, μ_trained)

