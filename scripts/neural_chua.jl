using Lux, Optimization, OptimizationOptimisers, Zygote, OrdinaryDiffEq, 
      CUDA, SciMLSensitivity, Random, ComponentArrays, StaticArrays
using Lux: setup, relu, SkipConnection
using GAIO
import DiffEqFlux: NeuralODE 

#rng for Lux.setup
rng = Random.seed!(234)
include("plotting.jl")

# Chua's circuit
function v(du, u, p, t)
    x, y, z = u
    a, b, m0, m1 = p
    du[1] = a*(y-m0*x-m1/3.0*x^3)
    du[2] = x-y+z
    du[3] = -b*y
    return
end

p_ode = NTuple{4,Float32}((18.0, 33.0, -0.2, 0.01))

tspan = (0f0, 2f0)
dt = 0.1f0

a, b, m0, m1 = p_ode
eq = Float32[ sqrt(-3*m0/m1), 0, -sqrt(-3*m0/m1) ] |> gpu   # equilibrium
x0 = Float32[ 0.6, 0.2, -5 ] |> gpu 
#δ = 5f0
#x0 = eq .+ Tuple(2δ * rand(Float32, 3) .- δ)               # start at a perturbed equilibrium

prob = ODEProblem(v, x0, tspan, p_ode)
true_sol = solve(prob, RadauIIA3(), saveat=dt)#=, sensealg=InterpolatingAdjoint())=# |> gpu

plot_trajectory(true_sol)

N_weights = 6
neural_net = Chain(
    Dense(3 => N_weights, relu),
    Dense(N_weights => N_weights, relu),
    #SkipConnection(Dense(N_weights => N_weights, relu), +),
    #SkipConnection(Dense(N_weights => N_weights, relu), +),
    Dense(N_weights => 3, relu)
)

p, st = setup(rng, neural_net) 
p = p |> ComponentArray |> gpu
p .+= 0.3 .* CUDA.rand(Float32, size(p))
p .*= -1
st = st |> gpu

prob_neuralode = NeuralODE(neural_net |> gpu, (tspan[1], tspan[2]+eps(Float32)), RadauIIA3(), saveat=dt)

function predict_neuralode(p)
  prob_neuralode(x0,p,st) |> first |> gpu
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, true_sol .- pred)
    return loss, pred
end

η = 1f-1
optalg = Adam(η)
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p,_)->loss_neuralode(p), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
p = Optimization.solve(
    optprob,
    optalg,
    callback = callback,
    maxiters = 5
)

η = 1f-4
optalg = Momentum(η)#Adam(η)
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p,_)->loss_neuralode(p), adtype)
optprob = Optimization.OptimizationProblem(optf, p)
p = Optimization.solve(
    optprob,
    optalg,
    callback = callback,
    maxiters = 50
)

# Use result params to create a single-step map
v_trained(x) = neural_net(x, result.u, st)
f_trained(x) = rk4_flow_map(v_trained, x, 0.1f0, 1)

function v_true(u, p=p_ode, t=0)
    x, y, z = u
    a, b, m0, m1 = p
    SA[ a*(y-m0*x-m1/3.0*x^3), x-y+z, -b*y ]
end
f_true(x) = rk4_flow_map(v_true, x, 0.1f0, 1)


# -------------------------------------

center, radius = (0,0,0), (12,3,20)
Q = Box{Float32}(center, radius)
P = BoxPartition(Q, (128,128,128))

no_of_points = 200
F_trained = BoxMap(:montecarlo, f_trained, P, no_of_points=no_of_points)
F_true = BoxMap(:montecarlo, f_true, P, no_of_points=no_of_points)

# computing the attractor by covering the 2d unstable manifold of two equilibria
S = cover(P, [eq, -eq])

W = unstable_set(F_true, S)
W_trained = unstable_set(F_trained, S)

plot_unstable_sets(W, W_trained)
plot_symdiff(W, W_trained)

T = TransferOperator(F_true, W, W)
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

