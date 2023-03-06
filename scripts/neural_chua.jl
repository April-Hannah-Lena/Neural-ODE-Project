using StaticArrays, Random, ProgressMeter
using OrdinaryDiffEq, SciMLSensitivity, Flux, CUDA
using ChaoticNDETools, NODEData
using GAIO
using GLMakie: plot, plot!, Figure, Axis3, Colorbar

Random.seed!(1234)

# Chua's circuit
function v(u, p, t)
    x, y, z = u
    a, b, m0, m1 = p
    SA{Float32}[ a*(y-m0*x-m1/3.0*x^3), x-y+z, -b*y ]
end

p_ode = SA{Float32}[ 18.0, 33.0, -0.2, 0.01 ]
v(u) = v(u, p_ode, 0)

tspan = (0f0, 50f0)
dt = 0.01f0

a, b, m0, m1 = p_ode
eq = SA{Float32}[ sqrt(-3*m0/m1), 0, -sqrt(-3*m0/m1) ]   # equilibrium
x0 = SA{Float32}[ 5, 1, 10 ]
#δ = 5f0
#x0 = eq .+ Tuple(2δ * rand(Float32, 3) .- δ)               # start at a perturbed equilibrium

prob = ODEProblem(v, x0, tspan, p_ode)
true_sol = solve(prob, Tsit5(), saveat=dt, sensealg=InterpolatingAdjoint())
train, valid = NODEDataloader(true_sol, 10; dt=dt, valid_set=0.8f0, GPU=true)

N_weights = 10
neural_net = Chain(
    Dense(3 => N_weights, relu),
    SkipConnection(Dense(N_weights => N_weights, relu), +),
    SkipConnection(Dense(N_weights => N_weights, relu), +),
    #SkipConnection(Dense(N_weights => N_weights, relu), +),
    Dense(N_weights => 3, relu)
) |> gpu

p, re_nn = Flux.destructure(neural_net)
neural_ode(u, p, t) = re_nn(p)(u)
neural_ode_prob = ODEProblem(neural_ode, x0, tspan, p)
model = ChaoticNDE(neural_ode_prob, alg=Tsit5(), sensealg=InterpolatingAdjoint())

loss(x, y) = sum(abs2, x - y)

function plot_NDE()
    plt = plot(sol.t, Array(model((sol.t,train[1][2])))', label="Neural ODE")
    plot!(plt, sol.t, Array(sol)', label="Training Data")
    plot!(plt, [train[1][1][1],train[end][1][end]],zeros(2),label="Length of Training Set", linewidth=5, ylims=[0,5])
    display(plt)
end

η = 1e-3
opt = Flux.AdamW(η)
opt_state = Flux.setup(opt, model)

TRAIN = true
if TRAIN

    @showprogress "Training..." for i_e = 1:100

        Flux.train!(model, train, opt_state) do m, t, x
            result = m((t,x))
            loss(result, x)
        end 

        #plot_NDE()

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

fig = Figure();
ax = Axis3(fig[1,1], aspect=(1,1.2,1))
plot!(ax, W, color=(:blue, 0.4), label="True unstable manifold")
ax2 = Axis3(fig[1,2], aspect=(1,1.2,1))
plot!(ax2, W_trained, color=(:red, 0.4), label="Predicted unstable manifold")
display(fig)

fig = Figure();
ax = Axis3(fig[1,1], aspect=(1,1.2,1))
plot!(ax, setdiff(W, W_trained), color=(:blue, 0.8))
plot!(ax, setdiff(W_trained, W), color=(:red, 0.8))
display(fig)

T = TransferOperator(F, W, W)
# we give Arpack some help converging to the eigenvalues,
# see the Arpack docs for explanations of keywords
tol, maxiter, v0 = eps()^(1/4), 1000, ones(Float32, size(T, 2))
λ, ev = eigs(T; nev=5, which=:LR, maxiter=maxiter, tol=tol, v0=v0)

μ = real ∘ ev[2]
μ_rescale = collect(values(μ))
μ_rescale .= sign(μ_rescale) .* (log ∘ abs).(μ_rescale)
μ = BoxFun(μ, μ_rescale)

fig = Figure();
ax = Axis3(fig[1,1], aspect=(1,1.2,1))
ms = plot!(ax, μ, colormap=(:jet, 0.4), label="True almost invariant sets")

T_trained = TransferOperator(F_trained, W, W)
λ_trained, ev_trained = eigs(T_trained; nev=5, which=:LR, maxiter=maxiter, tol=tol, v0=v0)

μ_trained = real ∘ ev[2]
μ_rescale = collect(values(μ_trained))
μ_rescale .= sign(μ_rescale) .* (log ∘ abs).(μ_rescale)
μ_trained = BoxFun(μ_trained, μ_rescale)

ax2 = Axis3(fig[1,2], aspect=(1,1.2,1))
plot!(ax2, μ_trained, colormap=(:jet, 0.4), label="Predicted almost invariant sets")
Colorbar(fig[1, 3], ms)
display(fig)

