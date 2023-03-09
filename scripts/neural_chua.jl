using StaticArrays, Random, ProgressMeter, BSON, Dates
using OrdinaryDiffEq, SciMLSensitivity, Flux, CUDA
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

p_ode = SA{Float32}[ 18.0, 33.0, -0.2, 0.01 ]
v(u) = v(u, p_ode, 0)

λ_max = 12.5
t0, t1 = 0f0, 1f-1
tspan = (t0, t1)
dt = 0.1f0

a, b, m0, m1 = p_ode
eq = SA{Float32}[ sqrt(-3*m0/m1), 0, -sqrt(-3*m0/m1) ]   # equilibrium
x0 = SA{Float32}[ -4.9, 7.6, 0.5 ]
#δ = 5f0
#x0 = eq .+ Tuple(2δ * rand(Float32, 3) .- δ)               # start at a perturbed equilibrium

prob = ODEProblem(v, x0, (t0, 500*t1), p_ode)
true_sol = solve(prob, Tsit5(), saveat=dt, sensealg=InterpolatingAdjoint())

plot_trajectory(true_sol)

train, valid = NODEDataloader(true_sol, 4; dt=dt, valid_set=0.8, GPU=true)

N_weights = 30
neural_net = Chain(
    u -> [u; u.^3],
    Dense(6 => N_weights, swish),
    #Dense(N_weights => N_weights, swish), 
    #Dense(N_weights => N_weights, swish), 
    SkipConnection(Dense(N_weights => N_weights, swish), +),
    SkipConnection(Dense(N_weights => N_weights, swish), +),
    Dense(N_weights => 3, swish)
) |> gpu

p, re_nn = Flux.destructure(neural_net)
p = p |> gpu
neural_ode(u, p, t) = re_nn(p)(u)
neural_ode_prob = ODEProblem(neural_ode, CuArray(x0), tspan, p)
model = ChaoticNDE(neural_ode_prob, alg=Tsit5(), gpu=true, sensealg=InterpolatingAdjoint())

loss(x, y) = sum(abs2, x - y)

η = 1e-4
θ = 1f-4
opt = Flux.OptimiserChain(Flux.WeightDecay(θ), Flux.RMSProp(η))
opt_state = Flux.setup(opt, model) 

#model.p .= CuArray(p_trained)
#opt = Flux.OptimiserChain(Flux.WeightDecay(θ*1f-4), Flux.RAdam(η))
#opt_state = Flux.setup(opt, model) 

model(train[1])
grad = Flux.gradient(model) do m
    result = m(train[1])
    loss(result, train[1][2])
end   

iter = 0
while isapprox(norm(grad[1].p), 0, atol=100) && iter < 4
    model.p .= 0.01 .* CUDA.rand(Float32, size(p))
    grad = Flux.gradient(model) do m
        result = m(train[1])
        loss(result, train[1][2])
    end    
    iter += 1
end

TRAIN = true
epochs = 30
prog = Progress(epochs)
if TRAIN
    for i_e = 1:epochs

        Flux.train!(model, train, opt_state) do m, t, x
            result = m((t,x))
            loss(result, x)
        end 

        l = loss( model(train[1]), train[1][2] )
        display(plot_nde(true_sol, model, train))

        if i_e % 30 == 0
            η /= 2
            Flux.adjust!(opt_state, η)
        end

        ProgressMeter.next!(prog, showvalues=[(:loss, l)])

    end
end

p_trained = Array(model.p)
BSON.@save "params_$( round(now(), Minute(1)) )" p_trained

v_trained = re_nn(CuArray(p_trained))

function v_true(u, p=p_ode, t=0)
    x, y, z = u
    a, b, m0, m1 = p
    SA[ a*(y-m0*x-m1/3.0*x^3), x-y+z, -b*y ]
end

# -------------------------------------


