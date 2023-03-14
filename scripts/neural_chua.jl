using StaticArrays, Random, ProgressMeter, BSON
using Dates: now, format
using OrdinaryDiffEq, SciMLSensitivity, Flux#, CUDA
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
dt = 1f-2

a, b, m0, m1 = p_ode
eq = SA{Float32}[ sqrt(-3*m0/m1), 0, -sqrt(-3*m0/m1) ]   # equilibrium
x0 = SA{Float32}[ -4.9, 7.6, 0.5 ]
#δ = 5f0
#x0 = eq .+ Tuple(2δ * rand(Float32, 3) .- δ)               # start at a perturbed equilibrium

prob = ODEProblem(v, x0, (t0, 150*t1), p_ode)
true_sol = solve(prob, Tsit5(), saveat=dt, sensealg=InterpolatingAdjoint())

plot_trajectory(true_sol)

train, valid = NODEDataloader(true_sol, 8; dt=dt, valid_set=0.8, GPU=false#=true=#)

N_weights = 20
neural_net = Chain(
    #BatchNorm(1),
    #u -> 2*u,
    #u -> [u; u.^3],
    Dense(3 => N_weights, swish),
    #Dense(N_weights => N_weights, swish), 
    #Dense(N_weights => N_weights, swish), 
    SkipConnection(Dense(N_weights => N_weights, swish), +),
    SkipConnection(Dense(N_weights => N_weights, swish), +),
    Dense(N_weights => 3)
) #|> gpu

p, re_nn = Flux.destructure(neural_net)
p = p #|> gpu
neural_ode(u, p, t) = re_nn(p)(u)
neural_ode_prob = ODEProblem(neural_ode, #=CuArray(x0)=#x0, tspan, p)
model = ChaoticNDE(neural_ode_prob, alg=Tsit5(), gpu=false#=true=#, sensealg=InterpolatingAdjoint())

loss(x, y) = sum(enumerate(zip(x, y))) do z
    (i, (xi, yi)) = z
    abs2(xi - yi) * 0.95^i     # give closer times higher weight
end

η = 1f-3
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

#=
iter = 0
while isapprox(norm(grad[1].p), 0, atol=100) && iter < 4
    model.p .= 0.01 .* CUDA.rand(Float32, size(p))
    grad = Flux.gradient(model) do m
        result = m(train[1])
        loss(result, train[1][2])
    end    
    iter += 1
end
=#

l = loss( model(train[1]), train[1][2] )

TRAIN = true
epochs = 20
prog = Progress(epochs)
if TRAIN
    for i_e = 1:epochs

        Flux.train!(model, train, opt_state) do m, t, x
            result = m((t,x))
            loss(result, x)
        end 

        global l = loss( model(train[1]), train[1][2] )
        display(plot_nde(true_sol, model, train, ndata=450))

        if i_e % 30 == 0
            η /= 2
            Flux.adjust!(opt_state, η)
        end

        ProgressMeter.next!(prog, showvalues=[(:loss, l)])

    end
end

p_trained = Array(model.p)
time = format(now(), "yyyy-mm-dd-HH-MM")
BSON.@save "params_$(time)_loss_$(l).bson" p_trained

# -------------------------------------


