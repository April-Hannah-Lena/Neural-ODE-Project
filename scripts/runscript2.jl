cd("./scripts")

using StaticArrays, Statistics, Random, ProgressMeter, BSON, ThreadsX, SMTPClient
using Dates: now, format
using OrdinaryDiffEq, SciMLSensitivity, Flux#, CUDA
using ChaoticNDETools, NODEData
#using GAIO

Random.seed!(1234)
#include("plotting.jl")

# ---------------------------------------------------

# Chua's circuit
function v(u, p, t)
    x, y, z = u
    a, b, m0, m1 = p
    SA{Float32}[ a*(y-m0*x-m1/3.0*x^3), x-y+z, -b*y ]
end

p_ode = SA{Float32}[ 18.0, 33.0, -0.2, 0.01 ]
v(u) = v(u, p_ode, 0)

λ_max = 12.5
t0, t1 = 0f0, 40f0
tspan = (t0, t1)
dt = 1f-2

a, b, m0, m1 = p_ode
x0 = SA{Float32}[ 2, 1.5, 6#=8=# ]
x1 = SA{Float32}[-5, 1.2, -4]
x2 = SA{Float32}[-6, 1.2, -6]

prob = ODEProblem(v, x1, (t0, t1), p_ode)
sol1 = solve(prob, RK4(), saveat=dt, sensealg=InterpolatingAdjoint())
_, valid1 = NODEDataloader(sol1, 8; dt=dt, valid_set=0.99, GPU=false)

prob = ODEProblem(v, x2, (t0, t1), p_ode)
sol2 = solve(prob, RK4(), saveat=dt, sensealg=InterpolatingAdjoint())
_, valid2 = NODEDataloader(sol2, 8; dt=dt, valid_set=0.99, GPU=false)

include("chua_script.jl")

# ---------------------------------------------------

TRAIN = true
PLOT = false

weights = 15#[10, 12, 15, 18, 20]
hidden_layers = 3#[1, 2, 3]
epochs = 90
tfins = Float32[3, 5, 8, 10, 12, 15, 18, 20]
β = Float32(0.99)#Float32[0.95, 0.98, 0.99, 1.]
θ = Float32(1f-4)#Float32[1f-2, 1f-3, 1f-4]
η = Float32(1f-3)#Float32[1f-3]

@show time = format(now(), "yyyy-mm-dd-HH-MM")
#=
params = [
    (w, l, e, t, β, θ, η) for
    w in weights,
    l in hidden_layers,
    e in epochs,
    t in tfins,
    β in βs,
    θ in θs,
    η in ηs
]

BSON.@save "./params/params_list_$(time).bson" params
=#
exit_code = false
p = Progress(length(tfins))

losses = progress_map(tfins; mapfun=ThreadsX.map, progress=p) do tfin
    try
        @show tfin
        l, _ = train_node(weights, hidden_layers, epochs, tfin, β, θ, η, TRAIN, PLOT)
        tfin => l
    catch ex
        ex isa InterruptException && rethrow()
        @show ex
        global exit_code = true
        tfin => NaN32
    end
end

losses = Dict(losses...)

BSON.@save "./params/losses_$(time).bson" losses

# ---------------------------------------------------

include("uname+passwd.jl")

opt = SendOptions(
    isSSL = true,
    username = username,
    passwd = passwd
)

body = IOBuffer(
    "Date: $(format(now(), "e, dd u yyyy HH:MM:SS")) +0100\r\n" *
    "From: Me <$(username)>\r\n" *
    "To: $(rcpt)\r\n" *
    "Subject: Benchmark $(exit_code ? "un" : "")successfully finished\r\n" *
    "\r\n" *
    "Go turn off the computer.\r\n"
)

url = "smtps://smtp.gmail.com:465"
resp = send(url, ["<$(rcpt)>"], "<$(username)>", body, opt)
