exit_code = false

try # ---------------------------------------------------

using StaticArrays, Random, ProgressMeter, BSON, ThreadsX
using Dates: now, format
using OrdinaryDiffEq, SciMLSensitivity, Flux#, CUDA
using ChaoticNDETools, NODEData
#using GAIO

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

include("chua_script.jl")

# ---------------------------------------------------

TRAIN = true
PLOT = false

weights = [5, 10, 15, 20]
hidden_layers = [1, 2, 3]
epochs = [180]
tfins = Float32[5, 8, 10, 12, 15, 18, 20]
βs = Float32[0.9, 0.95, 0.98, 0.99, 1.]
θs = Float32[1f-3, 1f-4]
ηs = Float32[1f-3]

params = [
    (N_weights, N_hidden_layers, N_epochs, tfin, β, θ, η)
    for N_weights in weights,
        N_hidden_layers in hidden_layers,
        N_epochs in epochs,
        tfin in tfins,
        β in βs,
        θ in θs,
        η in ηs
]

BSON.@save "./params/params_list.bson" params

losses = ThreadsX.collect(

    try
        train_node(N_weights, N_hidden_layers, N_epochs, tfin, β, θ, η, TRAIN, PLOT)
    catch ex
        ex isa InterruptException && rethrow()
        @show ex
        NaN32
    end

    for N_weights in weights,
        N_hidden_layers in hidden_layers,
        N_epochs in epochs,
        tfin in tfins,
        β in βs,
        θ in θs,
        η in ηs

)

BSON.@save "losses.bson" losses

catch ex # ---------------------------------------------------

exit_code = true

finally # ---------------------------------------------------

include("uname+passwd.jl")

opt = SendOptions(
    isSSL = true,
    username = username,
    passwd = passwd
)

body = IOBuffer(
    "Date: $(Dates.format(Dates.now(), "e, dd u yyyy HH:MM:SS")) +0100\r\n" *
    "From: Me <$(username)>\r\n" *
    "To: $(rcpt)\r\n" *
    "Subject: Benchmark $(exit_code ? "un" : "")successfully finished\r\n" *
    "\r\n" *
    "Go turn off the computer.\r\n"
)

url = "smtps://smtp.gmail.com:465"
resp = send(url, ["<$(rcpt)>"], "<$(username)>", body, opt)

end
