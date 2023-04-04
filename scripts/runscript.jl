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
t0, t1 = 0f0, 1f-1
tspan = (t0, t1)
dt = 1f-2

a, b, m0, m1 = p_ode
x0 = SA{Float32}[ 2, 1.5, 6 ]#-4.9, 7.6, 0.5 ]

include("chua_script.jl")

# ---------------------------------------------------

TRAIN = true
PLOT = false

weights = [10, 12, 15, 18, 20]
hidden_layers = [3]#[1, 2, 3]
epochs = [120]
tfins = Float32[15]#Float32[5, 8, 10, 12, 15, 18, 20]
βs = Float32[0.95, 0.98, 0.99, 1.]
θs = Float32[1f-2, 1f-3, 1f-4]
ηs = Float32[1f-3]

@show time = format(now(), "yyyy-mm-dd-HH-MM")

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

exit_code = false
p = Progress(length(params))

losses = progress_map(params; mapfun=ThreadsX.map, progress=p) do param
    try
        @show param
        l, _ = train_node(param..., TRAIN, PLOT)
        l
    catch ex
        ex isa InterruptException && rethrow()
        @show ex
        global exit_code = true
        NaN32
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
