## Machine Learning Model

We now come to the question of approximating this system based on trajectory data. The method of Chen et. al. suggests considering a recurrent neural network as a "sequence of transformations to a hidden state ``\mathbf{h}_t``" [1]:
```math
\mathbf{h}_{t+1} = \mathbf{h}_t + f(\mathbf{h}_t,\, \theta)
```
where ``t = 0, \ldots, T,\,\ \mathbf{h}_t \in \mathbb{R}^d``, and ``f`` is a single layer of a neural network. They then extend this to an ODE propagating the state of a hidden layer
```math
\frac{d \mathbf{h} (t)}{d t} = f(\mathbf{h}(t),\, t,\, \theta)
```
where ``t \in [0, T]``, and ``f`` is now an entire neural network. 

When considering a chaotic dynamical system, the "chaotic element" arises due to a nonlinearity in the system. This fact is particularly present in Chua's circuit, where the model is _almost_ entirely linear, save for one nonlinearity in the first component. Hence it is reasonable to desire that the neural network should contain a "linear" component, as well as a nonlinear one. Certainly there are many ways to achieve such a network; two possibilities are presented below. 

The first idea which comes to mind is to separate linear and nonlinear components explicitly in the network. More precisely, the neural network should model a function ``nn(x) = W \cdot x + g(x)`` where ``W \in \mathbb{R}^{d \times d}`` and ``g`` can be considered as an explicit nonlinearity. This can be implemented using Flux.jl's built in layers. 

```@example 1
using Flux
```

```@example 1
N_weights = 15

W = Dense(3 => 3)   # dimension = 3

g = Chain(
    Dense(3 => N_weights),
    Dense(N_weights => N_weights, swish),
    Dense(N_weights => 3)
)

nn = Parallel(+, W, g)
```

Another option would be to build in the linearity _implicitly_. One could use a more "typical" neural network with linear input and/or output layers, and use residual network layers to compute a nonlinearity. This allows the network model to "decide" for itself whether the linear component is strictly necessary. A concrete implementation would look like:

```@example 1
nn = Chain(
    Dense(3 => N_weights),

    SkipConnection(Dense(N_weights => N_weights, swish), +),
    SkipConnection(Dense(N_weights => N_weights, swish), +),
    SkipConnection(Dense(N_weights => N_weights, swish), +),

    Dense(N_weights => 3)
)
```

This model presents itself as a more "typical" residual neural network as described in literature. However, one observes by considering the network graph that this model also permits a linear and nonlinear component. Further, this structure benefits from the design of a residual neural network: the "vanishing gradient" problem is reduced [6], and extra layers can be added with reduced fear of overfitting since the model can simply "choose" to ignore unnecessary layers. 

During development of the project, both methods were tested and the implicit form performed better. However, the difference was not large. For the remainder of this report, the implicit method will be used. 


```@example 1
using CairoMakie
```


```@example 1
# for plotting training results
function plot_nde(sol, model, train; ndata=300)
    t = sol.t[1:ndata]
    pred = Array(model((t,train[1][2])))
    tr = Array(sol)
    fig, ax, ms = lines(t, pred[1, 1:ndata], label="Neural ODE dim 1")
    lines!(ax, t, pred[2, 1:ndata], label="Neural ODE dim 2")
    lines!(ax, t, pred[3, 1:ndata], label="Neural ODE dim 3")
    lines!(ax, t, tr[1, 1:ndata], label="Training Data dim 1")
    lines!(ax, t, tr[2, 1:ndata], label="Training Data dim 2")
    lines!(ax, t, tr[3, 1:ndata], label="Training Data dim 3")
    Legend(fig[1,2], ax)
    fig, ax
end
```

```@example 1
using StaticArrays, Statistics
using OrdinaryDiffEq, SciMLSensitivity#, CUDA
using NODEData, ChaoticNDETools
```

```@example 1
# Chua's circuit
function v(u, p, t)
    x, y, z = u
    a, b, m0, m1 = p
    SA{Float32}[ a*(y-m0*x-m1/3.0*x^3), x-y+z, -b*y ]
end

# parameters
p_ode = SA{Float32}[ 18.0, 33.0, -0.2, 0.01 ]
a, b, m0, m1 = p_ode

v(u) = v(u, p_ode, 0f0)

# equilibrium
x₊ = SA{Float32}[ sqrt(-3*m0/m1), 0, -sqrt(-3*m0/m1) ]
x₋ = -x₊

# integration time
t0, t1 = 0f0, 50f0
tspan = (t0, t1)
dt = 1f-2;
```

The data used will be the trajectories from the previous section. These are split into minibatches to both reduce the chance of the training model diverging, as well as to reduce condition number of the gadients for optimization. 


```@example 1
x0 = SA{Float32}[2, 1.5, 6]
prob = ODEProblem(v, x0, (t0, t1), p_ode)
sol = solve(prob, RK4(), saveat=dt, sensealg=InterpolatingAdjoint())
```


```@example 1
train, valid = NODEDataloader(sol, 8; dt=dt, valid_set=0.8, GPU=false#=true=#)
train
```

The parameters of the model are extracted and flattened to a vector $p$ so that the gradient of the loss w.r.t. $p$ can be directly calculated. 


```@example 1
p, re_nn = Flux.destructure(nn)
#p = p |> gpu
neural_ode(u, p, t) = re_nn(p)(u)
neural_ode(u) = neural_ode(u, p, 0f0)

neural_ode_prob = ODEProblem(neural_ode, #=CuArray(x0)=#x0, tspan, p)
model = ChaoticNDE(neural_ode_prob, alg=RK4(), gpu=false#=true=#, sensealg=InterpolatingAdjoint());
```

```@example 1
model(valid[1])
```

The final consideration required before training can be done is the loss function. The most naive loss function may be derived from the shooting method for boundary value problems. One integrates the model for some fixed time ``T``, and compute the difference (in norm) of the model trajectory to the true trajectory data. This technique is extended analogoously to the method of _multiple_ shooting, where the multiple small consecutive trajectories are compared. The resulting differences can be added together to obtain a scalar valued loss function, equivalent (up to a scaling factor) to a mean squared error. 
```math
L(\mathbf{x}, \mathbf{\hat{x}}; \mathbf{p}) = \sum_{i=1}^n \| \mathbf{x}(t_i) - \mathbf{\hat{x}}(t_i; \mathbf{p}) \| ^2
```
where ``\mathbf{x},\ \mathbf{\hat{x}}`` are true and predicted time series, evaluated at times ``t_1 < t_2 < \ldots < t_n = t_1 + T``. The paramteter vector ``p`` is the parameters of the neural network. 
While the mean squared error works quite well, a potential downfall can occur, particularly in periodic systems. In each small trajectory, the errors of the model will compound. However, the mean squared error weighs all of the errors equally. This leads to the potential case that the model is initially incorrect, but later along the trajectory it corrects itself. The model hence learns a fundamentally wrong trajectory, and cannot easily be trained out of this error. This can be seen in the following training attempt:


```@example 1
loss(x, y) = sum(abs2, x - y)

l = mean(valid) do v
    loss( model(v), v[2] )
end

θ = 1f-4
η = 1f-3
opt = Flux.OptimiserChain(Flux.WeightDecay(θ), Flux.RMSProp(η))
opt_state = Flux.setup(opt, model) 

N_epochs = 30
for i_e = 1:N_epochs

    Flux.train!(model, train, opt_state) do m, t, x
        result = m((t,x))
        loss(result, x)
    end 

    global l = mean(valid) do v
        loss( model(v), v[2] )
    end
    
    if i_e % 30 == 0
        global η /= 2
        Flux.adjust!(opt_state, η)
    end

end

l
```

```@example 1
model(valid[1])
```

```@example 1
valid[1][2]
```

```@example 1
fig1, ax1 = plot_nde(sol, model, train, ndata=150)

save("nde1.png", fig1); nothing # hide
```

![](nde1.png)

This error, and the following solution, was discovered in the development of this project. For a dynamical system, we expect the error to compound exponentially. Hence it would seem beneficial to ensure that the model stays as close to the true solution at the _beginning_ of the trajectory as possible. To encourage this, we add an _exponential weight factor_:
```math
L(\mathbf{x}, \mathbf{\hat{x}}; \mathbf{p}) = \sum_{i=1}^n \beta^i \cdot \| \mathbf{x}(t_i) - \mathbf{\hat{x}}(t_i; \mathbf{p}) \| ^2
```
where ``\beta \in (0,1)``. While optimizing parameters, ``\beta`` can be optimized as well. During testing, an optimal value of ``\beta = 0.99`` was observed. 


```@example 1
nn = Chain(
    Dense(3 => N_weights),

    SkipConnection(Dense(N_weights => N_weights, swish), +),
    SkipConnection(Dense(N_weights => N_weights, swish), +),
    SkipConnection(Dense(N_weights => N_weights, swish), +),

    Dense(N_weights => 3)
)

p, re_nn = Flux.destructure(nn)
#p = p |> gpu
neural_ode(u, p, t) = re_nn(p)(u)
neural_ode(u) = neural_ode(u, p, 0f0)

neural_ode_prob = ODEProblem(neural_ode, #=CuArray(x0)=#x0, tspan, p)
model = ChaoticNDE(neural_ode_prob, alg=RK4(), gpu=false#=true=#, sensealg=InterpolatingAdjoint())
```

```@example 1
β = 0.99f0
function loss(x, y, β)
    n = size(x, 2)
    βs = β .^ (1:n)
    sum( abs2, (x - y) .* βs' )
end

l = mean(valid) do v
    loss( model(v), v[2], 1f0 )
end

θ = 1f-4
η = 1f-3
opt = Flux.OptimiserChain(Flux.WeightDecay(θ), Flux.RMSProp(η))
opt_state = Flux.setup(opt, model) 

N_epochs = 30
for i_e = 1:N_epochs

    Flux.train!(model, train, opt_state) do m, t, x
        result = m((t,x))
        loss(result, x, β)
    end 

    global l = mean(valid) do v
        loss( model(v), v[2], 1f0 )
    end
    
    if i_e % 30 == 0
        global η /= 2
        Flux.adjust!(opt_state, η)
    end

end

l
```

```@example 1
model(valid[1])
```

```@example 1
valid[1][2]
```

```@example 1
fig2, ax2 = plot_nde(sol, model, train, ndata=150)

save("nde2.png", fig2); nothing # hide
```

![](nde2.png)


