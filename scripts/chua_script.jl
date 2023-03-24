function train_node(N_weights, N_hidden_layers, N_epochs, tfin, β, θ, η, TRAIN, PLOT)

    prob = ODEProblem(v, x0, (t0, tfin), p_ode)
    true_sol = solve(prob, Tsit5(), saveat=dt, sensealg=InterpolatingAdjoint())

    PLOT && plot_trajectory(true_sol)

    train, valid = NODEDataloader(true_sol, 8; dt=dt, valid_set=0.8, GPU=false#=true=#)

    neural_net = Chain(
        #BatchNorm(1),
        #u -> 2*u,
        #u -> [u; u.^3],
        Dense(3 => N_weights, swish),
        ntuple(
            _ -> SkipConnection(Dense(N_weights => N_weights, swish), +), 
            N_hidden_layers
        )...,
        Dense(N_weights => 3)
    ) #|> gpu

    p, re_nn = Flux.destructure(neural_net)
    #p = p
    neural_ode(u, p, t) = re_nn(p)(u)
    neural_ode_prob = ODEProblem(neural_ode, #=CuArray(x0)=#x0, tspan, p)
    model = ChaoticNDE(neural_ode_prob, alg=Tsit5(), gpu=false#=true=#, sensealg=InterpolatingAdjoint())

    loss(x, y, β) = sum(enumerate(zip(x, y))) do z
        (i, (xi, yi)) = z
        abs2(xi - yi) * β^i     # give closer times higher weight
    end

    opt = Flux.OptimiserChain(Flux.WeightDecay(θ), Flux.RMSProp(η))
    opt_state = Flux.setup(opt, model) 

    #model.p .= p_trained #|> gpu
    #opt = Flux.OptimiserChain(Flux.WeightDecay(θ*1f-4), Flux.RAdam(η))
    #opt_state = Flux.setup(opt, model) 

    model(train[1])
    grad = Flux.gradient(model) do m
        result = m(train[1])
        loss(result, train[1][2], β)
    end   

    l = loss( model(train[1]), train[1][2], β )

    #prog = Progress(N_epochs)
    if TRAIN
        for i_e = 1:N_epochs

            Flux.train!(model, train, opt_state) do m, t, x
                result = m((t,x))
                loss(result, x, β)
            end 

            global l = loss( model(train[1]), train[1][2], 1f0 )
            PLOT && display(plot_nde(true_sol, model, train, ndata=450))

            if i_e % 30 == 0
                η /= 2
                Flux.adjust!(opt_state, η)
            end

            #ProgressMeter.next!(prog, showvalues=[(:loss, l)])

        end
    end

    l = loss( model(train[1]), train[1][2], 1f0 )
    p_trained = Array(model.p)
    time = format(now(), "yyyy-mm-dd-HH-MM")
    BSON.@save "./params/params_$(time)_loss_$(l).bson" p_trained

    return l
end
