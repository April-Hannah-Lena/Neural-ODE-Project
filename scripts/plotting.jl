using GLMakie: plot, plot!, Figure, Axis3, Colorbar

function plot_nde(sol, model, train)
    fig, ax, ms = plot(sol.t, Array(model((sol.t,train[1][2])))', label="Neural ODE")
    plot!(ax, sol.t, Array(sol)', label="Training Data")
    fig
end

function plot_unstable_sets(W, W_trained)
    fig = Figure();
    ax = Axis3(fig[1,1], aspect=(1,1.2,1))
    plot!(ax, W, color=(:blue, 0.4), label="True unstable manifold")
    ax2 = Axis3(fig[1,2], aspect=(1,1.2,1))
    plot!(ax2, W_trained, color=(:red, 0.4), label="Predicted unstable manifold")
    fig
end

function plot_symdiff(W, W_trained)
    fig = Figure();
    ax = Axis3(fig[1,1], aspect=(1,1.2,1))
    plot!(ax, setdiff(W, W_trained), color=(:blue, 0.8))
    plot!(ax, setdiff(W_trained, W), color=(:red, 0.8))
    fig
end

function plot_measures(μ, μ_trained)
    vals = [values(μ); values(μ_trained)]
    colorrange = (minimum(vals), maximum(vals))
    fig = Figure();
    ax = Axis3(fig[1,1], aspect=(1,1.2,1))
    ms = plot!(ax, μ, colormap=(:jet, 0.4), colorrange=colorrange, label="True almost invariant sets")
    ax2 = Axis3(fig[1,2], aspect=(1,1.2,1))
    plot!(ax2, μ_trained, colormap=(:jet, 0.4), colorrange=colorrange, label="Predicted almost invariant sets")
    Colorbar(fig[1, 3], ms)
    fig
end
