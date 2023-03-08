using WGLMakie: plot, plot!, lines, lines!, Legend, Figure, Axis3, Colorbar

function plot_trajectory(sol)
    fig = Figure();
    ax = Axis3(fig[1,1], aspect=(1,1.2,1), azimuth=pi/10)
    ms = lines!(ax, Array(sol[1, :]), Array(sol[2, :]), Array(sol[3, :]))
    fig
end

function callback(p, l, pred; doplot = true)
    if doplot
        fig = Figure();
        ax = Axis3(fig[1,1], aspect=(1,1.2,1), azimuth=pi/10)
        ms1 = lines!(ax, Array(true_sol[1, :]), Array(true_sol[2, :]), Array(true_sol[3, :]), label="truth")
        ms2 = lines!(ax, Array(pred[1, :]), Array(pred[2, :]), Array(pred[3, :]), label="prediction")
        Legend(fig[1,2], ax)
        display(fig)
    end
    println("Loss: ", l)
    return false
end

function plot_nde(sol, model, train)
    try
        pred = Array(model((sol.t,train[1][2])))
        tr = Array(sol)
        fig, ax, ms = lines(sol.t, pred[1, :], label="Neural ODE dim 1")
        lines!(ax, sol.t, pred[2, :], label="Neural ODE dim 2")
        lines!(ax, sol.t, pred[3, :], label="Neural ODE dim 3")
        lines!(ax, sol.t, tr[1, :], label="Training Data dim 1")
        lines!(ax, sol.t, tr[2, :], label="Training Data dim 2")
        lines!(ax, sol.t, tr[3, :], label="Training Data dim 3")
        Legend(fig[1,2], ax)
        fig
    catch ex
    end
end

function plot_unstable_sets(W, W_trained)
    fig = Figure();
    ax = Axis3(fig[1,1], aspect=(1,1.2,1), azimuth=pi/10)
    plot!(ax, W, color=(:blue, 0.4), label="True unstable manifold")
    ax2 = Axis3(fig[1,2], aspect=(1,1.2,1), azimuth=pi/10)
    plot!(ax2, W_trained, color=(:red, 0.4), label="Predicted unstable manifold")
    fig
end

function plot_symdiff(W, W_trained)
    fig = Figure();
    ax = Axis3(fig[1,1], aspect=(1,1.2,1), azimuth=pi/10)
    plot!(ax, setdiff(W, W_trained), color=(:blue, 0.8))
    plot!(ax, setdiff(W_trained, W), color=(:red, 0.8))
    fig
end

function plot_measures(μ, μ_trained)
    vals = [values(μ); values(μ_trained)]
    colorrange = (minimum(vals), maximum(vals))
    fig = Figure();
    ax = Axis3(fig[1,1], aspect=(1,1.2,1), azimuth=pi/10)
    ms = plot!(ax, μ, colormap=(:jet, 0.4), colorrange=colorrange, label="True almost invariant sets")
    ax2 = Axis3(fig[1,2], aspect=(1,1.2,1), azimuth=pi/10)
    plot!(ax2, μ_trained, colormap=(:jet, 0.4), colorrange=colorrange, label="Predicted almost invariant sets")
    Colorbar(fig[1, 3], ms)
    fig
end
