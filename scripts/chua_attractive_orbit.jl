using GAIO

# Chua's circuit
const a, b, m0, m1 = 16.0, 33.0, -0.2, 0.01
v((x,y,z)) = (a*(y-m0*x-m1/3.0*x^3), x-y+z, -b*y)
f(x) = rk4_flow_map(v, x, 0.05, 10)  # 10 steps of RK4 with step size 0.05

center, radius = (0,0,0), (20,20,120)
Q = Box(center, radius)     # domain
P = BoxPartition(Q)
F = BoxMap(:montecarlo, f, Q, no_of_points=800)

S = cover(P, :)
C = chain_recurrent_set(F, S, steps=21)
σ = finite_time_lyapunov_exponents(F, C, T=0.25)

using WGLMakie: plot!, Figure, Axis3, Colorbar

fig = Figure();
ax = Axis3(fig[1,1], aspect=(1,1.2,1))
ms = plot!(ax, σ, colormap=(:jet, 0.4))
Colorbar(fig[1,2], ms)
fig

# we find an unstable manifold surroundng a fixed point
# as well as a stable periodic orbit
