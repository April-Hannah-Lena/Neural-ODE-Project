using NeuralODEProject
using Documenter

using WGLMakie
using StaticArrays, Statistics, BSON, Random
using DifferentialEquations, OrdinaryDiffEq
using GAIO
using Flux, SciMLSensitivity
using NODEData, ChaoticNDETools

ci = get(ENV, "CI", "false") == "true"
ENV["JULIA_DEBUG"] = Documenter
ENV["GKSwstype"] = "100"

#DocMeta.setdocmeta!(NeuralODEProject, :DocTestSetup, :(using NeuralODEProject); recursive=true)

makedocs(;
    modules=[NeuralODEProject],
    authors="April Herwig <aprillherwig@gmail.com>",
    repo="github.com/April-Hannah-Lena/NeuralODEProject.jl/blob/{commit}{path}#{line}",
    sitename="Neural ODE Project",
    format=Documenter.HTML(prettyurls=ci),
    pages=[
        "Home" => "index.md",
        "Introduction" => "intro.md",
        "Model" => "model.md",
        "Discussion" => "discussion.md",
        "References" => "refs.md"
    ],
)

if ci
    deploydocs(
        repo = "github.com/April-Hannah-Lena/Neural-ODE-Project.git",
        push_preview = true,
        versions = nothing
    )
end
