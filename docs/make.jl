using NeuralODEProject
using Documenter

DocMeta.setdocmeta!(NeuralODEProject, :DocTestSetup, :(using NeuralODEProject); recursive=true)

makedocs(;
    modules=[NeuralODEProject],
    authors="April Herwig <aprillherwig@gmail.com>",
    repo="https://github.com/April-Hannah-Lena/NeuralODEProject.jl/blob/{commit}{path}#{line}",
    sitename="NeuralODEProject.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
