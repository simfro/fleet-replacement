using FleetReplacement
using Documenter

DocMeta.setdocmeta!(FleetReplacement, :DocTestSetup, :(using FleetReplacement); recursive=true)

makedocs(;
    modules=[FleetReplacement],
    authors="Simon Frölander <simfro@kth.se> and contributors",
    sitename="FleetReplacement.jl",
    format=Documenter.HTML(;
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
