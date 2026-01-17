# Managing documentation
From within the `docs/` directory, you can build and serve the documentation locally using the following commands:
## Building documentation locally
`julia --project make.jl`
## Serving documentation locally
`julia -e 'using LiveServer; serve(dir="docs/build")'`