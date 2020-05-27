module SCoBA

using StaticArrays
using Hungarian
using LinearAlgebra
using Distributions
using Parameters
using DataStructures
using Random
using ArgParse
using Distances
using NearestNeighbors
using GeometryTypes
using TOML

using IterTools
using Reexport

# POMDP stuff
using POMDPs
using MCTS
using POMDPModelTools

# Include submodule files
include("solver/SCoBASolver.jl")
include("domains/SCoBADomains.jl")

@reexport using SCoBA.SCoBASolver
@reexport using SCoBA.SCoBADomains

end # module
